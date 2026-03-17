# -*- coding: UTF-8 -*-
import numpy as np
import torch
import warnings
from collections import defaultdict
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from agents.utils.functions import copy_params_dict, zerolike_params_dict


class EWC(BaseLearner):
    """
    Modified from https://avalanche-api.continualai.org/en/v0.2.0/_modules/avalanche/training/plugins/ewc.html#EWCPlugin
    """
    def __init__(self, model, args, keep_importance_data=False):
        super(EWC, self).__init__(model, args)
        # ewc_lambda：EWC 的正则化系数，决定旧任务的重要性 (args.lambda_impt)。
        self.ewc_lambda = args.lambda_impt
        # mode：EWC 计算方式，可选 "separate"（累积多个任务的惩罚项）或 "online"（仅保留上一个任务的惩罚项）。
        self.mode = args.ewc_mode
        # decay_factor：用于 online 模式的衰减因子。
        self.decay_factor = 0.5
        # separate 模式
        # 策略：
        #
        # 每个任务的 Fisher 信息矩阵都会 单独存储，不会丢弃旧任务的信息。
        # 计算 EWC 罚项时，累加所有旧任务的正则项，即所有任务的重要性都会影响参数更新。
        # 适用场景：
        #
        # 任务之间完全独立，没有强关联性。
        # 例如，在**非连续任务（task-based continual learning）**场景下，每个任务的目标不同，不能忽视旧任务的信息。
        if self.mode == "separate":
            self.keep_importance_data = True
        # online 模式
        # 策略：
        #
        # 只存储最近一个任务的 Fisher 信息矩阵，并使用 decay_factor 进行更新。
        # 历史任务的重要性信息会被指数衰减，不会完全保留。
        # 计算 EWC 罚项时，仅使用最近任务的 Fisher 信息矩阵。
        # 适用场景：
        #
        # 任务之间连续演化，新任务与旧任务有较强的相似性（如时间序列）。
        # 例如，在流式学习（streaming learning）场景下，数据分布是逐步变化的，而不是完全独立的任务。
        else:
            self.keep_importance_data = keep_importance_data
        # defaultdict(list) 的作用是 当访问一个不存在的键时，自动为该键创建一个空列表，这样就不需要手动检查键是否存在，直接 append 数据即可。
        # saved_params：存储每个任务训练完成时的模型参数。
        self.saved_params = defaultdict(list)
        # importances：存储每个任务的重要性（Fisher 信息矩阵）。
        self.importances = defaultdict(list)

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss = 0
        epoch_ewc_term = 0
   
        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            # 若是第一个任务 (task_now == 0)，仅计算交叉熵损失。
            if self.task_now == 0:
                step_loss = self.criterion(outputs, y.long())
            # 若是后续任务 (task_now > 0)，增加 EWC 罚项 ewc_penalty。
            else:
                ewc_penalty = self.ewc_penalty()
                step_loss = self.criterion(outputs, y.long()) + ewc_penalty
                epoch_ewc_term += ewc_penalty

            step_loss.backward()
            self.optimizer_step(epoch=epoch)

            epoch_loss += step_loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)
        epoch_ewc_term /= (batch_id + 1)

        return (epoch_loss, epoch_ewc_term), epoch_acc

    def epoch_loss_printer(self, epoch, acc, loss):
        print('Epoch {}/{}: Accuracy = {}, Loss = {}, Avg_EWC_term = {}, '.format(epoch + 1, self.epochs,
                                                                                     acc, loss[0], loss[1]))

    def after_task(self, x_train, y_train):
        """
        Calculate Fisher
        :return:
        """
        super(EWC, self).after_task(x_train, y_train)
        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        # compute_importances(dataloader) 计算 Fisher 信息矩阵。
        importances = self.compute_importances(dataloader)
        # update_importances(importances) 存储重要性值。
        self.update_importances(importances)
        # self.saved_params[self.task_now] = copy_params_dict(self.model) 记录当前任务结束时的模型参数。
        self.saved_params[self.task_now] = copy_params_dict(self.model)
        # clear previous parameter values
        if self.task_now > 0 and (not self.keep_importance_data):
            del self.saved_params[self.task_now - 1]

    def ewc_penalty(self, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """
        exp_counter = self.task_now
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(self.device)

        if self.mode == "separate":
            # Loop through all the old tasks
            for experience in range(exp_counter):
                for (k1, cur_param), (k2, saved_param), (k3, imp) in zip(
                        self.model.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience],
                ):
                    assert k1 == k2 == k3, "Error: keys do not match "
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    if saved_param.size() == torch.Size():
                        pass
                    else:
                        # 当模型动态扩展（如新增神经元）时，只约束旧参数部分，新参数不受限。
                        n_units = saved_param.shape[0]
                        cur_param = cur_param[:n_units]
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()

        elif self.mode == "online":
            # Only use the penalty calculated from the last task
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    self.model.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp],
            ):
                n_units = saved_param.shape[0]
                cur_param = cur_param[:n_units]
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")
        # λ 是 ewc_lambda 超参数
        loss_penalty = self.ewc_lambda * penalty

        return loss_penalty
    # 计算 Fisher 信息矩阵 (compute_importances)
    def compute_importances(self, dataloader):
        """
        Compute EWC importance matrix for each parameter
        """

        self.model.eval()
        self.criterion = torch.nn.CrossEntropyLoss()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if self.device == "cuda":
            for module in self.model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(self.model)

        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y.long())
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances
            ):
                assert k1 == k2
                if p.grad is not None:
                    grad = p.grad.data
                    # .clone()：防止修改原始 p.grad.data，避免影响计算图。
                    # .detach()：确保 imp 不影响梯度计算，节省内存，避免 RuntimeError。
                    imp += p.grad.data.clone().detach().pow(2)

        # update normalized fisher of current task
        max_fisher = max([torch.max(m) for _, m in importances])
        min_fisher = min([torch.min(m) for _, m in importances])

        # average over mini batch length
        for i in range(len(importances)):
            _, imp = importances[i]
            imp = (imp - min_fisher) / (max_fisher - min_fisher + 1e-32)
            importances[i][-1] = imp

        # # average over mini batch length
        # for _, imp in importances:
        #     imp /= float(len(dataloader))

        return importances

    @torch.no_grad()
    # 更新 Fisher 信息 (update_importances)
    def update_importances(self, importances):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or self.task_now == 0:
            self.importances[self.task_now] = importances

        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in zip(
                    self.importances[self.task_now - 1], importances
            ):
                assert k1 == k2, "Error in importance computation."
                self.importances[self.task_now].append(
                    (k1, (self.decay_factor * old_imp + curr_imp))
                )

            # clear previous parameter importances
            if self.task_now > 0 and (not self.keep_importance_data):
                del self.importances[self.task_now - 1]

        else:
            raise ValueError("Wrong EWC mode.")




