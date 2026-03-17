
import time
import torch
import torch.nn as nn
import numpy as np
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from agents.utils.soft_dtw_cuda import SoftDTW
from agents.utils.dilate_loss import dilate_loss
from agents.lwf import loss_fn_kd
from utils.utils import TriBandFDFilter, plot_freq_decomposition_multichannel_bcl
from agents.utils.functions import euclidean_dist

class Dkfd(BaseLearner):
    """
    基于人体活动识别的频域解藕蒸馏的时间序列类增量学习方法
    """

    def __init__(self, model, args):
        super(Dkfd, self).__init__(model, args)
        self.use_kd = True
        self.lambda_kd_fmap = getattr(args, 'lambda_kd_fmap', 0.0)
        self.lambda_kd_lwf = args.lambda_kd_lwf
        self.lambda_kd_fmap_freq = getattr(args, 'lambda_kd_fmap_freq', 0.0)
        self.ratio = getattr(args, 'ratio', 10.0)
        self.inject_lambda = getattr(args, 'inject_lambda', 1.0)
        self.data = args.data
        # self.energy_ratio = args.energy_ratio

        # Prototype Augmentation
        self.lambda_protoAug = args.lambda_protoAug
        self.prototype = None
        self.class_label = None
        self.adaptive_weight = args.adaptive_weight



        # 初始化频域相关组件
        if self.lambda_kd_fmap_freq > 0:
            self.mse_loss = nn.MSELoss()

            # 这里的 TriBandFDFilter 保持您之前的引入
            self.fd_filter = TriBandFDFilter()
            self.plot_freq_decomposition = plot_freq_decomposition_multichannel_bcl

            # 【新增】共享分类器 (Teacher Head) 的容器，运行时再赋值
        self.shared_head = None

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        meter_loss_total = 0.0
        meter_loss_ce = 0.0
        meter_loss_kd_fmap_freq = 0.0
        meter_loss_kd_fmap_freq_dc = 0.0
        meter_loss_kd_fmap_freq_low = 0.0
        meter_loss_kd_fmap_freq_high = 0.0
        meter_loss_ce_low = 0.0
        meter_loss_kd_pred = 0.0
        meter_loss_protoAug = 0.0
        meter_loss_ce_high = 0.0


        n_old_classes = self.teacher.head.out_features if self.teacher is not None else 0

        similarity_metric_dilate_loss = dilate_loss
        similarity_metric_dtw_loss = SoftDTW(use_cuda=(self.device == 'cuda'), gamma=1, normalize=False)
        similarity_metric_euclidean_dist = euclidean_dist
        self.model.train()

        # 【新增】准备共享分类器 (只做一次)
        if self.task_now > 0:
            self.shared_head = self.teacher.head
            for param in self.shared_head.parameters():
                param.requires_grad = False  # 冻结，只做尺子

        for batch_id, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.view(-1).long().to(self.device)
            total += y.size(0)
            self.optimizer.zero_grad()

            # 1. 基础前向传播
            # 获取特征图 [B, C, L]
            student_fmap = self.model.feature_map(x)
            # 【关键】获取 GAP 后的向量 [B, C]，用于分类和 ProtoAug
            student_feat_vec = torch.mean(student_fmap, dim=-1)

            # CrossEntropy 分类损失
            outputs = self.model.head(student_feat_vec)
            loss_new = self.criterion(outputs, y)

            # 初始化各子损失
            loss_kd_pred = torch.tensor(0.0, device=self.device)
            loss_kd_fmap_freq = torch.tensor(0.0, device=self.device)
            loss_protoAug = torch.tensor(0.0, device=self.device)
            loss_ce_low = torch.tensor(0.0, device=self.device)
            loss_ce_high = torch.tensor(0.0, device=self.device)
            loss_kd_fmap_freq_low = torch.tensor(0.0, device=self.device)
            loss_kd_fmap_freq_high = torch.tensor(0.0, device=self.device)


            # 频域解耦
            s_dc, s_low, s_high = self.fd_filter(student_fmap)

            if self.task_now > 0:
                # 获取 Teacher 信息 (No Grad)
                with torch.no_grad():
                    teacher_fmap = self.teacher.feature_map(x)
                    t_dc, t_low, t_high = self.fd_filter(teacher_fmap)
                    # teacher_feat_vec = torch.mean(teacher_fmap, dim=-1)  # Teacher 向量
                    teacher_logits = self.teacher(x)

                # —— 2.1: 频域蒸馏 & Outlier Check
                if self.lambda_kd_fmap_freq > 0:
                    s_low = s_low + s_dc
                    t_low = t_low + t_dc

                    s_fmap_low_norm = torch.nn.functional.layer_norm(s_low, s_low.shape[1:])
                    s_fmap_high_norm = torch.nn.functional.layer_norm(s_high, s_high.shape[1:])

                    t_fmap_low_norm = torch.nn.functional.layer_norm(t_low, t_low.shape[1:])
                    t_fmap_high_norm = torch.nn.functional.layer_norm(t_high, t_high.shape[1:])

                    t_high_trans = t_fmap_high_norm.permute(0, 2, 1).contiguous()
                    t_low_trans = t_fmap_low_norm.permute(0, 2, 1).contiguous()

                    s_high_trans = s_fmap_high_norm.permute(0, 2, 1).contiguous()
                    s_low_trans = s_fmap_low_norm.permute(0, 2, 1).contiguous()

                    loss_kd_f_low = similarity_metric_dtw_loss(s_low_trans, t_low_trans).mean()

                    loss_kd_f_high = similarity_metric_dtw_loss(s_high_trans, t_high_trans).mean()

                    loss_kd_fmap_freq_low = loss_kd_f_low
                    loss_kd_fmap_freq_high = loss_kd_f_high * self.ratio
                    loss_kd_fmap_freq =  loss_kd_f_low + loss_kd_f_high  * self.ratio



                # —— 2.3: LwF 预测蒸馏
                if self.lambda_kd_lwf > 0:
                    old_logits = outputs[:, :n_old_classes]
                    loss_kd_pred = loss_fn_kd(old_logits, teacher_logits)


                # ---------------------------
                # 3. ProtoAug: Adversarial Injection (SAAI)
                # ---------------------------
                    # ---------------------------
                    # 3. ProtoAug: (Targeted Adversarial Injection)
                    # ---------------------------
                if self.lambda_protoAug > 0 and self.prototype is not None:

                    # 1. 基础准备
                    old_protos = torch.from_numpy(self.prototype).float().to(self.device)
                    old_labels = torch.from_numpy(self.class_label).long().to(self.device)
                    n_total_old = old_protos.shape[0]
                    current_batch_size = student_feat_vec.shape[0]

                    # 2. 随机采样旧原型 (保持覆盖率)
                    # 我们先随机选出这一批要回放的旧原型
                    # indices: [B]
                    indices = torch.randint(0, n_total_old, (current_batch_size,)).to(self.device)

                    target_old_protos = old_protos[indices]  # [B, C]
                    target_old_labels = old_labels[indices]  # [B]

                    # -------------------------------------------------------
                    # 【核心修正】寻找最相似的新样本 (Find Nearest Enemy)
                    # -------------------------------------------------------
                    # 我们不跟随机的新样本做差，而是跟"最像的"做差

                    # 归一化
                    old_norm = torch.nn.functional.normalize(target_old_protos, dim=1)  # [B, C]
                    new_norm = torch.nn.functional.normalize(student_feat_vec, dim=1)  # [B, C]

                    # 计算相似度矩阵 [B, B]
                    # matrix[i, j] = 第 i 个旧原型 vs 第 j 个新样本
                    sim_matrix = torch.matmul(old_norm, new_norm.t())

                    # 找到每个旧原型的"最佳匹配" (Most Confusing New Sample)
                    # best_new_indices: [B] -> 每个旧原型对应的"最像新样本"的索引
                    _, best_new_indices = torch.max(sim_matrix, dim=1)

                    # 取出这些"对手"样本
                    # matched_new_feats: [B, C]
                    matched_new_feats = student_feat_vec[best_new_indices]

                    # -------------------------------------------------------
                    # 计算对抗方向 & 注入
                    # -------------------------------------------------------
                    # 此时，diff 指向的是决策边界最危险的方向 (Geodesic Direction)
                    diff = matched_new_feats.detach() - target_old_protos

                    # 归一化方向
                    norm_diff = torch.nn.functional.normalize(diff, p=2, dim=1)

                    # 注入步长 (Epsilon)
                    # 0.1 表示沿着最危险的方向，向新类逼近 10%
                    epsilon = self.inject_lambda

                    proto_aug = target_old_protos + epsilon * diff
                    proto_aug_label = target_old_labels

                    # 4. 计算 Loss
                    soft_feat_aug = self.model.head(proto_aug)
                    loss_protoAug = self.criterion(soft_feat_aug, proto_aug_label) * self.lambda_protoAug

                else:
                    loss_protoAug = torch.tensor(0.0, device=self.device)

            if self.adaptive_weight:
                factor_new = 1.0 / (self.task_now + 1)
                factor_old = 1.0 - factor_new
            else:
                factor_new = 1.0
                factor_old = 1.0

            # 2. 计算各项贡献
            contrib_ce = factor_new * loss_new
            contrib_ce_low = factor_new * loss_ce_low * 0
            contrib_ce_high = factor_new * loss_ce_high * 0

            contrib_freq_low = factor_old * self.lambda_kd_fmap_freq * loss_kd_fmap_freq_low
            contrib_freq_high = factor_old * self.lambda_kd_fmap_freq * loss_kd_fmap_freq_high
            contrib_freq = factor_old * self.lambda_kd_fmap_freq * loss_kd_fmap_freq
            contrib_lwf = factor_old * self.lambda_kd_lwf * loss_kd_pred
            contrib_proto = factor_old * loss_protoAug # lambda 已经在里面乘过了

            # 【关键】Shared Loss 也需要乘权重 (复用 lambda_protoAug 或 1.0)

            # 3. 总损失 (加入 Shared)
            step_loss = contrib_ce + contrib_freq + contrib_lwf + contrib_proto + \
                        contrib_ce_low + contrib_ce_high

            step_loss.backward()
            self.optimizer_step(epoch=epoch)

            # C. 统计累加
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

            meter_loss_total += step_loss.item()
            meter_loss_ce += contrib_ce.item()
            meter_loss_kd_fmap_freq += contrib_freq.item()
            meter_loss_kd_pred += contrib_lwf.item()
            meter_loss_protoAug += contrib_proto.item()
            meter_loss_ce_low += contrib_ce_low.item()
            meter_loss_ce_high += contrib_ce_high.item()

            meter_loss_kd_fmap_freq_low += contrib_freq_low.item()
            meter_loss_kd_fmap_freq_high += contrib_freq_high.item()

        num_batches = batch_id + 1
        epoch_acc = 100.0 * (correct / total)

        # 返回值 (保持原结构，把 shared 插在合适位置或通过 printer 打印)
        return (
            meter_loss_total / num_batches,
            meter_loss_ce / num_batches,
            meter_loss_kd_fmap_freq / num_batches,
            meter_loss_kd_pred / num_batches,
            meter_loss_protoAug / num_batches,
            meter_loss_ce_low / num_batches,
            meter_loss_ce_high / num_batches,
            meter_loss_kd_fmap_freq_low / num_batches,
            meter_loss_kd_fmap_freq_high / num_batches
        ), epoch_acc

    def epoch_loss_printer(self, epoch, acc, loss):
        # 记得更新 printer 接收参数
        print(
            'Epoch {}/{}: Acc={:.2f}%, Tot={:.4f} | '
            'CE={:.4f}, Freq={:.4f}, LwF={:.4f}, Proto={:.4f}, CE_low={:.4f}, CE_high={:.4f},meter_loss_kd_fmap_freq_low={:.4f},meter_loss_kd_fmap_freq_high={:.4f}'.format(
                epoch + 1, self.epochs, acc,
                loss[0], loss[1], loss[2], loss[3], loss[4],loss[5],loss[6], loss[7], loss[8]  # loss[7]是shared
            )
        )

    # after_task 和 protoSave 保持不变，可以直接复用原代码
    def after_task(self, x_train, y_train):
        super(Dkfd, self).after_task(x_train, y_train)
        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        if self.lambda_protoAug > 0:
            self.protoSave(model=self.model, loader=dataloader, current_task=self.task_now)

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                # 【修正4】统一使用 feature_map + mean，防止 model.feature 不存在
                # 假设输出 [B, C, T]
                fmap = model.feature_map(x)
                if fmap.dim() == 3:
                    feature = torch.mean(fmap, dim=-1)  # GAP -> [B, C]
                else:
                    feature = fmap

                features.append(feature.cpu().numpy())
                labels.append(y.numpy())

        # 合并
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        labels_set = np.unique(labels)

        prototype = []
        radius = []
        class_label = []

        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)

            # 取出该类所有特征
            feature_classwise = features[index]

            # 计算均值
            prototype.append(np.mean(feature_classwise, axis=0))

            if current_task == 0:
                # 计算迹作为半径 (用于辅助，Mixup 不强依赖这个)
                cov = np.cov(feature_classwise.T)
                # 处理 scalar 维度的特例
                tr = np.trace(cov) if cov.ndim > 0 else 0
                radius.append(tr / feature_classwise.shape[1])

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius)) if len(radius) > 0 else 0
            self.prototype = np.array(prototype)
            self.class_label = np.array(class_label)
            print(f"Task 0 Radius: {self.radius}")
        else:
            self.prototype = np.concatenate((np.array(prototype), self.prototype), axis=0)
            self.class_label = np.concatenate((np.array(class_label), self.class_label), axis=0)

        print(f"[ProtoSave] Total prototypes stored: {self.prototype.shape[0]}")
        model.train()