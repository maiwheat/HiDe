import time  # 原生时间模块，用于计时
import datetime  # 如需使用 datetime，单独导入，不拆分

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from agents.base import BaseLearner

from utils.data import Dataloader_from_numpy
from agents.utils.soft_dtw_cuda import SoftDTW
from agents.utils.functions import euclidean_dist, pod_loss_var, pod_loss_temp
from agents.lwf import loss_fn_kd
from agents.utils.dilate_loss import dilate_loss
import torchaudio.transforms as T
from pytorch_msssim import ssim
import math

from utils.utils import FDFilter, norm_feat, log_transform, AdaptiveFDFilter, verify_ac_energy_distribution


class TFusion(BaseLearner):
    def __init__(self, model, args):
        super(TFusion, self).__init__(model, args)
        self.use_kd = True
        # 如果 args 中没有 lambda_kd_fmap，就默认为 0.0（不做时域蒸馏）
        self.lambda_kd_fmap = getattr(args, 'lambda_kd_fmap', 0.0)  # 时域特征图蒸馏权重
        self.lambda_kd_lwf = args.lambda_kd_lwf           # 预测蒸馏 (LwF) 权重
        # 如果 args 中没有 lambda_kd_fmap_freq，就默认为 0.0（相当于不做频域蒸馏）
        self.lambda_kd_fmap_freq = getattr(args, 'lambda_kd_fmap_freq', 0.0)  # 频域特征图蒸馏权重
        self.alpha = getattr(args, 'alpha', 0.0)            # DILATE 中的 α 参数
        self.data = args.data
        self.energy_ratio = args.energy_ratio

        # Prototype Augmentation
        self.lambda_protoAug = args.lambda_protoAug
        self.prototype = None
        self.class_label = None
        self.adaptive_weight = args.adaptive_weight

        # --- 新增：初始化频域相关组件 ---
        if self.lambda_kd_fmap_freq > 0:
            # self.fd_filter = FDFilter().to(self.device)  # 只需一个实例
            self.mse_loss = nn.MSELoss()
        # -----------------------------
            # 初始化 (在 __init__ 中)
            self.fd_filter = AdaptiveFDFilter(energy_ratio=self.energy_ratio).to(self.device)


    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        meter_loss_total = 0.0
        meter_loss_ce = 0.0
        meter_loss_kd_fmap_freq = 0.0
        meter_loss_ce_low = 0.0
        meter_loss_kd_pred = 0.0
        meter_loss_protoAug = 0.0
        # 记录当前任务的旧类数量
        n_old_classes = self.teacher.head.out_features if self.teacher is not None else 0

        # 选择时域蒸馏度量（如 dilate_loss）
        similarity_metric_dilate_loss = dilate_loss
        similarity_metric_dtw_loss = SoftDTW(use_cuda=(self.device == 'cuda'), gamma=1, normalize=False)
        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            total += y.size(0)

            # —— 确保 y 是一维 LongTensor
            # 有时 Dataloader 返回 y 形状为 [batch_size, 1] 或 [1]，此处统一改为 [batch_size]
            y = y.view(-1).long()

            self.optimizer.zero_grad()

            # 1. CrossEntropy 分类损失
            outputs = self.model(x)  # 假设 outputs.shape == [batch_size, num_classes]
            loss_new = self.criterion(outputs, y)

            # 初始化各子损失
            loss_kd_pred = torch.tensor(0.0, device=self.device)
            loss_kd_fmap_freq = torch.tensor(0.0, device=self.device)
            loss_protoAug = torch.tensor(0.0, device=self.device)
            loss_ce_low = torch.tensor(0.0, device=self.device)

            if self.task_now > 0:
                # 频域蒸馏（仅在 lambda_kd_fmap_freq > 0 时生效）
                if self.lambda_kd_fmap_freq > 0:
                    # 获取特征图
                    student_fmap = self.model.feature_map(x)
                    with torch.no_grad():  # Teacher 不需要梯度，节省显存
                        teacher_fmap = self.teacher.feature_map(x)

                    verify_ac_energy_distribution(student_fmap)
                    verify_ac_energy_distribution(teacher_fmap)
                    # 关键步骤：先处理 Teacher，获得 Teacher 的低频/高频
                    # Teacher 以自己为基准
                    t_low, t_high = self.fd_filter(teacher_fmap, x_ref=None)

                    # 关键步骤：再处理 Student
                    # Student 必须以 Teacher (teacher_fmap) 为基准来切割！确保对齐！
                    s_low, s_high = self.fd_filter(student_fmap, x_ref=teacher_fmap)

                    t_low = t_low.permute(0, 2, 1).contiguous()
                    t_high = t_high.permute(0, 2, 1).contiguous()
                    s_low = s_low.permute(0, 2, 1).contiguous()
                    s_high = s_high.permute(0, 2, 1).contiguous()

                    # 现在的 t_low 和 s_low 包含的是完全相同的频率成分范围
                    loss_kd_f_low = similarity_metric_dilate_loss(s_low, t_low, self.alpha, 1, self.device).mean()
                    # loss_kd_f_low = self.mse_loss(s_low,t_low).mean()
                    loss_kd_f_high = similarity_metric_dtw_loss(s_high, t_high).mean()

                    loss_kd_fmap_freq = loss_kd_f_low + loss_kd_f_high

                    # 1. 提取 Student 低频 (不需 Teacher)
                    s_fmap_curr = self.model.feature_map(x)
                    s_low_curr, s_high_curr = self.fd_filter(s_fmap_curr, x_ref=None)
                    # --- 代码片段 ---
                    # 取第 0 个样本 (Batch 0)，第 0 个通道 (Channel 0)
                    # 转换为 numpy 数组
                    original_wave = s_fmap_curr[0, 0, :].detach().cpu().numpy()
                    low_freq_wave = s_low_curr[0, 0, :].detach().cpu().numpy()
                    diff_wave = (s_fmap_curr - s_low_curr)[0, 0, :].detach().cpu().numpy()  # 高频残差

                    # plt.figure(figsize=(10, 4))
                    # plt.title(f"Feature Map Comparison (Sample 0, Channel 0)")
                    # plt.plot(original_wave, label='Original (s_fmap)', color='black', alpha=0.5)
                    # plt.plot(low_freq_wave, label='Low Freq (s_low)', color='red', linewidth=2)
                    # plt.plot(diff_wave, label='Difference (High Freq)', color='blue', alpha=0.3) # 可选：看高频
                    # plt.legend()
                    # plt.grid(True)
                    # plt.show()  # 如果是在服务器跑，用 plt.savefig('compare.png')

                    # 2. 池化 + 分类
                    s_dc = s_fmap_curr.mean(dim=-1, keepdim=True)
                    s_low_feat_no_dc = s_low_curr - s_dc
                    s_low_feat = torch.mean(s_low_feat_no_dc, dim=-1)  # GAP
                    # print(s_low_feat)

                    logits_low = self.model.head(s_low_feat)

                    # 3. 计算 CE Loss
                    loss_ce_low = self.criterion(logits_low, y)
                    # # 1. 计算两个 Loss 的绝对差值
                    # diff = torch.abs(loss_new - loss_ce_low).item()
                    #
                    # # 2. 计算两个 Loss 的相对比率 (Ratio)
                    # # 如果 ratio 接近 1.0，说明两者几乎一样
                    # ratio = loss_ce_low.item() / (loss_new.item() + 1e-8)
                    #
                    # # 3. 计算输入给 Head 的特征的余弦相似度
                    # # 验证：全频特征 vs 低频特征 的方向是否重合
                    # # (需要先获取全频特征的池化向量)
                    # s_fmap_full = self.model.feature_map(x)  # 或者复用之前的
                    # s_feat_full = torch.mean(s_fmap_full, dim=-1)  # GAP
                    # # s_low_feat 已经在上面计算过了
                    #
                    # cos_sim = torch.nn.functional.cosine_similarity(s_feat_full, s_low_feat, dim=1).mean().item()
                    #
                    # # 4. 打印调试信息 (每隔 10 个 batch 打一次，避免刷屏)
                    # if batch_id % 10 == 0:
                    #     print(f"\n[Debug Batch {batch_id}]")
                    #     print(f"  CE Loss (Full): {loss_new.item():.4f}")
                    #     print(f"  CE Loss (Low) : {loss_ce_low.item():.4f}")
                    #     print(f"  Diff (Abs)    : {diff:.4f}")
                    #     print(f"  Ratio (Low/Full): {ratio:.2f}")
                    #     print(f"  Feature CosSim: {cos_sim:.4f}")
                    #
                    #     # 判定逻辑
                    #     if cos_sim > 0.99 and diff < 0.01:
                    #         print("  ⚠️ 警告: 低频特征与全频特征过于相似，辅助损失可能失效！")
                    #     else:
                    #         print("  ✅ 状态: 辅助损失正在提供差异化梯度。")
                    # # 调试打印
                    # energy_full = torch.norm(s_fmap_curr)
                    # energy_low = torch.norm(s_low_curr)
                    # print(f"Full Energy: {energy_full.item():.2f}, Low Energy: {energy_low.item():.2f}, Ratio: {energy_low / energy_full:.4f}")
                else:
                    loss_kd_fmap_freq = torch.tensor(0.0, device=self.device)
                    loss_ce_low = torch.tensor(0.0, device=self.device)

                # —— 2.3: LwF 预测蒸馏（仅在 lambda_kd_lwf > 0 时生效）
                if self.lambda_kd_lwf > 0:
                    old_logits = outputs[:, :self.teacher.head.out_features]
                    with torch.no_grad():
                        teacher_logits = self.teacher(x)
                    loss_kd_pred = loss_fn_kd(old_logits, teacher_logits)
                else:
                    loss_kd_pred = torch.tensor(0.0, device=self.device)

                # —— 2.4: 所有 KD 子损失加权求和
                loss_kd = (
                        self.lambda_kd_lwf * loss_kd_pred +
                        self.lambda_kd_fmap_freq * loss_kd_fmap_freq
                )
                # —— 3: ProtoAug (升级版)
                # if self.lambda_protoAug > 0:
                #     proto_aug = []
                #     proto_aug_label = []
                #
                #     # 获取当前已有的所有旧类标签
                #     available_classes = list(self.class_prototypes.keys())
                #
                #     for _ in range(self.args.batch_size):
                #         # A. 随机选一个旧类
                #         c = np.random.choice(available_classes)
                #
                #         # B. 获取该类的原型组和半径组
                #         protos = self.class_prototypes[c]['prototypes']
                #         radii = self.class_prototypes[c]['radii']
                #
                #         # C. 随机选该类下的一个原型索引
                #         p_idx = np.random.randint(len(protos))
                #
                #         # D. 取出原型和对应的半径
                #         selected_proto = protos[p_idx]
                #         selected_radius = radii[p_idx]
                #
                #         # E. 生成增强样本: Prototype + Gaussian Noise * Radius
                #         # 确保半径不过大，可以加一个缩放系数 self.radius_scale (如 1.0)
                #         noise = np.random.normal(0, 1, self.args.feature_dim)
                #         temp = selected_proto + noise * selected_radius
                #
                #         proto_aug.append(temp)
                #         proto_aug_label.append(c)
                #
                #     proto_aug = torch.from_numpy(np.array(proto_aug, dtype=np.float32)).to(self.device)
                #     proto_aug_label = torch.tensor(proto_aug_label, device=self.device, dtype=torch.long)
                #
                #     soft_feat_aug = self.model.head(proto_aug)
                #     loss_protoAug = self.criterion(soft_feat_aug, proto_aug_label)
                #     loss_protoAug = self.lambda_protoAug * loss_protoAug
                # —— 3: ProtoAug
                if self.lambda_protoAug > 0:
                    proto_aug = []
                    proto_aug_label = []
                    index = list(range(n_old_classes))
                    for _ in range(self.args.batch_size):
                        np.random.shuffle(index)
                        temp = (
                                self.prototype[index[0]]
                                + np.random.normal(0, 1, self.args.feature_dim) * self.radius
                        )
                        proto_aug.append(temp)
                        proto_aug_label.append(self.class_label[index[0]])

                    proto_aug = torch.from_numpy(
                        np.asarray(proto_aug, dtype=np.float32)
                    ).to(self.device)
                    proto_aug_label = torch.tensor(
                        proto_aug_label, device=self.device, dtype=torch.long
                    )
                    soft_feat_aug = self.model.head(proto_aug)
                    loss_protoAug = self.criterion(soft_feat_aug, proto_aug_label)
                    loss_protoAug = self.lambda_protoAug * loss_protoAug
                else:
                    loss_protoAug = torch.tensor(0.0, device=self.device)
            else:
                # self.task_now == 0：跳过所有 KD、ProtoAug
                loss_kd = torch.tensor(0.0, device=self.device)
                loss_kd_pred = torch.tensor(0.0, device=self.device)
                loss_kd_fmap_freq = torch.tensor(0.0, device=self.device)
                loss_protoAug = torch.tensor(0.0, device=self.device)
                loss_ce_low = torch.tensor(0.0, device=self.device)

            # 1. 计算自适应因子 (Adaptive Factor)
            if self.adaptive_weight:
                factor_new = 1.0 / (self.task_now + 1)
                factor_old = 1.0 - factor_new
            else:
                factor_new = 1.0
                factor_old = 1.0



            # 2. 计算各项对梯度的【真实贡献】
            # CE 部分
            contrib_ce = factor_new * loss_new
            contrib_ce_low =  factor_new * loss_ce_low * 0
            # KD 部分 (先乘 lambda, 再乘 adaptive factor)
            contrib_freq = factor_old * self.lambda_kd_fmap_freq * loss_kd_fmap_freq
            contrib_lwf = factor_old * self.lambda_kd_lwf * loss_kd_pred
            contrib_proto = factor_old * self.lambda_protoAug * loss_protoAug

            # 3. 总损失 (各项贡献之和)
            step_loss = contrib_ce + contrib_freq + contrib_lwf + contrib_proto + contrib_ce_low

            step_loss.backward()
            self.optimizer_step(epoch=epoch)

            # ==========================================
            # C. 统计累加 (Accumulate)
            # ==========================================
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

            meter_loss_total += step_loss.item()
            meter_loss_ce += contrib_ce.item()  # 累加的是加权后的
            meter_loss_kd_fmap_freq += contrib_freq.item()
            meter_loss_kd_pred += contrib_lwf.item()
            meter_loss_protoAug += contrib_proto.item()
            meter_loss_ce_low += contrib_ce_low.item()

            # ==========================================
            # D. 计算 Epoch 平均值
            # ==========================================
        num_batches = batch_id + 1
        epoch_acc = 100.0 * (correct / total)

        return (
            meter_loss_total / num_batches,
            meter_loss_ce / num_batches,
            meter_loss_kd_fmap_freq / num_batches,
            meter_loss_kd_pred / num_batches,
            meter_loss_protoAug / num_batches,
            meter_loss_ce_low / num_batches
        ), epoch_acc

    # def epoch_loss_printer(self, epoch, acc, loss):
    #     """
    #     打印每个 epoch 的详细损失与准确率
    #     loss: tuple -> (总损失, CE, 时域KD, LwF, ProtoAug, 频域KD)
    #     """
    #     print(
    #         'Epoch {}/{}: Accuracy = {:.2f}%, Total_loss = {:.4f}, '
    #         'CE = {:.4f}, DILATE = {:.4f}, LwF = {:.4f}, protoAug_loss = {:.4f}, freqKD_loss = {:.4f}'.format(
    #             epoch + 1,
    #             self.epochs,
    #             acc,
    #             loss[0],
    #             loss[1],
    #             self.lambda_kd_fmap * loss[2],
    #             self.lambda_kd_lwf * loss[3],
    #             loss[4],
    #             self.lambda_kd_fmap_freq * loss[5]
    #         )
    #     )
    def epoch_loss_printer(self, epoch, acc, loss):
        """
        现在打印的就是真实的贡献值，直接相加等于 Total
        loss: (Total, CE, Freq, LwF, Proto)
        """
        print(
            'Epoch {}/{}: Accuracy = {:.2f}%, Total = {:.4f} | '
            'CE = {:.4f}, FreqKD = {:.4f}, LwF = {:.4f}, Proto = {:.4f}, ce_low = {:.4f}'.format(
                epoch + 1, self.epochs, acc,
                loss[0],  # Total
                loss[1],  # CE (Weighted)
                loss[2],  # FreqKD (Weighted)
                loss[3],  # LwF (Weighted)
                loss[4],  # Proto (Weighted)
                loss[5]
            )
        )
    def after_task(self, x_train, y_train):
        super(TFusion, self).after_task(x_train, y_train)
        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        if self.lambda_protoAug > 0:
            self.protoSave(model=self.model, loader=dataloader, current_task=self.task_now)


    def protoSave(self, model, loader, current_task):
        """
        在每个任务结束时，根据当前 task 中提取的特征，保存每个类别的 prototype（均值）以及对应的 radius。
        """
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                feature = model.feature(x.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(y.cpu().numpy())
                    features.append(feature.cpu().numpy())

        labels = np.concatenate(labels, axis=0)    # [batch_size * num_batches]
        features = np.concatenate(features, axis=0)  # [batch_size * num_batches, feature_dim]
        labels_set = np.unique(labels)               # 本任务的所有类别标签
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []

        for item in labels_set:
            idx = np.where(labels == item)[0]
            class_label.append(item)
            feature_classwise = features[idx]
            prototype.append(np.mean(feature_classwise, axis=0))

            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(f"Initial prototype radius: {self.radius:.4f}")
        else:
            # 将新任务的 prototype 与已有 prototype 拼接起来
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)

        model.train()
