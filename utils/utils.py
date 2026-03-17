# -*- coding: UTF-8 -*-
import torch
import random
import numpy as np
import sys
import os
import pickle
import psutil
import torch.nn as nn
from matplotlib import pyplot as plt


# 重写logger，使用print时都会记录到日志当中
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
        https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, path, patience=5, mode='max', verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_best = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, model):
        if self.mode == 'max':
            score = val_metric
        else:
            score = -val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        # score是一个负数，如果损失函数减小，score会变大，假如score一直小于best_score+delta，那么说明模型没有进步，就会触发早停
        # delta: 最小改善幅度（如果改进小于 delta，不算提升）
        # 最终模型就是“得分拐点”对应的模型
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0
    # 保存验证集上表现最好的模型，每次找到更好的模型，都会覆盖之前的模型，保存到 self.path
    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_metric_best:.6f} --> {val_metric:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_best = val_metric


# 定义一个函数，用于设置随机种子
def seed_fixer(seed):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置Python内置的随机模块的随机种子
    random.seed(seed)
    # 如果CUDA可用
    if torch.cuda.is_available():
        # 设置CUDA的随机种子
        torch.cuda.manual_seed(seed)
        # 设置所有CUDA设备的随机种子
        torch.cuda.manual_seed_all(seed)
        # 禁用CUDA的自动优化
        torch.backends.cudnn.benchmark = False
        # 设置CUDA的确定性
        torch.backends.cudnn.deterministic = True


# IO
def save_pickle(file, path):
    filehandler = open(path, "wb")
    pickle.dump(file, filehandler)
    filehandler.close()

# 加载源数据，pkl的数据文件
def load_pickle(path):
    file = open(path, 'rb')
    result = pickle.load(file)
    file.close()
    return result

def check_ram_usage():
    """
    Compute the RAM usage of the current process.
        Returns:
            mem (float): Memory occupation in Megabytes
    """

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)

    return mem


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def ohe_label(label_tensor, dim, device="cpu"):
    # Returns one-hot-encoding of input label tensor
    # label_tensor: tensor of data's label, sth like (5000,)
    # dim: number of classes so far
    n_labels = label_tensor.size(0)
    zero_tensor = torch.zeros((n_labels, dim), device=device, dtype=torch.float)
    return zero_tensor.scatter_(1, label_tensor.reshape((n_labels, 1)), 1)


class BinaryCrossEntropy():
    def __init__(self, dim, device):
        self.dim = dim
        self.device = device
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def __call__(self, logits, labels):
        targets = ohe_label(labels, dim=self.dim, device=self.device)
        loss = self.criterion(logits, targets)
        return loss


class BinaryCrossEntropywithLogits():
    def __init__(self, dim, device):
        self.dim = dim
        self.device = device
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def __call__(self, logits, target_logits):
        targets = torch.sigmoid(target_logits)
        loss = self.criterion(logits, targets)
        return loss





def list_subtraction(l1, l2):
    """
    return l1-l2
    """
    return [item for item in l1 if item not in l2]


def nonzero_indices(bool_mask_tensor):
    # Returns tensor which contains indices of nonzero elements in bool_mask_tensor
    return bool_mask_tensor.nonzero(as_tuple=True)[0]


def euclidean_distance(u, v):
    euclidean_distance_ = (u - v).pow(2).sum(1)
    return euclidean_distance_


def mini_batch_deep_features(model, total_x, num):
    """
        Compute deep features with mini-batches.
            Args:
                model (object): neural network.
                total_x (tensor): data tensor.
                num (int): number of data.
            Returns
                deep_features (tensor): deep feature representation of data tensor.
    """
    is_train = False
    if model.training:
        is_train = True
        model.eval()
    if hasattr(model, "feature"):
        model_has_feature_extractor = True
    else:
        model_has_feature_extractor = False
        # delete the last fully connected layer
        modules = list(model.children())[:-1]
        # make feature extractor
        model_features = torch.nn.Sequential(*modules)

    with torch.no_grad():
        bs = 64
        num_itr = num // bs + int(num % bs > 0)
        sid = 0
        deep_features_list = []
        for i in range(num_itr):
            eid = sid + bs if i != num_itr - 1 else num
            batch_x = total_x[sid: eid]

            if model_has_feature_extractor:
                batch_deep_features_ = model.feature(batch_x)
            else:
                batch_deep_features_ = torch.squeeze(model_features(batch_x))

            deep_features_list.append(batch_deep_features_.reshape((batch_x.size(0), -1)))
            sid = eid
        if num_itr == 1:
            deep_features_ = deep_features_list[0]
        else:
            deep_features_ = torch.cat(deep_features_list, 0)
    if is_train:
        model.train()
    return deep_features_


class FDFilter(nn.Module):
    def __init__(self):
        super(FDFilter, self).__init__()

    def forward(self, x):
        """
        x: [B, C, T]
        """
        B, C, T = x.shape

        # rFFT over time dimension
        freq_domain = torch.fft.rfft(x, dim=-1)  # [B, C, F]
        F = freq_domain.shape[-1]

        # dynamic binary masks
        half = F // 2

        # shape = [1, 1, F], broadcast to all batches/channels
        low_mask = torch.zeros(F, device=x.device)
        high_mask = torch.zeros(F, device=x.device)
        low_mask[:half] = 1
        high_mask[half:] = 1

        low_mask = low_mask.view(1, 1, F)
        high_mask = high_mask.view(1, 1, F)

        # frequency filtering
        low_fd = freq_domain * low_mask
        high_fd = freq_domain * high_mask

        # inverse FFT
        low = torch.fft.irfft(low_fd, n=T, dim=-1).real
        high = torch.fft.irfft(high_fd, n=T, dim=-1).real

        return low, high
def norm_feat(x):
    # x: [B, D, L]
    norms = torch.norm(x, p=2, dim=-1, keepdim=True)  # normalize along time
    return x / norms.clamp(min=1e-6)

def log_transform(x, eps=1e-6):
    # x: [B, D, L]
    return torch.sign(x) * torch.log1p(torch.abs(x) + eps)


class AdaptiveFDFilter(nn.Module):
    """
    基于能量分布的自适应高低频分割 (支持 KD 对齐版)
    """

    def __init__(self, energy_ratio=0.8):
        super(AdaptiveFDFilter, self).__init__()
        self.energy_ratio = energy_ratio  # 低频包含的能量占比 (推荐 0.85 - 0.9)

    def forward(self, x, x_ref=None):
        """
        x: 需要滤波的目标特征 (通常是 Student)
        x_ref: 参考特征 (通常是 Teacher)。
               如果提供了 x_ref，则根据 x_ref 计算分割点，并应用到 x。
               如果没有提供，则根据 x 自身计算分割点。
        """
        # 1. 确定用来计算分割点的基准数据
        target = x_ref if x_ref is not None else x

        B, C, T = target.shape

        # 2. FFT 计算基准数据的频谱
        freq_domain_target = torch.fft.rfft(target, dim=-1)
        F = freq_domain_target.shape[-1]

        # 3. 计算功率谱密度 (PSD) 并找到分割点
        # 这里的 dim=(0,1) 意味着对当前 Batch 所有样本和通道取平均，保证 Mask 统一
        psd = torch.abs(freq_domain_target) ** 2
        psd_mean = psd.mean(dim=(0, 1))

        cumulative_energy = torch.cumsum(psd_mean, dim=0)
        total_energy = cumulative_energy[-1]
        cumulative_ratio = cumulative_energy / (total_energy + 1e-8)  # 防止除零

        # 找到 split_idx
        split_idx = torch.searchsorted(cumulative_ratio, self.energy_ratio).item()
        split_idx = max(1, min(split_idx, F - 1))  # 边界保护
        # split_idx = split_idx + 1
        # 4. 创建掩码 (Mask)
        low_mask = torch.zeros(F, device=x.device)
        high_mask = torch.zeros(F, device=x.device)
        low_mask[:split_idx] = 1
        high_mask[split_idx:] = 1

        # 广播形状以适配 [B, C, F]
        low_mask = low_mask.view(1, 1, F)
        high_mask = high_mask.view(1, 1, F)

        # 5. 对输入 x 进行滤波 (注意：这里是对 x 进行 FFT)
        freq_domain_x = torch.fft.rfft(x, dim=-1)

        low_fd = freq_domain_x * low_mask
        high_fd = freq_domain_x * high_mask

        # 6. 逆 FFT 回时域
        low = torch.fft.irfft(low_fd, n=T, dim=-1)
        high = torch.fft.irfft(high_fd, n=T, dim=-1)

        return low, high


# class AdaptiveFDFilter(nn.Module):
#     """
#     基于能量分布的自适应高低频分割 (AC能量占比版 - 忽略DC影响)
#     """
#
#     def __init__(self, energy_ratio=0.85):
#         # 注意：因为排除了DC，剩下的都是变化量，
#         # 所以 ratio 建议设高一点 (比如 0.8 ~ 0.95)，表示保留 80%-95% 的动态变化趋势
#         super(AdaptiveFDFilter, self).__init__()
#         self.energy_ratio = energy_ratio
#
#     def forward(self, x, x_ref=None):
#         """
#         x: 需要滤波的目标特征 (Student)
#         x_ref: 参考特征 (Teacher)
#         """
#         # 1. 确定用来计算分割点的基准数据
#         target = x_ref if x_ref is not None else x
#
#         B, C, T = target.shape
#
#         # 2. FFT 计算基准数据的频谱
#         freq_domain_target = torch.fft.rfft(target, dim=-1)
#         F = freq_domain_target.shape[-1]
#
#         # 3. 计算功率谱密度 (PSD)
#         psd = torch.abs(freq_domain_target) ** 2
#         psd_mean = psd.mean(dim=(0, 1))  # [F]
#
#         # ================== 修改核心开始 ==================
#         # 策略：计算 Ratio 时忽略 DC (Index 0)，只计算 AC 分量 (Index 1~) 的分布
#
#         if F > 1:
#             # 取出 AC 部分 (跳过第 0 个点)
#             ac_energy = psd_mean[1:]
#
#             # 计算 AC 的累积能量
#             cumulative_ac = torch.cumsum(ac_energy, dim=0)
#             total_ac = cumulative_ac[-1] + 1e-8  # 防止除零
#
#             # 计算 AC 占比
#             cumulative_ratio = cumulative_ac / total_ac
#
#             # 在 AC 数组里寻找满足 energy_ratio 的分割点
#             # searchsorted 返回的是 ac_energy 里的索引 (0 对应原始的 1)
#             ac_split_idx = torch.searchsorted(cumulative_ratio, self.energy_ratio).item()
#
#             # 映射回原始索引：
#             # +1: 因为 ac_energy 是从原始 index 1 开始的 (补回 DC 的位置)
#             # +1: 因为 searchsorted 找到的是刚好达标的点，切片 [:idx] 不包含该点，所以要往后延一位
#             split_idx = ac_split_idx + 1 + 1
#         else:
#             # 如果 F=1 (极罕见)，只能全取
#             split_idx = 1
#
#         # 边界保护：
#         # max(2, ...): 强制至少保留 DC(0) 和 基波(1)。
#         #              因为如果只留 DC，就没有时序信息了，不仅失去了低频意义，还容易导致梯度消失。
#         # min(..., F): 防止索引越界
#         split_idx = max(2, min(split_idx, F))
#         # ================== 修改核心结束 ==================
#
#         # 4. 创建掩码 (Mask)
#         low_mask = torch.zeros(F, device=x.device)
#         high_mask = torch.zeros(F, device=x.device)
#
#         # 这里的切片 [:split_idx] 一定包含了 Index 0 (DC)，因为 split_idx >= 2
#         low_mask[:split_idx] = 1
#         high_mask[split_idx:] = 1
#
#         # 广播形状以适配 [B, C, F]
#         low_mask = low_mask.view(1, 1, F)
#         high_mask = high_mask.view(1, 1, F)
#
#         # 5. 对输入 x 进行滤波
#         freq_domain_x = torch.fft.rfft(x, dim=-1)
#
#         low_fd = freq_domain_x * low_mask
#         high_fd = freq_domain_x * high_mask
#
#         # 6. 逆 FFT 回时域
#         low = torch.fft.irfft(low_fd, n=T, dim=-1)
#         high = torch.fft.irfft(high_fd, n=T, dim=-1)
#
#         return low, high


def verify_ac_energy_distribution(s_fmap):
    """
    验证除去 DC 后，能量是否集中在低频 (动态适配任意长度版)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. 计算频谱
    # s_fmap: [B, C, L]
    freq_domain = torch.fft.rfft(s_fmap, dim=-1)

    # 2. 计算功率
    power = torch.abs(freq_domain) ** 2

    # 3. 平均化
    avg_power = power.mean(dim=(0, 1))  # [F]

    # 4. 剔除 DC 分量
    dc_energy = avg_power[0].item()
    ac_energy_dist = avg_power[1:]  # 取 Index 1 到最后

    # 归一化 AC 能量
    total_ac = ac_energy_dist.sum() + 1e-8
    ac_ratios = (ac_energy_dist / total_ac).detach().cpu().numpy()

    # # 5. 打印统计信息
    # print(f"\n[Spectrum Analysis] Time Length L={s_fmap.shape[-1]}")
    # print(f"  DC Energy (Mean): {dc_energy:.2f}")
    # print(f"  Total AC Energy : {total_ac.item():.2f}")
    #
    # # 6. 【修复点】动态生成 X 轴索引
    # num_ac_points = len(ac_ratios)
    # indices = np.arange(1, num_ac_points + 1)  # [1, 2, ..., 9]
    #
    # # 动态生成颜色：前 1/3 为红色(低频)，其余为蓝色(高频)
    # colors = ['red'] * int(num_ac_points * 0.3) + ['blue'] * (num_ac_points - int(num_ac_points * 0.3))
    # # 补齐颜色长度防止越界 (因为 int 向下取整可能少一个)
    # if len(colors) < num_ac_points:
    #     colors += ['blue'] * (num_ac_points - len(colors))
    #
    # # 7. 画图
    # plt.figure(figsize=(10, 4))
    # bars = plt.bar(indices, ac_ratios, color=colors, alpha=0.7)
    #
    # plt.title(f"AC Energy Distribution (L={s_fmap.shape[-1]}, Excl. DC)")
    # plt.xlabel("Frequency Index")
    # plt.ylabel("Ratio of AC Energy")
    # plt.xticks(indices)
    # plt.grid(axis='y', alpha=0.3)
    #
    # # 标数值
    # for bar in bars:
    #     yval = bar.get_height()
    #     # 只标注比较高的柱子，防止文字重叠
    #     if yval > 0.05:
    #         plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval * 100:.1f}%", ha='center', va='bottom',
    #                  fontsize=8)

    # plt.show()
    # plt.savefig(f'spectrum_verify_L{s_fmap.shape[-1]}.png')
# --- 调用方式 ---
# 在 train_epoch 获取 feature_map 后调用一次即可
# s_fmap_curr = self.model.feature_map(x)
# verify_ac_energy_distribution(s_fmap_curr)


def plot_freq_decomposition(x_raw, x_low, x_high, epoch, batch_idx, sample_idx=0):
    """
    绘制单个样本所有通道叠加后的时域分解图
    适配输入形状: [B, L, C]
    操作：将 [L, C] 维度的数据在 C 维度求和 -> [L]
    """
    # 1. 数据处理：选择样本 -> 沿通道维度求和 -> 转 Numpy
    # x_raw shape: [B, L, C]
    # x_raw[sample_idx] shape: [L, C]
    # dim=-1 代表沿着最后一个维度 (C) 求和

    sum_raw = torch.sum(x_raw[sample_idx], dim=-1).cpu().detach().numpy()
    sum_low = torch.sum(x_low[sample_idx], dim=-1).cpu().detach().numpy()
    sum_high = torch.sum(x_high[sample_idx], dim=-1).cpu().detach().numpy()

    # 2. 绘图布局：上下两张图
    plt.figure(figsize=(12, 8))

    # --- 子图 1: 宏观趋势对比 (总Raw vs 总Low) ---
    plt.subplot(2, 1, 1)
    # 原始总信号
    plt.plot(sum_raw, color='black', alpha=0.6, label='Sum Raw (Total Input)', linewidth=1.5)
    # 低频总信号
    plt.plot(sum_low, color='cyan', alpha=0.9, label='Sum Low (Total Trend)', linewidth=2, linestyle='--')

    plt.title(f'Epoch {epoch} | Batch {batch_idx} | Sample {sample_idx}\nMacro Trend: Sum of All Channels')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Amplitude Sum')

    # --- 子图 2: 高频细节 (总High) ---
    plt.subplot(2, 1, 2)
    plt.plot(sum_high, color='red', alpha=0.8, label='Sum High (Total Detail/Noise)', linewidth=1)

    # 画一条 0 刻度线作为参考
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)

    plt.title('Micro Detail: Sum of High Freq Components')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude Sum')

    plt.tight_layout()

    # 显示图像 (阻塞模式)
    plt.show()
    plt.close()


def plot_freq_decomposition_multichannel(x_raw, x_low, x_high, epoch, batch_idx, sample_idx=0, max_channels=None):
    """
    绘制单个样本【每个通道】的时域分解图 (Per-Channel Decomposition)
    布局：每个通道占一行，左边画 Raw+Low，右边画 High。

    Args:
        x_raw: 原始数据 [B, L, C]
        x_low: 低频数据 [B, L, C]
        x_high: 高频数据 [B, L, C]
        epoch: 当前 Epoch
        batch_idx: 当前 Batch 索引
        sample_idx: Batch 中的样本索引 (默认第0个)
        max_channels: 如果通道太多，限制最多画几个 (默认 None，即全部画)
    """
    # 1. 数据准备：取样本 -> 转 CPU -> 转 Numpy -> [L, C]
    # 假设输入是 [B, L, C] 格式
    raw_np = x_raw[sample_idx].cpu().detach().numpy()
    low_np = x_low[sample_idx].cpu().detach().numpy()
    high_np = x_high[sample_idx].cpu().detach().numpy()

    seq_len, n_channels = raw_np.shape

    # 限制最大通道数（防止 HAR 数据集通道过多图太长）
    if max_channels is not None and n_channels > max_channels:
        print(f"Note: Channels {n_channels} > {max_channels}, showing first {max_channels} only.")
        n_channels = max_channels

    # 2. 创建画布: N 行 2 列
    # figsize 高度随通道数自动调整
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 2.5 * n_channels), sharex=True)

    # 兼容处理 n_channels=1 的情况 (matplotlib 返回的 axes 不是数组)
    if n_channels == 1:
        axes = np.array([axes])

    # 总标题
    plt.suptitle(f'Epoch {epoch} | Batch {batch_idx} | Sample {sample_idx}\nPer-Channel Frequency Decomposition',
                 fontsize=15, y=1.01, fontweight='bold')

    # 3. 循环绘制每个通道
    for c in range(n_channels):
        # --- 左图：Raw vs Low (宏观趋势) ---
        ax_left = axes[c, 0]
        # 原始信号 (半透明黑色)
        ax_left.plot(raw_np[:, c], color='black', alpha=0.4, linewidth=1.0, label='Raw')
        # 低频信号 (青色虚线，加粗)
        ax_left.plot(low_np[:, c], color='darkcyan', alpha=0.9, linewidth=1.5, linestyle='--', label='Low (Trend)')

        ax_left.set_ylabel(f'Ch {c} Amp', fontweight='bold')
        ax_left.grid(True, alpha=0.3)

        # 只在第一行显示图例和标题，避免杂乱
        if c == 0:
            ax_left.set_title('Macro View: Raw vs Low (Trend)', fontsize=12, fontweight='bold', color='navy')
            ax_left.legend(loc='upper right', fontsize='small', framealpha=0.8)

        # --- 右图：High (微观细节) ---
        ax_right = axes[c, 1]
        # 高频信号 (红色)
        ax_right.plot(high_np[:, c], color='crimson', alpha=0.8, linewidth=1.0, label='High (Detail)')
        # 0 刻度参考线
        ax_right.axhline(0, color='gray', linestyle=':', linewidth=0.5)

        ax_right.grid(True, alpha=0.3)

        # 只在第一行显示标题
        if c == 0:
            ax_right.set_title('Micro View: High (Detail/Noise)', fontsize=12, fontweight='bold', color='darkred')
            ax_right.legend(loc='upper right', fontsize='small', framealpha=0.8)

        # 只在最后一行显示 X 轴标签
        if c == n_channels - 1:
            ax_left.set_xlabel('Time Steps', fontsize=10)
            ax_right.set_xlabel('Time Steps', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_freq_decomposition_multichannel_bcl(x_raw, x_low, x_high, epoch, batch_idx, sample_idx=0, max_channels=10):
    """
    绘制单个样本【每个通道】的时域分解图 (Per-Channel Decomposition)
    布局：每个通道占一行，左边画 Raw+Low，右边画 High。

    Args:
        x_raw:  原始数据 [B, C, L]  <-- 修改：Channel 在中间
        x_low:  低频数据 [B, C, L]
        x_high: 高频数据 [B, C, L]
        epoch: 当前 Epoch
        batch_idx: 当前 Batch 索引
        sample_idx: Batch 中的样本索引 (默认第0个)
        max_channels: 如果通道太多，限制最多画几个 (默认 None，即全部画)
    """
    # 1. 数据准备：取样本 -> 转 CPU -> 转 Numpy -> [C, L]
    # 假设输入是 [B, C, L] 格式，取完 sample_idx 后变成 [C, L]
    raw_np = x_raw[sample_idx].cpu().detach().numpy()
    low_np = x_low[sample_idx].cpu().detach().numpy()
    high_np = x_high[sample_idx].cpu().detach().numpy()

    # 【修改点 1】: 获取维度时，顺序变为 (Channel, Length)
    n_channels, seq_len = raw_np.shape

    # 限制最大通道数（防止通道过多图太长）
    if max_channels is not None and n_channels > max_channels:
        print(f"Note: Channels {n_channels} > {max_channels}, showing first {max_channels} only.")
        n_channels = max_channels

    # 2. 创建画布: N 行 2 列
    # figsize 高度随通道数自动调整
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 2.5 * n_channels), sharex=True)

    # 兼容处理 n_channels=1 的情况 (matplotlib 返回的 axes 不是数组)
    if n_channels == 1:
        axes = np.array([axes])

    # 总标题
    plt.suptitle(f'Epoch {epoch} | Batch {batch_idx} | Sample {sample_idx}\nPer-Channel Frequency Decomposition',
                 fontsize=15, y=1.01, fontweight='bold')

    # 3. 循环绘制每个通道
    for c in range(n_channels):
        # --- 左图：Raw vs Low (宏观趋势) ---
        ax_left = axes[c, 0]

        # 【修改点 2】: 数据切片从 [:, c] 改为 [c, :] (因为现在是 [C, L])
        # 原始信号 (半透明黑色)
        ax_left.plot(raw_np[c, :], color='black', alpha=0.4, linewidth=1.0, label='Raw')
        # 低频信号 (青色虚线，加粗)
        ax_left.plot(low_np[c, :], color='darkcyan', alpha=0.9, linewidth=1.5, linestyle='--', label='Low (Trend)')

        ax_left.set_ylabel(f'Ch {c} Amp', fontweight='bold')
        ax_left.grid(True, alpha=0.3)

        # 只在第一行显示图例和标题，避免杂乱
        if c == 0:
            ax_left.set_title('Macro View: Raw vs Low (Trend)', fontsize=12, fontweight='bold', color='navy')
            ax_left.legend(loc='upper right', fontsize='small', framealpha=0.8)

        # --- 右图：High (微观细节) ---
        ax_right = axes[c, 1]

        # 【修改点 3】: 数据切片改为 [c, :]
        # 高频信号 (红色)
        ax_right.plot(high_np[c, :], color='crimson', alpha=0.8, linewidth=1.0, label='High (Detail)')
        # 0 刻度参考线
        ax_right.axhline(0, color='gray', linestyle=':', linewidth=0.5)

        ax_right.grid(True, alpha=0.3)

        # 只在第一行显示标题
        if c == 0:
            ax_right.set_title('Micro View: High (Detail/Noise)', fontsize=12, fontweight='bold', color='darkred')
            ax_right.legend(loc='upper right', fontsize='small', framealpha=0.8)

        # 只在最后一行显示 X 轴标签
        if c == n_channels - 1:
            ax_left.set_xlabel('Time Steps', fontsize=10)
            ax_right.set_xlabel('Time Steps', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.close()


# class TriBandFDFilter(nn.Module):
#     """
#     三频段滤波器：提取 DC、低频 (AC前半段)、高频 (AC后半段)
#     """
#
#     def __init__(self):
#         super(TriBandFDFilter, self).__init__()
#
#     def forward(self, x):
#         """
#         x: [B, C, T]
#         return: dc, low, high (全部为时域数据，形状与 x 相同)
#         """
#         B, C, T = x.shape
#
#         # 1. FFT 变换到频域
#         # rfft 得到的频谱长度 F = T // 2 + 1
#         freq_domain_x = torch.fft.rfft(x, dim=-1)
#         F = freq_domain_x.shape[-1]
#
#         # 2. 定义分割点
#         # DC 是 index 0
#         # 剩下的频率范围是 1 ~ F-1 (共 F-1 个点)
#         # 我们取剩下部分的中间点作为低高频分割
#         # 也就是 "前半段" 和 "后半段"
#         ac_len = F - 1
#         split_idx = 1 + (ac_len // 2)
#
#         # 3. 创建掩码 (Mask)
#         dc_mask = torch.zeros(F, device=x.device)
#         low_mask = torch.zeros(F, device=x.device)
#         high_mask = torch.zeros(F, device=x.device)
#
#         # 设置掩码区域
#         dc_mask[0] = 1              # 仅保留 DC
#         low_mask[1:split_idx] = 1   # DC 之后到中间
#         high_mask[split_idx:] = 1   # 中间到最后
#
#         # 广播形状以适配 [1, 1, F] 方便与 [B, C, F] 相乘
#         dc_mask = dc_mask.view(1, 1, F)
#         low_mask = low_mask.view(1, 1, F)
#         high_mask = high_mask.view(1, 1, F)
#
#         # 4. 频域滤波
#         dc_fd = freq_domain_x * dc_mask
#         low_fd = freq_domain_x * low_mask
#         high_fd = freq_domain_x * high_mask
#
#         # 5. 逆 FFT 回时域 (IRFFT)
#         # 注意：必须要指定 n=T，否则偶数长度的信号可能会丢失最后一个点
#         dc = torch.fft.irfft(dc_fd, n=T, dim=-1)
#         low = torch.fft.irfft(low_fd, n=T, dim=-1)
#         high = torch.fft.irfft(high_fd, n=T, dim=-1)
#
#         return dc, low, high


import torch
import torch.nn as nn


# class TriBandFDFilter(nn.Module):
#     """
#     实例感知的三频段滤波器 (Instance-aware TriBandFDFilter)
#     基于功率谱密度 (PSD) 动态划分低频和高频。
#     """
#
#     def __init__(self, energy_threshold=0.7):
#         """
#         Args:
#             energy_threshold (float): 定义低频分量应包含的总能量比例 (0~1)。
#                                       例如 0.75 表示低频带将自动延伸，直到包含 75% 的 AC 能量。
#         """
#         super(TriBandFDFilter, self).__init__()
#         self.energy_threshold = energy_threshold
#
#     def forward(self, x):
#         """
#         x: [B, C, T]
#         return: dc, low, high (全部为时域数据，形状与 x 相同)
#         """
#         B, C, T = x.shape
#
#         # 1. FFT 变换到频域
#         # rfft 得到的频谱长度 F = T // 2 + 1
#         # freq_domain_x: [B, C, F]
#         freq_domain_x = torch.fft.rfft(x, dim=-1)
#         F = freq_domain_x.shape[-1]
#
#         # 2. 计算功率谱 (Power Spectrum) / 能量
#         # Energy = |Amplitude|^2
#         energy = torch.abs(freq_domain_x) ** 2  # [B, C, F]
#
#         # 3. 分离 AC 分量 (去掉 DC，因为 DC 总是被单独提取)
#         # 我们只根据 AC 部分的能量分布来决定 Low/High 的边界
#         ac_energy = energy[:, :, 1:]  # [B, C, F-1]
#
#         # 计算 AC 总能量
#         total_ac_energy = torch.sum(ac_energy, dim=-1, keepdim=True) + 1e-8  # 防止除0
#
#         # 计算累积能量占比 (Cumulative Distribution)
#         # cumsum: [B, C, F-1]
#         cumsum_energy = torch.cumsum(ac_energy, dim=-1)
#         energy_ratio = cumsum_energy / total_ac_energy
#
#         # 4. 生成动态掩码 (Dynamic Masks)
#         # 只要累积能量占比小于阈值，就视为“主要成分/低频”
#         # [B, C, F-1]
#         is_low = (energy_ratio <= self.energy_threshold).float()
#         is_high = 1.0 - is_low
#
#         # 5. 构建完整的滤波器掩码
#         # DC Mask: 只有第0个位置是1
#         dc_mask = torch.zeros_like(energy)
#         dc_mask[:, :, 0] = 1.0
#
#         # Low Mask: 第0位置是0，后面跟随计算出的 is_low
#         low_mask = torch.zeros_like(energy)
#         low_mask[:, :, 1:] = is_low
#
#         # High Mask: 第0位置是0，后面跟随计算出的 is_high
#         high_mask = torch.zeros_like(energy)
#         high_mask[:, :, 1:] = is_high
#
#         # 6. 应用滤波
#         # 注意：这里是点乘，利用了广播机制自动匹配 [B, C, F]
#         dc_fd = freq_domain_x * dc_mask
#         low_fd = freq_domain_x * low_mask
#         high_fd = freq_domain_x * high_mask
#
#         # 7. 逆 FFT 回时域
#         dc = torch.fft.irfft(dc_fd, n=T, dim=-1)
#         low = torch.fft.irfft(low_fd, n=T, dim=-1)
#         high = torch.fft.irfft(high_fd, n=T, dim=-1)
#
#         return dc, low, high

# class TriBandFDFilter(nn.Module):
#     """
#     三频段滤波器：提取 DC、低频 (AC前半段)、高频 (AC后半段)
#     """
#
#     def __init__(self):
#         super(TriBandFDFilter, self).__init__()
#
#     def forward(self, x):
#         """
#         x: [B, C, T]
#         return: dc, low, high (全部为时域数据，形状与 x 相同)
#         """
#         B, C, T = x.shape
#
#         # 1. FFT 变换到频域
#         # rfft 得到的频谱长度 F = T // 2 + 1
#         freq_domain_x = torch.fft.rfft(x, dim=-1)
#         F = freq_domain_x.shape[-1]
#
#         # 2. 定义分割点
#         # DC 是 index 0
#         # 剩下的频率范围是 1 ~ F-1 (共 F-1 个点)
#         # 我们取剩下部分的中间点作为低高频分割
#         # 也就是 "前半段" 和 "后半段"
#         ac_len = F - 1
#         split_idx = 1 + (ac_len // 2)
#
#         # 3. 创建掩码 (Mask)
#         dc_mask = torch.zeros(F, device=x.device)
#         low_mask = torch.zeros(F, device=x.device)
#         high_mask = torch.zeros(F, device=x.device)
#
#         # 设置掩码区域
#         dc_mask[0] = 1              # 仅保留 DC
#         low_mask[1:split_idx] = 1   # DC 之后到中间
#         high_mask[split_idx:] = 1   # 中间到最后
#
#         # # 索引 1 : 仅保留第一个频率分量作为 "Low"
#         # if F > 1:
#         #     low_mask[1] = 1
#         #
#         #     # 索引 2~End : 剩下的所有频率归为 "High"
#         # if F > 2:
#         #     high_mask[2:] = 1
#         #     # ---------------------
#
#         # 广播形状以适配 [1, 1, F] 方便与 [B, C, F] 相乘
#         dc_mask = dc_mask.view(1, 1, F)
#         low_mask = low_mask.view(1, 1, F)
#         high_mask = high_mask.view(1, 1, F)
#
#         # 4. 频域滤波
#         dc_fd = freq_domain_x * dc_mask
#         low_fd = freq_domain_x * low_mask
#         high_fd = freq_domain_x * high_mask
#
#         # 5. 逆 FFT 回时域 (IRFFT)
#         # 注意：必须要指定 n=T，否则偶数长度的信号可能会丢失最后一个点
#         dc = torch.fft.irfft(dc_fd, n=T, dim=-1)
#         low = torch.fft.irfft(low_fd, n=T, dim=-1)
#         high = torch.fft.irfft(high_fd, n=T, dim=-1)
#
#         return dc, low, high


import torch
import torch.nn as nn


class TriBandFDFilter(nn.Module):
    """
    基于最大类间谱方差 (MISV) 的无参数自适应滤波器。
    原理类似于 Otsu 算法：寻找一个频率点，使得低频带和高频带的能量方差最大。
    """

    def __init__(self):
        super(TriBandFDFilter, self).__init__()

    def forward(self, x):
        B, C, T = x.shape
        freq_domain_x = torch.fft.rfft(x, dim=-1)
        F = freq_domain_x.shape[-1]

        # 1. 计算功率谱能量
        energy = torch.abs(freq_domain_x) ** 2  # [B, C, F]
        ac_energy = energy[:, :, 1:]  # 去掉 DC, [B, C, F-1]

        N = F - 1
        # 准备索引和权重
        # 计算全局平均能量
        total_energy = torch.sum(ac_energy, dim=-1, keepdim=True) + 1e-8

        # 2. 预计算累积概率和累积均值 (用于快速计算类间方差)
        # p_i: 每个频点的能量占比
        p_i = ac_energy / total_energy
        omega = torch.cumsum(p_i, dim=-1)  # 累积能量占比 (ω)

        # mu: 累积均值
        idx_grid = torch.arange(1, N + 1, device=x.device).float().view(1, 1, -1)
        mu_i = torch.cumsum(p_i * idx_grid, dim=-1)
        mu_total = mu_i[:, :, -1:]

        # 3. 计算类间方差 (Inter-band Variance)
        # 公式: sigma^2(f) = [mu_total * omega(f) - mu(f)]^2 / [omega(f) * (1 - omega(f))]
        numerator = (mu_total * omega - mu_i) ** 2
        denominator = omega * (1.0 - omega) + 1e-8
        inter_variance = numerator / denominator

        # 4. 寻找最大方差对应的频率索引 (最优分割点)
        knee_idx = torch.argmax(inter_variance, dim=-1, keepdim=True)

        # 5. 生成掩码
        idx_range = torch.arange(N, device=x.device).view(1, 1, -1)
        is_low = (idx_range <= knee_idx).float()

        dc_mask = torch.zeros_like(energy)
        dc_mask[:, :, 0] = 1.0
        low_mask = torch.zeros_like(energy)
        low_mask[:, :, 1:] = is_low
        high_mask = torch.zeros_like(energy)
        high_mask[:, :, 1:] = 1.0 - is_low

        # 6. 逆变换回时域
        dc = torch.fft.irfft(freq_domain_x * dc_mask, n=T, dim=-1)
        low = torch.fft.irfft(freq_domain_x * low_mask, n=T, dim=-1)
        high = torch.fft.irfft(freq_domain_x * high_mask, n=T, dim=-1)

        return dc, low, high

class TriBandFDFilterNLD(nn.Module):
    """
    三频段滤波器：提取 DC、低频 (AC前半段)、高频 (AC后半段)
    适配输入形状: [N, L, D] (Batch, Length, Dimension)
    """

    def __init__(self):
        super(TriBandFDFilterNLD, self).__init__()

    def forward(self, x):
        """
        x: [N, L, D]  <-- 注意这里 L 是第 1 维
        return: dc, low, high (形状与 x 相同 [N, L, D])
        """
        N, L, D = x.shape

        # 1. FFT 变换到频域
        # 指定 dim=1，因为 L (Time/Length) 在第 1 维
        # 结果 shape: [N, F, D], 其中 F = L // 2 + 1
        freq_domain_x = torch.fft.rfft(x, dim=1)
        F = freq_domain_x.shape[1]

        # 2. 定义分割点 (逻辑不变)
        # DC 是 index 0
        # AC 是 1 ~ F-1
        ac_len = F - 1
        split_idx = 1 + (ac_len // 2)

        # 3. 创建掩码 (Mask)
        dc_mask = torch.zeros(F, device=x.device)
        low_mask = torch.zeros(F, device=x.device)
        high_mask = torch.zeros(F, device=x.device)

        # 设置掩码区域
        dc_mask[0] = 1              # 仅保留 DC
        low_mask[1:split_idx] = 1   # DC 之后到中间
        high_mask[split_idx:] = 1   # 中间到最后

        # 【关键修改】调整掩码形状以适配 [N, F, D]
        # 我们需要掩码形状为 [1, F, 1]，这样才能广播到 [N, F, D]
        dc_mask = dc_mask.view(1, F, 1)
        low_mask = low_mask.view(1, F, 1)
        high_mask = high_mask.view(1, F, 1)

        # 4. 频域滤波
        # [N, F, D] * [1, F, 1] -> [N, F, D]
        dc_fd = freq_domain_x * dc_mask
        low_fd = freq_domain_x * low_mask
        high_fd = freq_domain_x * high_mask

        # 5. 逆 FFT 回时域 (IRFFT)
        # 必须指定 dim=1 和 n=L
        dc = torch.fft.irfft(dc_fd, n=L, dim=1)
        low = torch.fft.irfft(low_fd, n=L, dim=1)
        high = torch.fft.irfft(high_fd, n=L, dim=1)

        return dc, low, high