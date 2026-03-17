"""
可视化原始数据集的能量分布
包括：
1. 时域能量分布（按类别）
2. 频域能量分布（功率谱密度）
3. 各通道能量对比
4. 类别能量统计
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
import torch
import torch.nn as nn
from models.utils import TransposedInstanceNorm1d
from utils.utils import TriBandFDFilterNLD

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_dataset(dataset_name):
    """
    加载数据集
    Args:
        dataset_name: 数据集名称，如 'DailySports', 'WISDM', 'HAR', 'GrabMyo', 'UWave'
    Returns:
        x_train, y_train, x_test, y_test
    """
    path_dict = {
        'DailySports': '../data/saved/DailySports/',
        'WISDM': '../data/saved/WISDM/',
        'HAR': '../data/saved/HAR_inertial/',
        'GrabMyo': '../data/saved/GRABMyo/',
        'UWave': '../data/saved/UWave/',
    }

    if dataset_name not in path_dict:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(path_dict.keys())}")

    path = path_dict[dataset_name]

    with open(path + 'x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open(path + 'state_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(path + 'x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open(path + 'state_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    # 确保标签是一维数组
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    return x_train, y_train, x_test, y_test


def plot_total_time_domain_signals(x, y, dataset_name, n_samples=3, max_classes=5, save_path=None):
    """
    绘制各通道相加后的总时域图 (Sum of All Channels)
    Args:
        x: 输入数据 (n_samples, seq_len, n_channels)
        y: 标签
        dataset_name: 数据集名称
        n_samples: 每个类别展示多少个随机样本
        max_classes: 最多展示多少个类别
    """
    unique_classes = np.unique(y)

    if len(unique_classes) > max_classes:
        unique_classes = unique_classes[:max_classes]

    n_classes = len(unique_classes)

    # 【核心操作】在通道维度求和
    # x shape: (N, L, C) -> (N, L)
    x_total = np.sum(x, axis=2)

    # 创建图形：行数=类别数，列数=样本数
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(4 * n_samples, 3 * n_classes), sharex=True, sharey='row')

    # 处理维度问题
    if n_classes == 1 and n_samples == 1:
        axes = np.array([[axes]])
    elif n_classes == 1:
        axes = axes.reshape(1, -1)
    elif n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, cls in enumerate(unique_classes):
        class_indices = np.where(y == cls)[0]

        # 随机采样
        if len(class_indices) >= n_samples:
            selected_indices = np.random.choice(class_indices, n_samples, replace=False)
        else:
            selected_indices = np.random.choice(class_indices, n_samples, replace=True)

        for j, idx in enumerate(selected_indices):
            ax = axes[i, j]

            # 获取该样本的总和信号
            signal_sum = x_total[idx]

            # 绘制波形
            ax.plot(signal_sum, color='darkviolet', linewidth=1.5, alpha=0.9, label='Sum of Channels')

            # 画一条 0 刻度线作为参考
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            # 设置标题
            if i == 0:
                ax.set_title(f'Sample {j + 1}', fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'Class {int(cls)}\nSum Amplitude', fontsize=12, fontweight='bold')

            # 图例
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize='small')

            ax.grid(True, alpha=0.3)

            if i == n_classes - 1:
                ax.set_xlabel('Time Step', fontsize=10)

    plt.suptitle(f'{dataset_name}: Total Time-Domain Signals (Sum of All Channels)', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + f'{dataset_name}_total_signals.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}{dataset_name}_total_signals.png")

    plt.show()


def plot_individual_channels_signals(x, y, dataset_name, max_classes=5, save_path=None):
    """
    绘制单个样本各个通道的时域图 (Individual Channels)
    每个类别随机挑选 1 个样本进行展示，并在同一张图上绘制其所有通道。

    Args:
        x: 输入数据 (n_samples, seq_len, n_channels)
        y: 标签
        dataset_name: 数据集名称
        max_classes: 最多展示多少个类别
        save_path: 保存路径
    """
    # 获取基本信息
    n_total_samples, seq_len, n_channels = x.shape
    unique_classes = np.unique(y)

    # 限制显示的最大类别数
    if len(unique_classes) > max_classes:
        unique_classes = unique_classes[:max_classes]
    n_classes = len(unique_classes)

    # 创建图形：行数=类别数，列数=1 (因为只选一个样本)
    # sharex=True: 所有子图共用X轴
    # sharey='row': 如果有多个样本在一行时共用Y轴，这里列数=1，效果等于不共用，保证每个类别的幅度能自动缩放适应
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 3 * n_classes), sharex=True)

    # 处理当只有一个类别时 axes 不是列表的问题，确保它是可迭代的
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 生成用于不同通道的颜色表 (使用 tab10 或 tab20 这种对比度高的分类色盘)
    cmap_name = 'tab10' if n_channels <= 10 else 'tab20'
    # 获取 colormap 对象 (兼容新旧版 matplotlib)
    try:
        cmap = mpl.colormaps[cmap_name].resampled(n_channels)
    except AttributeError:
        # 旧版 matplotlib
        cmap = plt.cm.get_cmap(cmap_name, n_channels)

    colors = [cmap(i) for i in range(n_channels)]

    print(f"Plotting individual channels for {n_classes} classes. Total channels: {n_channels}")

    for i, cls in enumerate(unique_classes):
        ax = axes[i]
        class_indices = np.where(y == cls)[0]

        # 随机采样 1 个样本
        if len(class_indices) == 0:
            print(f"Warning: Class {cls} has no samples.")
            continue

        selected_idx = np.random.choice(class_indices, 1)[0]

        # 获取该样本的数据，形状为 (seq_len, n_channels)
        sample_data = x[selected_idx]

        # --- 核心修改：循环绘制每一个通道 ---
        for c in range(n_channels):
            # 获取第 c 个通道的数据
            channel_signal = sample_data[:, c]
            # 绘制波形，指定颜色和标签
            ax.plot(channel_signal, color=colors[c], linewidth=1.2, alpha=0.8, label=f'Ch {c}')

        # 画一条 0 刻度线作为参考
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # 设置 Y 轴标签
        ax.set_ylabel(f'Class {int(cls)}\nAmplitude', fontweight='bold', fontsize=10)
        # 设置子图标题 (显示选择了哪个索引的样本)
        ax.set_title(f'Class {int(cls)} - Random Sample Idx: {selected_idx}', fontsize=10)

        # 图例设置
        # 为了避免每个子图都显示图例导致拥挤，通常只在第一个子图显示，或者把图例放在图外
        # 这里选择只在第一个图显示，并设置列数防止太长
        if i == 0:
            ncol_legend = min(n_channels, 5)  # 图例最多显示5列
            ax.legend(loc='upper right', fontsize='x-small', ncol=ncol_legend, framealpha=0.5)

        ax.grid(True, alpha=0.3, linestyle=':')

        # 只在最后一个子图设置 X 轴标签
        if i == n_classes - 1:
            ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')

    # 设置总标题
    plt.suptitle(f'{dataset_name}: Individual Channel Signals (1 Sample Per Class)', fontsize=14, y=1.01,
                 fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_file = save_path + f'{dataset_name}_individual_channels.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved chart to: {save_file}")

    plt.show()


def plot_individual_channels_fixed(x, y, dataset_name, max_classes=5, sample_offset=0, save_path=None):
    """
    绘制单个样本各个通道的时域图 (固定样本版本)

    Args:
        x: 输入数据 (n_samples, seq_len, n_channels)
        y: 标签
        dataset_name: 数据集名称
        max_classes: 最多展示多少个类别
        sample_offset: 指定选取该类别的第几个样本 (默认为0，即取第一个)。
                       如果该类别样本数不足，会自动取最后一个，防止报错。
        save_path: 保存路径
    """
    # 获取基本信息
    n_total_samples, seq_len, n_channels = x.shape
    unique_classes = np.unique(y)

    # 限制显示的最大类别数
    if len(unique_classes) > max_classes:
        unique_classes = unique_classes[:max_classes]
    n_classes = len(unique_classes)

    # 创建图形
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 3 * n_classes), sharex=True)

    # 处理当只有一个类别时 axes 不是列表的问题
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 生成颜色表
    cmap_name = 'tab10' if n_channels <= 10 else 'tab20'
    try:
        cmap = mpl.colormaps[cmap_name].resampled(n_channels)
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name, n_channels)

    colors = [cmap(i) for i in range(n_channels)]

    print(f"Plotting fixed samples (Offset={sample_offset}) for {n_classes} classes.")

    for i, cls in enumerate(unique_classes):
        ax = axes[i]
        class_indices = np.where(y == cls)[0]

        # 检查样本是否存在
        if len(class_indices) == 0:
            print(f"Warning: Class {cls} has no samples.")
            continue

        # --- 核心修改：选取固定样本 ---
        # 确保 offset 不会越界。如果该类只有 3 个样本，你想要第 10 个，它会取第 3 个(索引2)。
        target_idx_in_class = min(sample_offset, len(class_indices) - 1)
        selected_idx = class_indices[target_idx_in_class]

        # 获取该样本的数据
        sample_data = x[selected_idx]

        # 循环绘制每一个通道
        for c in range(n_channels):
            channel_signal = sample_data[:, c]
            ax.plot(channel_signal, color=colors[c], linewidth=1.2, alpha=0.8, label=f'Ch {c}')

        # 画 0 刻度线
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # 设置标签和标题
        ax.set_ylabel(f'Class {int(cls)}\nAmplitude', fontweight='bold', fontsize=10)
        # 标题明确显示这是第几个样本
        ax.set_title(f'Class {int(cls)} - Fixed Sample Idx: {selected_idx} (Offset: {target_idx_in_class})',
                     fontsize=10)

        # 图例只在第一个图显示
        if i == 0:
            ncol_legend = min(n_channels, 5)
            ax.legend(loc='upper right', fontsize='x-small', ncol=ncol_legend, framealpha=0.5)

        ax.grid(True, alpha=0.3, linestyle=':')

        if i == n_classes - 1:
            ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')

    plt.suptitle(f'{dataset_name}: Fixed Sample Channels (Offset {sample_offset})', fontsize=14, y=1.01,
                 fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_file = save_path + f'{dataset_name}_fixed_offset_{sample_offset}.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved chart to: {save_file}")

    plt.show()

def compute_time_domain_energy(x):
    """
    计算时域能量（信号的平方和）
    Args:
        x: 输入信号, shape (n_samples, seq_len, n_channels)
    Returns:
        energy: 每个样本的能量, shape (n_samples,)
    """
    # 计算每个样本的总能量（所有通道的平方和）
    energy = np.sum(x ** 2, axis=(1, 2))
    return energy


def compute_frequency_domain_energy(x, fs=20):
    """
    计算频域能量（功率谱密度）
    Args:
        x: 输入信号, shape (n_samples, seq_len, n_channels)
        fs: 采样频率
    Returns:
        freq: 频率数组
        psd: 功率谱密度, shape (n_samples, n_freq)
    """
    n_samples, seq_len, n_channels = x.shape

    # 对每个通道计算FFT，然后求平均
    psd_all = []
    for i in range(n_samples):
        psd_sample = []
        for ch in range(n_channels):
            # 计算功率谱密度
            freq, psd_ch = signal.welch(x[i, :, ch], fs=fs, nperseg=min(256, seq_len))
            psd_sample.append(psd_ch)
        # 对所有通道求平均
        psd_all.append(np.mean(psd_sample, axis=0))

    psd_all = np.array(psd_all)
    return freq, psd_all


def compute_channel_energy(x):
    """
    计算每个通道的能量
    Args:
        x: 输入信号, shape (n_samples, seq_len, n_channels)
    Returns:
        channel_energy: 每个通道的平均能量, shape (n_channels,)
    """
    # 计算每个通道在所有样本上的平均能量
    channel_energy = np.mean(np.sum(x ** 2, axis=1), axis=0)
    return channel_energy


def plot_time_energy_distribution(x, y, dataset_name, save_path=None):
    """
    绘制时域能量分布（按类别）
    """
    # 计算每个样本的能量
    energies = compute_time_domain_energy(x)

    # 获取所有类别
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 1. 箱线图：显示每个类别的能量分布
    class_energies = [energies[y == cls] for cls in unique_classes]

    axes[0].boxplot(class_energies, labels=[f'Class {int(cls)}' for cls in unique_classes])
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Energy', fontsize=12)
    axes[0].set_title(f'{dataset_name}: Time-domain Energy Distribution by Class', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2. 小提琴图：显示能量分布的密度
    parts = axes[1].violinplot(class_energies, positions=range(n_classes),
                                showmeans=True, showmedians=True)
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Energy Density', fontsize=12)
    axes[1].set_title(f'{dataset_name}: Energy Density Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(n_classes))
    axes[1].set_xticklabels([f'Class {int(cls)}' for cls in unique_classes])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + f'{dataset_name}_time_energy_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}{dataset_name}_time_energy_distribution.png")

    plt.show()


def plot_frequency_energy_distribution(x, y, dataset_name, fs=20, save_path=None):
    """
    绘制频域能量分布（功率谱密度）
    """
    # 计算频域能量
    freq, psd_all = compute_frequency_domain_energy(x, fs=fs)

    # 获取所有类别
    unique_classes = np.unique(y)

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 1. 每个类别的平均功率谱
    for cls in unique_classes:
        class_mask = (y == cls)
        psd_class_mean = np.mean(psd_all[class_mask], axis=0)
        axes[0].semilogy(freq, psd_class_mean, label=f'Class {int(cls)}', linewidth=2, alpha=0.7)

    axes[0].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[0].set_ylabel('Power Spectral Density', fontsize=12)
    axes[0].set_title(f'{dataset_name}: Average Power Spectral Density by Class', fontsize=14, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=max(1, len(unique_classes)//10))
    axes[0].grid(True, alpha=0.3)

    # 2. 所有样本的频域能量热图
    # 对每个类别全部采样
    sampled_psd = []
    sampled_labels = []
    for cls in unique_classes:
        class_mask = (y == cls)
        class_psd = psd_all[class_mask]
        n_samples = len(class_psd)  # 使用全部样本
        indices = np.random.choice(len(class_psd), n_samples, replace=False)
        sampled_psd.append(class_psd[indices])
        sampled_labels.extend([int(cls)] * n_samples)

    sampled_psd = np.vstack(sampled_psd)

    im = axes[1].imshow(sampled_psd, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_xlabel('Frequency Bin', fontsize=12)
    axes[1].set_ylabel('Sample Index (grouped by class)', fontsize=12)
    axes[1].set_title(f'{dataset_name}: Power Spectral Density Heatmap (Sampled)', fontsize=14, fontweight='bold')

    # 添加类别分隔线
    cumulative_samples = 0
    for i, cls in enumerate(unique_classes[:-1]):  # 不需要在最后一个类别后画线
        class_count = np.sum(y == cls)
        cumulative_samples += class_count
        axes[1].axhline(y=cumulative_samples - 0.5, color='white', linewidth=2, linestyle='--', alpha=0.7)

    plt.colorbar(im, ax=axes[1], label='PSD')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + f'{dataset_name}_frequency_energy_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}{dataset_name}_frequency_energy_distribution.png")

    plt.show()


def plot_channel_energy_comparison(x, y, dataset_name, save_path=None):
    """
    绘制各通道能量对比
    """
    n_channels = x.shape[2]
    unique_classes = np.unique(y)

    # 计算每个类别每个通道的平均能量
    channel_energies = np.zeros((len(unique_classes), n_channels))

    for i, cls in enumerate(unique_classes):
        class_mask = (y == cls)
        class_data = x[class_mask]
        # 计算每个通道的平均能量
        channel_energies[i] = np.mean(np.sum(class_data ** 2, axis=1), axis=0)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 堆叠柱状图
    x_pos = np.arange(len(unique_classes))
    width = 0.6

    bottom = np.zeros(len(unique_classes))
    colors = plt.cm.Set3(np.linspace(0, 1, n_channels))

    for ch in range(n_channels):
        axes[0].bar(x_pos, channel_energies[:, ch], width, bottom=bottom,
                   label=f'Channel {ch+1}', color=colors[ch], alpha=0.8)
        bottom += channel_energies[:, ch]

    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Energy', fontsize=12)
    axes[0].set_title(f'{dataset_name}: Channel Energy Contribution by Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f'Class {int(cls)}' for cls in unique_classes])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. 热图显示
    im = axes[1].imshow(channel_energies.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Channel', fontsize=12)
    axes[1].set_title(f'{dataset_name}: Channel Energy Heatmap', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(unique_classes)))
    axes[1].set_xticklabels([f'{int(cls)}' for cls in unique_classes])
    axes[1].set_yticks(range(n_channels))
    axes[1].set_yticklabels([f'Ch {ch+1}' for ch in range(n_channels)])
    plt.colorbar(im, ax=axes[1], label='Energy')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + f'{dataset_name}_channel_energy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}{dataset_name}_channel_energy_comparison.png")

    plt.show()


def plot_energy_statistics(x, y, dataset_name, save_path=None):
    """
    绘制能量统计摘要
    """
    energies = compute_time_domain_energy(x)
    unique_classes = np.unique(y)

    # 计算统计量
    stats = {}
    for cls in unique_classes:
        class_mask = (y == cls)
        class_energies = energies[class_mask]
        stats[int(cls)] = {
            'mean': np.mean(class_energies),
            'std': np.std(class_energies),
            'min': np.min(class_energies),
            'max': np.max(class_energies),
            'median': np.median(class_energies),
            'q25': np.percentile(class_energies, 25),
            'q75': np.percentile(class_energies, 75)
        }

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    classes = list(stats.keys())

    # 1. 均值和标准差
    means = [stats[cls]['mean'] for cls in classes]
    stds = [stats[cls]['std'] for cls in classes]

    axes[0, 0].bar(range(len(classes)), means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Class', fontsize=12)
    axes[0, 0].set_ylabel('Mean Energy', fontsize=12)
    axes[0, 0].set_title('Mean Energy with Std Dev', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(range(len(classes)))
    axes[0, 0].set_xticklabels([f'Class {cls}' for cls in classes])
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. 最小值和最大值范围
    mins = [stats[cls]['min'] for cls in classes]
    maxs = [stats[cls]['max'] for cls in classes]

    axes[0, 1].fill_between(range(len(classes)), mins, maxs, alpha=0.3, color='coral', label='Min-Max Range')
    axes[0, 1].plot(range(len(classes)), means, 'o-', color='darkred', linewidth=2, markersize=8, label='Mean')
    axes[0, 1].set_xlabel('Class', fontsize=12)
    axes[0, 1].set_ylabel('Energy', fontsize=12)
    axes[0, 1].set_title('Energy Range (Min-Max)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(range(len(classes)))
    axes[0, 1].set_xticklabels([f'Class {cls}' for cls in classes])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 中位数和四分位数
    medians = [stats[cls]['median'] for cls in classes]
    q25s = [stats[cls]['q25'] for cls in classes]
    q75s = [stats[cls]['q75'] for cls in classes]

    axes[1, 0].fill_between(range(len(classes)), q25s, q75s, alpha=0.3, color='lightgreen', label='IQR (Q25-Q75)')
    axes[1, 0].plot(range(len(classes)), medians, 's-', color='darkgreen', linewidth=2, markersize=8, label='Median')
    axes[1, 0].set_xlabel('Class', fontsize=12)
    axes[1, 0].set_ylabel('Energy', fontsize=12)
    axes[1, 0].set_title('Median Energy with IQR', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(classes)))
    axes[1, 0].set_xticklabels([f'Class {cls}' for cls in classes])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 变异系数 (CV = std/mean)
    cvs = [stats[cls]['std'] / stats[cls]['mean'] if stats[cls]['mean'] > 0 else 0 for cls in classes]

    axes[1, 1].bar(range(len(classes)), cvs, alpha=0.7, color='mediumpurple')
    axes[1, 1].set_xlabel('Class', fontsize=12)
    axes[1, 1].set_ylabel('Coefficient of Variation', fontsize=12)
    axes[1, 1].set_title('Energy Variability (CV = Std/Mean)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(range(len(classes)))
    axes[1, 1].set_xticklabels([f'Class {cls}' for cls in classes])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + f'{dataset_name}_energy_statistics.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}{dataset_name}_energy_statistics.png")

    plt.show()

    # 打印统计表格
    print(f"\n{'='*80}")
    print(f"Energy Statistics for {dataset_name}")
    print(f"{'='*80}")
    print(f"{'Class':<10} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15} {'Median':<15}")
    print(f"{'-'*80}")
    for cls in classes:
        print(f"{cls:<10} {stats[cls]['mean']:<15.2f} {stats[cls]['std']:<15.2f} "
              f"{stats[cls]['min']:<15.2f} {stats[cls]['max']:<15.2f} {stats[cls]['median']:<15.2f}")
    print(f"{'='*80}\n")


def visualize_dataset_energy(dataset_name='DailySports', fs=20, save_path='../result/plots/'):
    """
    完整的数据集能量可视化
    Args:
        dataset_name: 数据集名称
        fs: 采样频率 (Hz)
        save_path: 保存路径
    """
    print(f"\nLoading {dataset_name} dataset...")

    # 加载数据
    x_train, y_train, x_test, y_test = load_dataset(dataset_name)

    print(f"Train data shape: {x_train.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Number of channels: {x_train.shape[2]}")

    # 创建保存目录
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # 使用训练集进行可视化（也可以选择测试集或合并）
    x, y = x_train, y_train

    fd_filter = TriBandFDFilterNLD()

    # fd_filter 返回: dc, low, high
    x_dc, x_low, x_high = fd_filter(x)

    # B. 构建 Low 输入 (包含 DC) 和 High 输入
    x_in_low = x_dc + x_low  # 低频分支包含基准值
    x_in_high = x_high  # 高频分支只看纹理



    # 转换为 torch tensor (PyTorch 模块需要 tensor 输入)
    x_tensor = torch.from_numpy(x).float()

    input_norm_ln = nn.LayerNorm(x.shape[-1], elementwise_affine=False)

    input_norm_in = TransposedInstanceNorm1d(x.shape[-1],affine=False)

    x_norm = input_norm_in(x_tensor)
    # x_in_low_norm = input_norm_in(x_in_low)
    # x_in_high_norm = input_norm_in(x_in_high)

    x_norm_ln = input_norm_ln(x_tensor)
    # x_in_low_norm_ln = input_norm_ln(x_in_low)
    # x_in_high_norm_ln = input_norm_ln(x_in_high)

    # 转换回 numpy array 用于可视化
    x_norm = x_norm.detach().cpu().numpy()
    x_norm_ln = x_norm_ln.detach().cpu().numpy()

    print("\n" + "="*80)
    print(f"Visualizing Energy Distribution for {dataset_name}")
    print("="*80)

    # # === 新增部分 ===
    # print("\n0.5. Plotting total time-domain signals (Sum)...")
    # plot_total_time_domain_signals(x, y, dataset_name,
    #                                n_samples=3,
    #                                max_classes=5,
    #                                save_path=save_path)

    # plot_individual_channels_signals(x_norm, y, dataset_name,
    #                                  max_classes=20,
    #                                  save_path=save_path)

    plot_individual_channels_fixed(x_norm_ln, y, dataset_name,
                                  max_classes=20,
                                  sample_offset=0,
                                  save_path=save_path)

    # # 1. 时域能量分布
    # print("\n1. Plotting time-domain energy distribution...")
    # plot_time_energy_distribution(x, y, dataset_name, save_path)
    #
    # # 2. 频域能量分布
    # print("\n2. Plotting frequency-domain energy distribution...")
    # plot_frequency_energy_distribution(x, y, dataset_name, fs, save_path)
    #
    # # 3. 通道能量对比
    # print("\n3. Plotting channel energy comparison...")
    # plot_channel_energy_comparison(x, y, dataset_name, save_path)
    #
    # # 4. 能量统计
    # print("\n4. Plotting energy statistics...")
    # plot_energy_statistics(x, y, dataset_name, save_path)

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80 + "\n")





if __name__ == "__main__":
    # 示例：可视化不同数据集的能量分布

    # 可选的数据集: 'DailySports', 'WISDM', 'HAR', 'GrabMyo', 'UWave'
    datasets = ['GrabMyo']  # 可以添加更多数据集

    # 不同数据集的采样频率（需要根据实际情况调整）
    sampling_rates = {
        'DailySports': 25,  # 25 Hz
        'WISDM': 20,        # 20 Hz
        'HAR': 50,          # 50 Hz
        'GrabMyo': 200,     # 200 Hz
        'UWave': 100,       # 估计值，需要确认
    }

    for dataset in datasets:
        fs = sampling_rates.get(dataset, 20)  # 默认20Hz
        visualize_dataset_energy(dataset_name=dataset, fs=fs, save_path='../result/plots/')