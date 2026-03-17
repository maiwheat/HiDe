"""
检查原始数据的直流分量（DC Component）
验证数据是否经过去均值预处理
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_har_data():
    """加载 HAR 数据集"""
    path = '../data/saved/HAR_inertial/'
    with open(path + 'x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open(path + 'state_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    return x_train, y_train


def check_dc_component(x, dataset_name='HAR'):
    """
    检查数据的直流分量（均值）
    Args:
        x: 输入数据, shape (n_samples, seq_len, n_channels)
        dataset_name: 数据集名称
    """
    n_samples, seq_len, n_channels = x.shape

    print("\n" + "="*80)
    print(f"直流分量（DC Component）分析 - {dataset_name} 数据集")
    print("="*80)

    # 1. 计算每个通道的全局均值
    channel_means = np.mean(x, axis=(0, 1))  # shape: (n_channels,)
    channel_stds = np.std(x, axis=(0, 1))

    print(f"\n数据形状: {x.shape}")
    print(f"通道数: {n_channels}")

    # HAR 数据集的通道说明
    if dataset_name == 'HAR' and n_channels == 9:
        channel_names = [
            'Total_Acc_X', 'Total_Acc_Y', 'Total_Acc_Z',  # 0-2: 总加速度（含重力）
            'Body_Acc_X', 'Body_Acc_Y', 'Body_Acc_Z',     # 3-5: 身体加速度（去除重力）
            'Body_Gyro_X', 'Body_Gyro_Y', 'Body_Gyro_Z'   # 6-8: 陀螺仪（角速度）
        ]
    else:
        channel_names = [f'Channel_{i}' for i in range(n_channels)]

    print("\n" + "-"*80)
    print(f"{'通道名称':<20} {'全局均值 (DC)':<20} {'标准差':<20} {'|均值|/标准差':<15}")
    print("-"*80)

    for i, name in enumerate(channel_names):
        ratio = np.abs(channel_means[i]) / channel_stds[i] if channel_stds[i] > 0 else 0
        print(f"{name:<20} {channel_means[i]:<20.6f} {channel_stds[i]:<20.6f} {ratio:<15.6f}")

    # 2. 计算每个样本的均值分布
    sample_means = np.mean(x, axis=(1, 2))  # shape: (n_samples,)

    print("\n" + "-"*80)
    print("样本级别统计:")
    print(f"  所有样本的均值: {np.mean(sample_means):.6f}")
    print(f"  样本均值的标准差: {np.std(sample_means):.6f}")
    print(f"  样本均值的范围: [{np.min(sample_means):.6f}, {np.max(sample_means):.6f}]")

    # 3. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 3.1 每个通道的直流分量（柱状图）
    ax1 = axes[0, 0]
    colors = ['red', 'red', 'red', 'blue', 'blue', 'blue', 'green', 'green', 'green'][:n_channels]
    bars = ax1.bar(range(n_channels), channel_means, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Channel Index', fontsize=12)
    ax1.set_ylabel('Mean Value (DC Component)', fontsize=12)
    ax1.set_title(f'{dataset_name}: DC Component per Channel', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(n_channels))
    ax1.set_xticklabels([name.replace('_', '\n') for name in channel_names], rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加图例（HAR 特有）
    if dataset_name == 'HAR' and n_channels == 9:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Total Acc (含重力)'),
            Patch(facecolor='blue', alpha=0.7, label='Body Acc (去除重力)'),
            Patch(facecolor='green', alpha=0.7, label='Body Gyro (角速度)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

    # 3.2 每个通道的标准差（对比）
    ax2 = axes[0, 1]
    bars = ax2.bar(range(n_channels), channel_stds, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Channel Index', fontsize=12)
    ax2.set_ylabel('Standard Deviation', fontsize=12)
    ax2.set_title(f'{dataset_name}: Standard Deviation per Channel', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(n_channels))
    ax2.set_xticklabels([name.replace('_', '\n') for name in channel_names], rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3.3 均值/标准差比率（判断是否去均值）
    ax3 = axes[1, 0]
    ratios = [np.abs(channel_means[i]) / channel_stds[i] if channel_stds[i] > 0 else 0
              for i in range(n_channels)]
    bars = ax3.bar(range(n_channels), ratios, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='阈值=0.1 (接近0均值)')
    ax3.set_xlabel('Channel Index', fontsize=12)
    ax3.set_ylabel('|Mean| / Std', fontsize=12)
    ax3.set_title(f'{dataset_name}: Ratio of |Mean| to Std\n(接近0说明已去均值)',
                  fontsize=14, fontweight='bold')
    ax3.set_xticks(range(n_channels))
    ax3.set_xticklabels([name.replace('_', '\n') for name in channel_names], rotation=45, ha='right', fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 3.4 样本均值分布（直方图）
    ax4 = axes[1, 1]
    ax4.hist(sample_means, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label=f'零均值 (实际均值={np.mean(sample_means):.4f})')
    ax4.set_xlabel('Sample Mean Value', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title(f'{dataset_name}: Distribution of Sample Means', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'../result/plots/{dataset_name}_dc_component_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n保存图像: ../result/plots/{dataset_name}_dc_component_analysis.png")
    plt.show()

    # 4. 结论
    print("\n" + "="*80)
    print("结论:")
    print("="*80)

    if dataset_name == 'HAR' and n_channels == 9:
        print("HAR 数据集包含 3 种类型的传感器数据：")
        print("  1. Total_Acc (通道 0-2): 总加速度 = 重力 + 身体运动")
        print("     → 应该有较大的直流分量（重力加速度 ~9.8 m/s²）")
        print(f"     → 实际均值: {channel_means[0:3]}")
        print()
        print("  2. Body_Acc (通道 3-5): 身体加速度 = Total_Acc - 重力")
        print("     → 直流分量应该接近 0（已去除重力）")
        print(f"     → 实际均值: {channel_means[3:6]}")
        print()
        print("  3. Body_Gyro (通道 6-8): 陀螺仪角速度")
        print("     → 直流分量应该接近 0（角速度本身是高频信号）")
        print(f"     → 实际均值: {channel_means[6:9]}")
        print()

        # 判断是否已经标准化
        avg_ratio = np.mean([np.abs(channel_means[i]) / channel_stds[i]
                            for i in range(n_channels) if channel_stds[i] > 0])

        if avg_ratio < 0.1:
            print("✓ 数据可能已经过标准化处理（均值接近0）")
        elif np.abs(channel_means[0]) > 5 or np.abs(channel_means[1]) > 5 or np.abs(channel_means[2]) > 5:
            print("✓ Total_Acc 通道保留了重力分量（未完全去均值）")
            print("✓ Body_Acc 和 Body_Gyro 通道已去除直流分量")
        else:
            print("✓ 数据保持原始状态（部分通道有直流分量）")

    print("="*80 + "\n")

    return channel_means, channel_stds


def compare_with_frequency_domain(x, channel_idx=0, dataset_name='HAR'):
    """
    对比时域均值（DC）和频域的0频率分量
    """
    from scipy.fft import fft, fftfreq

    # 随机选择一个样本
    sample = x[0, :, channel_idx]  # shape: (seq_len,)

    # 时域均值
    time_mean = np.mean(sample)

    # 频域变换
    fft_vals = fft(sample)
    freqs = fftfreq(len(sample), d=1/50)  # HAR 采样率 50Hz

    # DC 分量是 FFT 的第一个值（0频率）
    dc_freq = np.abs(fft_vals[0]) / len(sample)  # 归一化

    print(f"\n频域验证 (Channel {channel_idx}):")
    print(f"  时域均值 (DC): {time_mean:.6f}")
    print(f"  频域0频分量: {dc_freq:.6f}")
    print(f"  两者是否一致: {'✓ 是' if np.isclose(time_mean, dc_freq) else '✗ 否'}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 时域信号
    axes[0].plot(sample, linewidth=1)
    axes[0].axhline(y=time_mean, color='red', linestyle='--', linewidth=2,
                    label=f'均值 (DC) = {time_mean:.4f}')
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title(f'Time Domain Signal (Channel {channel_idx})', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 频域信号（只显示正频率部分）
    n = len(sample) // 2
    axes[1].plot(freqs[:n], np.abs(fft_vals[:n]) / len(sample), linewidth=1)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2,
                    label=f'DC (0 Hz) = {dc_freq:.4f}')
    axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylabel('Magnitude', fontsize=12)
    axes[1].set_title(f'Frequency Domain (Channel {channel_idx})', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'../result/plots/{dataset_name}_time_freq_dc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n保存图像: ../result/plots/{dataset_name}_time_freq_dc_comparison.png")
    plt.show()


if __name__ == "__main__":
    # 1. 加载数据
    print("加载 HAR 数据集...")
    x_train, y_train = load_har_data()

    # 2. 检查直流分量
    channel_means, channel_stds = check_dc_component(x_train, dataset_name='HAR')

    # 3. 频域验证（选择一个有直流分量的通道，如 Total_Acc_Y）
    compare_with_frequency_domain(x_train, channel_idx=1, dataset_name='HAR')

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)