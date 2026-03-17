import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def get_dtw_path_with_penalty(s1, s2, temporal_penalty_weight=0.0):
    """
    使用动态规划计算 DTW 路径。
    temporal_penalty_weight: 模拟 DILATE 中的 TDI (时间扭曲惩罚)
    如果 weight > 0，则会对偏离对角线的对齐施加惩罚 (i-j)^2
    """
    n, m = len(s1), len(s2)
    # 初始化累积代价矩阵
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # 记录路径回溯
    traceback = np.zeros((n + 1, m + 1, 2), dtype=int)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 1. 基础形状距离 (Shape Loss)
            dist = (s1[i - 1] - s2[j - 1]) ** 2

            # 2. 时间扭曲惩罚 (Temporal Loss / TDI) - DILATE 的核心
            # 惩罚 (i - j) 的偏差
            time_penalty = temporal_penalty_weight * ((i - j) ** 2)

            cost = dist + time_penalty

            # 动态规划找最小路径
            prev_costs = [
                dtw_matrix[i - 1, j],  # Insertion
                dtw_matrix[i, j - 1],  # Deletion
                dtw_matrix[i - 1, j - 1]  # Match
            ]
            best_prev_idx = np.argmin(prev_costs)

            dtw_matrix[i, j] = cost + prev_costs[best_prev_idx]

            if best_prev_idx == 0:
                traceback[i, j] = [i - 1, j]
            elif best_prev_idx == 1:
                traceback[i, j] = [i, j - 1]
            else:
                traceback[i, j] = [i - 1, j - 1]

    # 回溯路径
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = traceback[i, j]

    return np.array(path[::-1])


def plot_alignment(ax, s1, s2, path, title, color_map='viridis'):
    """绘制时序对齐图"""
    # 将 s1 向上平移以便展示
    offset = 4.0
    s1_shifted = s1 + offset

    # 绘制两条曲线
    ax.plot(s1_shifted, label='Series A (Teacher)', color='black', linewidth=2)
    ax.plot(s2, label='Series B (Student)', color='gray', linewidth=2, linestyle='--')

    # 准备绘制对齐线
    lines = []
    colors = []

    for (i, j) in path:
        # 每隔几个点画一条线，避免太密集
        if i % 2 == 0:
            # 起点 (i, s1[i]), 终点 (j, s2[j])
            lines.append([(i, s1_shifted[i]), (j, s2[j])])

            # 计算时间扭曲程度作为颜色依据
            distortion = abs(i - j)
            colors.append(distortion)

    # 创建线集合
    lc = LineCollection(lines, cmap=color_map, alpha=0.6, linewidth=1)
    lc.set_array(np.array(colors))
    ax.add_collection(lc)

    # 设置坐标轴
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-2, offset + 2)
    ax.set_yticks([])
    ax.set_xlabel("Time Steps")
    ax.legend(loc='upper right')

    # 添加简单的说明文字
    if "DILATE" in title:
        ax.text(0.02, 0.5, "Strong Temporal Constraint\n(Lines are vertical)",
                transform=ax.transAxes, color='green', fontsize=10, fontweight='bold')
    else:
        ax.text(0.02, 0.5, "Flexible Warping\n(Lines are slanted)",
                transform=ax.transAxes, color='red', fontsize=10, fontweight='bold')


# ==========================================
# 1. 生成模拟数据 (Simulate Data)
# ==========================================
t = np.linspace(0, 4 * np.pi, 60)
# Series 1: 标准波形
series1 = np.sin(t)
# Series 2: 它是 S1 的一种变形，相位发生了偏移，且局部频率有变化
# 这模拟了 Student 模型学到了形状，但时间没对准的情况
series2 = np.sin(t * 1.2 + 0.5)

# ==========================================
# 2. 计算路径 (Compute Paths)
# ==========================================

# Path A: Standard DTW (Pure Shape)
# 惩罚权重 = 0，允许任意扭曲，只要数值像就行
path_dtw = get_dtw_path_with_penalty(series1, series2, temporal_penalty_weight=0.0)

# Path B: DILATE (Shape + Time)
# 惩罚权重 > 0，强迫路径接近对角线，模拟 Dilate Loss 的效果
# 增加权重会迫使连线变直
path_dilate = get_dtw_path_with_penalty(series1, series2, temporal_penalty_weight=0.15)

# ==========================================
# 3. 绘图 (Visualization)
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=120)

# Plot 1: DTW
plot_alignment(axes[0], series1, series2, path_dtw,
               title="Standard DTW (Focus on Shape)", color_map='Reds')

# Plot 2: DILATE
plot_alignment(axes[1], series1, series2, path_dilate,
               title="DILATE / Ours (Shape + Temporal Penalty)", color_map='Greens')

plt.tight_layout()
plt.show()