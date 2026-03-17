import matplotlib.pyplot as plt
import numpy as np

# 1. 原始数据
charactertrajectories_acc = [82.31, 80.87, 80.23, 80.81, 77.38]
uwave_acc = [67.84, 66.26, 71.38, 69.38, 67.5]
har_acc = [79.1,78,82.4,81.21,78.85]

charactertrajectories_forget = [16.2, 18.06, 18.89, 18.51, 22.87]
uwave_forget = [32.06, 28.58, 24.35, 31.56, 29.1]
har_forget = [15.72,16.57,13.57,11.97,13.3]

# 2. 计算平均值 (Mean of two datasets)
# 使用 np.mean 方便地对两个列表对应位置求均值
avg_acc = np.mean([charactertrajectories_acc, uwave_acc,har_acc], axis=0)
forgetting = np.mean([charactertrajectories_forget, uwave_forget,har_forget], axis=0)

# 3. 参数配置
beta_values = ['0.02', '0.04', '0.06', '0.08', '0.10']

# --- 开始绘图 ---
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=128)

# 绘制平均准确率 (柱状图)
color_acc = '#1f4e79'
ax1.set_xlabel(r'$\alpha$', fontsize=14)
ax1.set_ylabel('Average Accuracy (%)', color=color_acc, fontsize=14)
bars = ax1.bar(beta_values, avg_acc, color=color_acc, alpha=0.8, width=0.4, label='Avg Acc')
ax1.tick_params(axis='y', labelcolor=color_acc)

# 动态调整 Y 轴范围：取最小值减 5，最大值加 5
ax1.set_ylim(min(avg_acc) - 5, max(avg_acc) + 5)

# 创建第二个坐标轴绘制遗忘率 (折线图)
ax2 = ax1.twinx()
color_forg = '#d9534f'
ax2.set_ylabel('Average Forgetting Rate (%)', color=color_forg, fontsize=14)
line = ax2.plot(beta_values, forgetting, color=color_forg, marker='s', markersize=8,
                linewidth=2, label='Avg Forgetting')
ax2.tick_params(axis='y', labelcolor=color_forg)

# 动态调整 Y 轴范围
ax2.set_ylim(min(forgetting) - 5, max(forgetting) + 5)

# 装饰
# plt.title(r'Performance Trade-off: Accuracy vs Forgetting (Mean of Datasets)', fontsize=16)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱状图上方标注准确率数值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', color=color_acc, fontweight='bold')

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', frameon=True)

plt.tight_layout()
plt.show()