# -*- coding: UTF-8 -*-
import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace
from sklearn.manifold import TSNE  # 必须引入 TSNE

# from plot.gr_tsne import dataloader

# ==============================================================================
#                               1. 环境与路径设置
# ==============================================================================
# 获取当前脚本所在目录，并将项目根目录加入系统路径，以便导入模块
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)  # 假设脚本在项目根目录的下一级
os.chdir(project_root)
sys.path.insert(0, project_root)

# 导入项目组件 (请确保这些路径在你的项目中存在)
from models.base import setup_model
from utils.stream import IncrementalTaskStream
from utils.data import Dataloader_from_numpy
from utils.utils import TriBandFDFilter, seed_fixer

# ==============================================================================
#                               2. 全局配置参数 (CONFIG)
# ==============================================================================
CONFIG = {
    # --- [关键] 实验路径设置 ---
    "result_root_dir": "result/tune_and_exp",  # 结果存放的根目录
    "exp_dataset_dir": "CNN_dailysports",  # 数据集对应的文件夹
    "exp_folder_name": "Offline_LN_Jan-15-16-37-43",  # 具体实验文件夹名 (请修改为你实际的文件夹)
    "ckpt_filename": "ckpt_r0.pt",  # 权重文件名

    # --- [关键] 评估数据集设置 ---
    "eval_data_name": "dailysports",  # 数据集名称 ('har', 'wisdm', 'dailysports' 等)
    "eval_cls_order": list(range(18)),  # 类别列表 (DailySports通常是19类, HAR是6类)
    "batch_size": 64,  # 推理时的 Batch Size

    # --- 模型架构参数 (必须与训练时一致) ---
    "model": {
        "encoder": "CNN",
        "head": "Linear",
        "norm": "LN",  # 'LN' 或 'BN'
        "feature_dim": 128,
        "n_layers": 4,
        "dropout": 0,
        "input_norm": "LN",
        "agent": "Offline"
    },

    # --- t-SNE 参数 ---
    "tsne_perplexity": 40,  # 数据量大时适当调大 (30-50)
    "tsne_iter": 1000,  # 迭代次数

    # --- 其他 ---
    "seed": 1234,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # === 这里设置你想看的类别 ===
    "target_classes": [2, 3]
}


# ==============================================================================
#                               3. 辅助函数
# ==============================================================================

def create_model_args(cfg):
    """根据 CONFIG 字典构建模型参数对象"""
    m_cfg = cfg["model"]
    args = SimpleNamespace(
        data=cfg["eval_data_name"],
        encoder=m_cfg["encoder"],
        head=m_cfg["head"],
        norm=m_cfg["norm"],
        feature_dim=m_cfg["feature_dim"],
        n_layers=m_cfg["n_layers"],
        dropout=m_cfg["dropout"],
        device=cfg["device"],
        agent=m_cfg["agent"],
        input_norm=m_cfg["input_norm"],
        stream_split='exp'
    )
    return args


def run_analysis_full():
    # --- A. 环境初始化 ---
    seed_fixer(CONFIG["seed"])
    device = CONFIG["device"]
    print(f"Using device: {device}")

    # 构造路径
    base_result_path = os.path.join(project_root, CONFIG["result_root_dir"], CONFIG["exp_dataset_dir"])
    teacher_exp_path = CONFIG["exp_folder_name"]
    teacher_ckpt = os.path.join(base_result_path, teacher_exp_path, CONFIG["ckpt_filename"])

    # --- B. 加载模型 (仅加载 Encoder) ---
    print(f"\n[1/5] Loading model from: {teacher_ckpt}")
    args_t = create_model_args(CONFIG)
    teacher_model = setup_model(args_t)

    if os.path.exists(teacher_ckpt):
        full_state_dict = torch.load(teacher_ckpt, map_location=device)
        # 过滤掉分类头 (Head) 的权重，只加载特征提取器 (Encoder)
        encoder_only_dict = {k: v for k, v in full_state_dict.items() if 'head' not in k}
        msg = teacher_model.load_state_dict(encoder_only_dict, strict=False)
        print(f"Successfully loaded encoder weights. (Skipped layers: {msg.unexpected_keys})")
    else:
        print(f"Error: Checkpoint file not found at {teacher_ckpt}")
        return

    teacher_model.to(device)
    teacher_model.eval()

    # --- C. 准备数据流 (关键修改：加载受试者 ID) ---
    print(f"\n[2/5] Setting up data stream for: {CONFIG['eval_data_name']}")
    task_stream = IncrementalTaskStream(
        data=CONFIG['eval_data_name'],
        scenario='class',
        cls_order=CONFIG['eval_cls_order'],
        split='exp'
    )

    # [修改点 1] 开启 load_subject=True，让 dataloader 返回受试者 ID
    task_stream.setup(load_subject=True)

    # 收集测试集数据
    all_x, all_y, all_subjects = [], [], []

    # 遍历所有任务收集数据
    for t_idx in range(len(task_stream.tasks)):
        test_set = task_stream.tasks[t_idx][2]  # 获取测试集

        # [修改点 2] 尝试解包出 Subject ID
        if len(test_set) == 3:
            x_tmp, y_tmp, subj_tmp = test_set
            all_subjects.append(subj_tmp)
        else:
            # 如果数据集不支持返回 Subject，打印警告并使用全 0 占位
            print("Warning: Subject IDs not found in dataloader. Using dummy IDs.")
            x_tmp, y_tmp = test_set
            all_subjects.append(np.zeros_like(y_tmp))

        all_x.append(x_tmp)
        all_y.append(y_tmp)

    x_test_total = np.concatenate(all_x)
    y_test_total = np.concatenate(all_y)
    subject_total = np.concatenate(all_subjects)  # 拼接所有受试者 ID

    # === 类别过滤逻辑 (保持你之前的逻辑，增加了 subject_total 的同步过滤) ===
    target_cls = CONFIG.get("target_classes", None)
    if target_cls is not None:
        mask = np.isin(y_test_total, target_cls)
        x_test_total = x_test_total[mask]
        y_test_total = y_test_total[mask]
        subject_total = subject_total[mask]  # [重要] 受试者数组也要同步过滤
        print(f"Filtered samples: {len(y_test_total)} (Classes: {target_cls})")
    # =================================================================

    print(f"Total samples (FULL DATASET): {len(y_test_total)}")

    # 创建 Dataloader (只需 x 用于特征提取)
    dataloader = Dataloader_from_numpy(
        x_test_total,
        y_test_total,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    # --- D. 特征提取 ---
    print("\n[3/5] Extracting features...")
    features_list = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            fmap = teacher_model.feature_map(x)

            # 全局平均池化 (GAP)
            pooled_features = torch.mean(fmap, dim=2)
            features_list.append(pooled_features.cpu().numpy())

    all_features = np.concatenate(features_list, axis=0)

    # --- E. 运行 t-SNE ---
    print(f"\n[4/5] Running t-SNE on {len(all_features)} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=CONFIG["tsne_perplexity"],
        n_iter=CONFIG["tsne_iter"],
        random_state=CONFIG["seed"],
        init='pca',
        learning_rate='auto',
        n_jobs=-1
    )
    X_embedded = tsne.fit_transform(all_features)

    # --- F. 可视化绘图 ---
    print("\n[5/5] Generating plots...")
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 10))
    agent_name = CONFIG["exp_folder_name"].split('_')[0]

    # [修改点 3] 绘图设置：颜色(hue)代表受试者，形状(style)代表动作类别
    scatter = sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=subject_total,  # <--- 关键：用受试者 ID 上色
        style=y_test_total,  # 可选：用形状区分动作类别 (如果不加这一行，就只看受试者)
        palette="tab20",  # tab20 适合区分较多的人数 (如 20 人以内)
        s=60,  # 点的大小
        alpha=0.8,  # 透明度
        edgecolor='w',  # 白色描边，让重叠的点更清晰
        linewidth=0.5,
        ax=ax,
        legend="full"
    )

    ax.set_title(f't-SNE Colored by Subject ID (Classes: {target_cls})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=14)
    ax.set_ylabel('Dim 2', fontsize=14)

    # 移动图例到图表外侧
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Subject ID")

    save_fn = f"tsne_subject_variation_{agent_name}_{CONFIG['eval_data_name']}.png"
    save_path = os.path.join(current_script_dir, save_fn)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDONE! Plot saved to: {save_path}")
    plt.show()



# def run_analysis_full():
#     # --- A. 初始化环境 ---
#     seed_fixer(CONFIG["seed"])
#     device = CONFIG["device"]
#     print(f"Using device: {device}")
#
#     # 构造完整路径
#     base_result_path = os.path.join(project_root, CONFIG["result_root_dir"], CONFIG["exp_dataset_dir"])
#     teacher_exp_path = CONFIG["exp_folder_name"]
#     teacher_ckpt = os.path.join(base_result_path, teacher_exp_path, CONFIG["ckpt_filename"])
#
#     # --- B. 加载模型 (仅 Encoder) ---
#     print(f"\n[1/5] Loading model from: {teacher_ckpt}")
#     args_t = create_model_args(CONFIG)
#     teacher_model = setup_model(args_t)
#
#     if os.path.exists(teacher_ckpt):
#         full_state_dict = torch.load(teacher_ckpt, map_location=device)
#         # 过滤掉 Head 层的权重，只加载 Encoder
#         encoder_only_dict = {k: v for k, v in full_state_dict.items() if 'head' not in k}
#         msg = teacher_model.load_state_dict(encoder_only_dict, strict=False)
#         print(f"Successfully loaded encoder weights. (Skipped layers: {msg.unexpected_keys})")
#     else:
#         print(f"Error: Checkpoint file not found at {teacher_ckpt}")
#         return
#
#     teacher_model.to(device)
#     teacher_model.eval()
#
#     # --- C. 准备数据流 ---
#     print(f"\n[2/5] Setting up data stream for: {CONFIG['eval_data_name']}")
#     task_stream = IncrementalTaskStream(
#         data=CONFIG['eval_data_name'],
#         scenario='class',
#         cls_order=CONFIG['eval_cls_order'],
#         split='exp'
#     )
#     task_stream.setup(load_subject=False)
#
#     # 收集所有任务的测试集数据
#     all_x, all_y = [], []
#     for t_idx in range(len(task_stream.tasks)):
#         x_tmp, y_tmp = task_stream.tasks[t_idx][2]  # Index 2 是 Test set
#         all_x.append(x_tmp)
#         all_y.append(y_tmp)
#
#     x_test_total = np.concatenate(all_x)
#     y_test_total = np.concatenate(all_y)
#     # === 插入过滤代码 ===
#     target_cls = CONFIG.get("target_classes", None)
#     if target_cls is not None:
#         mask = np.isin(y_test_total, target_cls)
#         x_test_total = x_test_total[mask]
#         y_test_total = y_test_total[mask]
#         print(f"Filtered samples: {len(y_test_total)} (Classes: {target_cls})")
#     # ==================
#     print(f"Total samples (FULL DATASET): {len(y_test_total)}")
#
#     dataloader = Dataloader_from_numpy(
#         x_test_total,
#         y_test_total,
#         batch_size=CONFIG["batch_size"],
#         shuffle=False
#     )
#
#     # --- D. 特征提取 & 能量计算 ---
#     print("\n[3/5] Extracting features and calculating energy...")
#     fd_filter = TriBandFDFilter().to(device)
#
#     ratios = []  # 存储低频能量占比
#     features_list = []  # 存储特征向量
#     labels_list = []  # 存储标签
#
#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.to(device)
#             # 1. 获取特征图 (Batch, Channel, Time)
#             fmap = teacher_model.feature_map(x)
#
#             # 2. 能量计算
#             dc, low_only, high = fd_filter(fmap)
#             energy_low = torch.sum((dc + low_only) ** 2, dim=(1, 2))
#             energy_high = torch.sum(high ** 2, dim=(1, 2))
#             total_ac_energy = energy_low + energy_high + 1e-8
#             sample_ratios = energy_low / total_ac_energy
#             ratios.extend(sample_ratios.cpu().numpy().tolist())
#
#             # 3. 特征池化 (GAP) 用于 t-SNE
#             # 假设 fmap 是 (Batch, Channel, Time)，我们在 Time 维度做平均
#             pooled_features = torch.mean(fmap, dim=2)
#             features_list.append(pooled_features.cpu().numpy())
#             labels_list.append(y.numpy())
#
#     ratios = np.array(ratios)
#     all_features = np.concatenate(features_list, axis=0)
#     all_labels = np.concatenate(labels_list, axis=0)
#
#     # 打印基础统计
#     print(f"  -> Mean Low-Freq Ratio: {np.mean(ratios):.4f}")
#
#     # --- E. 运行 t-SNE ---
#     print(f"\n[4/5] Running t-SNE on {len(all_features)} samples (Please wait)...")
#     tsne = TSNE(
#         n_components=2,
#         perplexity=CONFIG["tsne_perplexity"],
#         n_iter=CONFIG["tsne_iter"],
#         random_state=CONFIG["seed"],
#         init='pca',
#         learning_rate='auto',
#         n_jobs=-1  # 尝试使用所有 CPU 核心加速
#     )
#     X_embedded = tsne.fit_transform(all_features)
#     print("  -> t-SNE calculation finished.")
#
#     # --- F. 可视化绘图 ---
#     print("\n[5/5] Generating plots...")
#     sns.set_theme(style="whitegrid")
#
#     # 创建宽画布 (左: 直方图, 右: 散点图)
#     fig, axes = plt.subplots(1, 2, figsize=(24, 10))
#     agent_name = CONFIG["exp_folder_name"].split('_')[0]
#
#     # Plot 1: 能量分布
#     sns.histplot(ratios, bins=50, kde=True, color='royalblue', edgecolor='black', alpha=0.7, ax=axes[0])
#     axes[0].axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ratios):.2f}')
#     axes[0].set_title(f'Feature Energy Distribution ({agent_name})', fontsize=16, fontweight='bold')
#     axes[0].set_xlabel('Low-Frequency Energy Ratio', fontsize=14)
#     axes[0].set_ylabel('Count', fontsize=14)
#     axes[0].legend()
#     axes[0].set_xlim(0, 1.0)
#
#     # Plot 2: t-SNE 散点图
#     # 针对全量数据的绘图参数优化
#     scatter = sns.scatterplot(
#         x=X_embedded[:, 0],
#         y=X_embedded[:, 1],
#         hue=all_labels,
#         palette="tab20",  # 适合多类别的色板
#         s=10,  # 点的大小 (越小越不容易重叠)
#         alpha=0.6,  # 透明度 (越低越能看清密集区域)
#         linewidth=0,  # 去掉点的描边，提高渲染速度
#         ax=axes[1],
#         legend="full"
#     )
#
#     axes[1].set_title(f't-SNE Visualization ({agent_name} - Full Data)', fontsize=16, fontweight='bold')
#     axes[1].set_xlabel('Dim 1', fontsize=14)
#     axes[1].set_ylabel('Dim 2', fontsize=14)
#
#     # 调整 Legend 位置到图外，防止遮挡
#     sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1), title="Class ID")
#
#     # 保存图片
#     save_fn = f"analysis_full_{agent_name}_{CONFIG['eval_data_name']}.png"
#     save_path = os.path.join(current_script_dir, save_fn)
#
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"\nDONE! Plot saved to: {save_path}")
#     plt.show()


# # if __name__ == '__main__':
# #     run_analysis_full()
#
# def run_analysis_full():
#     # --- A. 初始化环境 ---
#     seed_fixer(CONFIG["seed"])
#     device = CONFIG["device"]
#     print(f"Using device: {device}")
#
#     # ... (加载模型和数据流部分保持不变) ...
#     # 假设代码运行到这里，dataloader 已经准备好，teacher_model 已经加载
#     base_result_path = os.path.join(project_root, CONFIG["result_root_dir"], CONFIG["exp_dataset_dir"])
#     teacher_exp_path = CONFIG["exp_folder_name"]
#     teacher_ckpt = os.path.join(base_result_path, teacher_exp_path, CONFIG["ckpt_filename"])
#
#     # 重新加载模型的简略代码（为了上下文完整）
#     args_t = create_model_args(CONFIG)
#     teacher_model = setup_model(args_t)
#     if os.path.exists(teacher_ckpt):
#         full_state_dict = torch.load(teacher_ckpt, map_location=device)
#         encoder_only_dict = {k: v for k, v in full_state_dict.items() if 'head' not in k}
#         teacher_model.load_state_dict(encoder_only_dict, strict=False)
#     teacher_model.to(device)
#     teacher_model.eval()
#
#     # ... (数据加载部分 dataloader 准备好) ...
#     # 这里为了代码能跑，重新模拟一下 dataloader 变量的存在
#     # 实际运行时请保留你原代码中的 task_stream 和 dataloader 初始化逻辑
#     if 'dataloader' not in locals():
#         # 如果是作为独立片段，这里需要上下文，但如果是替换原函数，直接往下看
#         pass
#
#         # --- D. 特征提取 & 能量计算 & [新增] DC统计 ---
#     print("\n[3/5] Extracting features and calculating energy...")
#     fd_filter = TriBandFDFilter().to(device)
#
#     ratios = []  # 存储低频能量占比
#     features_list = []
#     labels_list = []
#
#     # === [新增] 存储 DC 统计数据的列表 ===
#     dc_max_list = []  # 存储每个样本中最大的那个 DC 值
#     dc_min_list = []  # 存储每个样本中最小的那个 DC 值
#     dc_mean_list = []  # 存储每个样本的平均 DC 值
#
#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.to(device)
#             # 1. 获取特征图 (Batch, Channel, Time)
#             fmap = teacher_model.feature_map(x)
#
#             # 2. 能量计算
#             dc, low_only, high = fd_filter(fmap)
#
#             # === [新增] DC 极值计算 ===
#             # dc shape: [B, C, T]。因为 DC 在 T 维度是常数，我们取平均消除维度
#             dc_val = torch.mean(dc, dim=2)  # Shape: [B, C]
#
#             # 计算当前 Batch 每个样本的最大 DC 和 最小 DC
#             # max(dim=1) 返回 (values, indices)，我们只取 values
#             batch_max_dc = torch.max(dc_val, dim=1).values
#             batch_min_dc = torch.min(dc_val, dim=1).values
#             batch_mean_dc = torch.mean(dc_val, dim=1)
#
#             dc_max_list.extend(batch_max_dc.cpu().numpy().tolist())
#             dc_min_list.extend(batch_min_dc.cpu().numpy().tolist())
#             dc_mean_list.extend(batch_mean_dc.cpu().numpy().tolist())
#             # =========================
#
#             energy_low = torch.sum((dc + low_only) ** 2, dim=(1, 2))
#             energy_high = torch.sum(high ** 2, dim=(1, 2))
#             total_ac_energy = energy_low + energy_high + 1e-8
#             sample_ratios = energy_low / total_ac_energy
#             ratios.extend(sample_ratios.cpu().numpy().tolist())
#
#             # 3. 特征池化 (GAP) 用于 t-SNE
#             pooled_features = torch.mean(fmap, dim=2)
#             features_list.append(pooled_features.cpu().numpy())
#             labels_list.append(y.numpy())
#
#     ratios = np.array(ratios)
#     all_features = np.concatenate(features_list, axis=0)
#     all_labels = np.concatenate(labels_list, axis=0)
#
#     # === [新增] 转换 DC 统计数据并计算全局极值 ===
#     dc_max_arr = np.array(dc_max_list)
#     dc_min_arr = np.array(dc_min_list)
#     dc_mean_arr = np.array(dc_mean_list)
#
#     # 全局统计
#     global_max_dc = np.max(dc_max_arr)  # 整个数据集中最大的 DC 值
#     global_min_dc = np.min(dc_min_arr)  # 整个数据集中最小的 DC 值
#     avg_max_dc = np.mean(dc_max_arr)  # 所有样本最大 DC 的平均值
#     avg_min_dc = np.mean(dc_min_arr)  # 所有样本最小 DC 的平均值
#
#     # 打印基础统计
#     print("-" * 40)
#     print(f"  -> Mean Low-Freq Ratio : {np.mean(ratios):.4f}")
#     print(f"  -> [DC Stats] Global Max DC : {global_max_dc:.4f}")
#     print(f"  -> [DC Stats] Global Min DC : {global_min_dc:.4f}")
#     print(f"  -> [DC Stats] Avg Sample Max: {avg_max_dc:.4f}")
#     print(f"  -> [DC Stats] Avg Sample Min: {avg_min_dc:.4f}")
#     print(f"  -> [DC Stats] Overall Mean  : {np.mean(dc_mean_arr):.4f}")
#     print("-" * 40)
#
#     # --- E. 运行 t-SNE ---
#     # ... (保持不变) ...
#     print(f"\n[4/5] Running t-SNE on {len(all_features)} samples...")
#     tsne = TSNE(
#         n_components=2,
#         perplexity=CONFIG["tsne_perplexity"],
#         n_iter=CONFIG["tsne_iter"],
#         random_state=CONFIG["seed"],
#         init='pca',
#         learning_rate='auto',
#         n_jobs=-1
#     )
#     X_embedded = tsne.fit_transform(all_features)
#     print("  -> t-SNE calculation finished.")
#
#     # --- F. 可视化绘图 ---
#     print("\n[5/5] Generating plots...")
#     sns.set_theme(style="whitegrid")
#
#     # 修改: 创建 2x2 的图表，把 DC 的分布也画出来 (或者保持 1x2，这里展示 1x3 更丰富)
#     # 为了直观，我们改为 3 个子图
#     fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 变宽一点
#     agent_name = CONFIG["exp_folder_name"].split('_')[0]
#
#     # Plot 1: 能量分布 (保持不变)
#     sns.histplot(ratios, bins=50, kde=True, color='royalblue', edgecolor='black', alpha=0.7, ax=axes[0])
#     axes[0].axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ratios):.2f}')
#     axes[0].set_title(f'Feature Energy Distribution', fontsize=14, fontweight='bold')
#     axes[0].set_xlabel('Low-Frequency Ratio', fontsize=12)
#     axes[0].legend()
#
#     # === [新增] Plot 2: DC 值分布 (Max/Min/Mean) ===
#     # 绘制每个样本的 最大DC 和 最小DC 的分布
#     sns.kdeplot(dc_max_arr, fill=True, color='firebrick', label='Max DC per sample', ax=axes[1])
#     sns.kdeplot(dc_min_arr, fill=True, color='teal', label='Min DC per sample', ax=axes[1])
#     # 标出全局极值
#     axes[1].axvline(global_max_dc, color='red', linestyle=':', label=f'Global Max: {global_max_dc:.2f}')
#     axes[1].axvline(global_min_dc, color='green', linestyle=':', label=f'Global Min: {global_min_dc:.2f}')
#
#     axes[1].set_title(f'DC Component Range Analysis', fontsize=14, fontweight='bold')
#     axes[1].set_xlabel('DC Value', fontsize=12)
#     axes[1].set_ylabel('Density', fontsize=12)
#     axes[1].legend()
#
#     # Plot 3: t-SNE 散点图 (原来的 Plot 2)
#     scatter = sns.scatterplot(
#         x=X_embedded[:, 0],
#         y=X_embedded[:, 1],
#         hue=all_labels,
#         palette="tab20",
#         s=10,
#         alpha=0.6,
#         linewidth=0,
#         ax=axes[2],  # 改为 axes[2]
#         legend="full"
#     )
#
#     axes[2].set_title(f't-SNE Visualization', fontsize=14, fontweight='bold')
#     axes[2].set_xlabel('Dim 1', fontsize=12)
#     axes[2].set_ylabel('Dim 2', fontsize=12)
#     sns.move_legend(axes[2], "upper left", bbox_to_anchor=(1, 1), title="Class ID")
#
#     # 保存图片
#     save_fn = f"analysis_dc_full_{agent_name}_{CONFIG['eval_data_name']}.png"
#     save_path = os.path.join(current_script_dir, save_fn)
#
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"\nDONE! Plot saved to: {save_path}")
#     plt.show()


if __name__ == '__main__':
    run_analysis_full()