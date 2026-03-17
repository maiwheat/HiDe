# -*- coding: UTF-8 -*-
import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace
from sklearn.manifold import TSNE

# ==============================================================================
#                               1. 环境与路径设置
# ==============================================================================
# 获取当前脚本所在目录，并将项目根目录加入系统路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)  # 假设脚本在项目根目录的下一级
os.chdir(project_root)
sys.path.insert(0, project_root)

# 导入项目组件 (请确保这些路径在你的项目中存在)
# 如果报错 ImportError，请检查你的文件夹结构
try:
    from models.base import setup_model
    from utils.stream import IncrementalTaskStream
    from utils.data import Dataloader_from_numpy
    from utils.utils import TriBandFDFilter, seed_fixer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the correct project structure.")
    sys.exit(1)

# ==============================================================================
#                               2. 全局配置参数 (CONFIG)
# ==============================================================================
CONFIG = {
    # --- [关键] 实验路径设置 (请修改为你实际的路径) ---
    "result_root_dir": "result/tune_and_exp",  # 结果存放的根目录
    "exp_dataset_dir": "CNN_uwave",  # 数据集对应的文件夹
    "exp_folder_name": "Offline_BN_Jan-15-16-02-42",  # 具体实验文件夹名
    "ckpt_filename": "ckpt_r0.pt",  # 权重文件名

    # --- [关键] 评估数据集设置 ---
    "eval_data_name": "uwave",  # 数据集名称 ('har', 'wisdm', 'dailysports')
    "eval_cls_order": list(range(8)),  # DailySports 通常是 19 类 (0-18)
    "batch_size": 64,  # 推理时的 Batch Size

    # --- 模型架构参数 (必须与训练时一致) ---
    "model": {
        "encoder": "CNN",
        "head": "Linear",
        "norm": "BN",  # 'LN' 或 'BN'
        "feature_dim": 128,
        "n_layers": 4,
        "dropout": 0,
        "input_norm": "IN",
        "agent": "Offline"
    },

    # --- t-SNE 参数 ---
    "tsne_perplexity": 40,  # 数据量大时适当调大 (30-50)
    "tsne_iter": 1000,  # 迭代次数

    # --- 其他 ---
    "seed": 1234,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
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
    # --- A. 初始化环境 ---
    seed_fixer(CONFIG["seed"])
    device = CONFIG["device"]
    print(f"Using device: {device}")

    # 构造完整路径
    base_result_path = os.path.join(project_root, CONFIG["result_root_dir"], CONFIG["exp_dataset_dir"])
    teacher_exp_path = CONFIG["exp_folder_name"]
    teacher_ckpt = os.path.join(base_result_path, teacher_exp_path, CONFIG["ckpt_filename"])

    # --- B. 加载模型 (仅 Encoder) ---
    print(f"\n[1/5] Loading model from: {teacher_ckpt}")
    args_t = create_model_args(CONFIG)
    try:
        teacher_model = setup_model(args_t)

        if os.path.exists(teacher_ckpt):
            full_state_dict = torch.load(teacher_ckpt, map_location=device)
            # 过滤掉 Head 层的权重，只加载 Encoder
            encoder_only_dict = {k: v for k, v in full_state_dict.items() if 'head' not in k}
            msg = teacher_model.load_state_dict(encoder_only_dict, strict=False)
            print(f"Successfully loaded encoder weights.")
            if msg.missing_keys: print(f"  Missing: {msg.missing_keys}")
            if msg.unexpected_keys: print(f"  Unexpected: {msg.unexpected_keys}")
        else:
            print(f"Error: Checkpoint file not found at {teacher_ckpt}")
            return
    except Exception as e:
        print(f"Model setup failed: {e}")
        return

    teacher_model.to(device)
    teacher_model.eval()

    # --- C. 准备数据流 ---
    print(f"\n[2/5] Setting up data stream for: {CONFIG['eval_data_name']}")
    try:
        task_stream = IncrementalTaskStream(
            data=CONFIG['eval_data_name'],
            scenario='class',
            cls_order=CONFIG['eval_cls_order'],
            split='exp'
        )
        task_stream.setup(load_subject=False)

        # 收集所有任务的测试集数据
        all_x, all_y = [], []
        for t_idx in range(len(task_stream.tasks)):
            x_tmp, y_tmp = task_stream.tasks[t_idx][2]  # Index 2 是 Test set
            all_x.append(x_tmp)
            all_y.append(y_tmp)

        x_test_total = np.concatenate(all_x)
        y_test_total = np.concatenate(all_y)
        print(f"Total samples (FULL DATASET): {len(y_test_total)}")

        dataloader = Dataloader_from_numpy(
            x_test_total,
            y_test_total,
            batch_size=CONFIG["batch_size"],
            shuffle=False
        )
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # --- D. 特征提取 & 能量计算 & DC统计 ---
    print("\n[3/5] Extracting features and calculating statistics...")
    fd_filter = TriBandFDFilter().to(device)

    # 数据容器
    ratios = []  # 低频能量占比
    features_list = []  # 聚类用特征
    labels_list = []

    # [新增] DC 统计容器
    dc_max_list = []  # 存每个样本的最大 DC
    dc_min_list = []  # 存每个样本的最小 DC

    total_batches = len(y_test_total) // CONFIG["batch_size"] + 1

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            # 1. 获取特征图 (Batch, Channel, Time)
            fmap = teacher_model.feature_map(x)

            # 2. 频域滤波
            dc, low_only, high = fd_filter(fmap)

            # === [核心修改] DC 极值统计 ===
            # DC 在 Time 维度是常数，先压缩维度: [B, C, T] -> [B, C]
            # 使用 mean 是为了消除浮点误差，其实数值都一样
            dc_val = torch.mean(dc, dim=2)

            # 计算当前 Batch 中，每个样本在所有通道中的 Max 和 Min
            # values shape: [B]
            batch_max_dc = torch.max(dc_val, dim=1).values
            batch_min_dc = torch.min(dc_val, dim=1).values

            dc_max_list.extend(batch_max_dc.cpu().numpy().tolist())
            dc_min_list.extend(batch_min_dc.cpu().numpy().tolist())
            # ============================

            # 3. 能量计算
            energy_low = torch.sum((dc + low_only) ** 2, dim=(1, 2))
            energy_high = torch.sum(high ** 2, dim=(1, 2))
            total_ac_energy = energy_low + energy_high + 1e-8
            sample_ratios = energy_low / total_ac_energy
            ratios.extend(sample_ratios.cpu().numpy().tolist())

            # 4. 特征池化 (GAP) 用于 t-SNE
            pooled_features = torch.mean(fmap, dim=2)
            features_list.append(pooled_features.cpu().numpy())
            labels_list.append(y.numpy())

            if batch_idx % 10 == 0:
                print(f"\r  Processing batch {batch_idx}/{total_batches}", end="")

    print("\n  -> Feature extraction complete.")

    # 转换格式
    ratios = np.array(ratios)
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    dc_max_arr = np.array(dc_max_list)
    dc_min_arr = np.array(dc_min_list)

    # --- 计算全局统计量 ---
    global_max_dc = np.max(dc_max_arr)  # 整个数据集见过的最大值
    global_min_dc = np.min(dc_min_arr)  # 整个数据集见过的最小值
    avg_sample_max = np.mean(dc_max_arr)  # 平均每个样本的最大值
    avg_sample_min = np.mean(dc_min_arr)  # 平均每个样本的最小值

    # 打印详细报告
    print("\n" + "=" * 50)
    print("ANALYSIS REPORT")
    print("=" * 50)
    print(f"1. Low-Frequency Energy Ratio (Mean): {np.mean(ratios):.4f}")
    print("-" * 30)
    print("2. DC Component Statistics (Intensity):")
    print(f"   - Global Max DC Value : {global_max_dc:.4f}")
    print(f"   - Global Min DC Value : {global_min_dc:.4f}")
    print(f"   - Avg Sample Max DC   : {avg_sample_max:.4f} (Typical Peak)")
    print(f"   - Avg Sample Min DC   : {avg_sample_min:.4f} (Typical Valley)")
    print("=" * 50)

    # --- E. 运行 t-SNE ---
    print(f"\n[4/5] Running t-SNE on {len(all_features)} samples...")
    # 如果数据量太大(>10000)，为了速度可以先采样，这里默认跑全量
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
    print("  -> t-SNE calculation finished.")

    # --- F. 可视化绘图 (3子图布局) ---
    print("\n[5/5] Generating plots...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    agent_name = CONFIG["exp_folder_name"].split('_')[0]

    # --- 子图 1: 能量分布 (Energy Ratio) ---
    sns.histplot(ratios, bins=50, kde=True, color='royalblue', edgecolor=None, alpha=0.6, ax=axes[0])
    axes[0].axvline(np.mean(ratios), color='navy', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ratios):.2f}')
    axes[0].set_title(f'Low-Freq Energy Ratio', fontweight='bold')
    axes[0].set_xlabel('Ratio (Low / Total)')
    axes[0].set_xlim(0, 1.0)
    axes[0].legend()

    # --- 子图 2: DC 极值分布 (DC Stats) [新增] ---
    # 绘制 "每个样本的最大值" 分布
    sns.kdeplot(dc_max_arr, fill=True, color='crimson', alpha=0.3, label='Max DC per Sample', ax=axes[1])
    # 绘制 "每个样本的最小值" 分布
    sns.kdeplot(dc_min_arr, fill=True, color='teal', alpha=0.3, label='Min DC per Sample', ax=axes[1])

    # 标注全局极值线
    axes[1].axvline(global_max_dc, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
    axes[1].text(global_max_dc, axes[1].get_ylim()[1] * 0.9, f' Global Max\n {global_max_dc:.2f}', color='red',
                 ha='right')

    axes[1].axvline(global_min_dc, color='teal', linestyle=':', linewidth=1.5, alpha=0.8)
    axes[1].text(global_min_dc, axes[1].get_ylim()[1] * 0.9, f'Global Min \n{global_min_dc:.2f} ', color='teal',
                 ha='left')

    axes[1].set_title(f'DC Component Dynamic Range', fontweight='bold')
    axes[1].set_xlabel('Feature Value Magnitude')
    axes[1].set_ylabel('Density')
    axes[1].legend(loc='upper center')

    # --- 子图 3: t-SNE 散点图 ---
    scatter = sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=all_labels,
        palette="tab20",  # 适合多类别
        s=15,
        alpha=0.7,
        edgecolor=None,
        ax=axes[2],
        legend="full"
    )
    axes[2].set_title(f'Feature Space (t-SNE)', fontweight='bold')
    axes[2].set_xlabel('Dim 1')
    axes[2].set_ylabel('Dim 2')

    # 优化 Legend 位置
    sns.move_legend(axes[2], "upper left", bbox_to_anchor=(1, 1), title="Class ID", frameon=False)

    # 保存与展示
    save_fn = f"analysis_FullReport_{agent_name}_{CONFIG['eval_data_name']}.png"
    save_path = os.path.join(current_script_dir, save_fn)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDONE! Analysis plot saved to: {save_path}")
    # plt.show() # 如果在无界面服务器运行，请注释此行


if __name__ == '__main__':
    run_analysis_full()