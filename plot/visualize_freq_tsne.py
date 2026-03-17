# -*- coding: UTF-8 -*-


# -*- coding: UTF-8 -*-
"""
Feature Alignment t-SNE Visualization Script (UWave Fixed Version)
Features:
1. Adaptive model loading.
2. Configurable task selection.
3. [Update] Saves separate images for Baseline and Ours.
4. [Update] Uses circular markers only.
5. [Fix] Legend moved inside the plot (Top-Right).
6. [Fix] Forces plt.show() to block and display images before closing.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as PathEffects

# Force Agg backend to prevent server errors if needed
# matplotlib.use('Agg')
from sklearn.manifold import TSNE
from types import SimpleNamespace
import matplotlib.cm as cm

# Setup Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from models.base import setup_model
from utils.stream import IncrementalTaskStream
from utils.data import Dataloader_from_numpy
from utils.utils import seed_fixer


# ==========================================
# Core Functions
# ==========================================
def load_checkpoint_robust(model, ckpt_path, device):
    """
    增强版权重加载：
    1. 自动缝合分裂的分类头 (fc1 + fc2 -> fc)
    2. 自动处理 Cosine Head 的 sigma 参数
    3. 自动调整维度不匹配
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1. 加载原始权重字典
    state_dict = torch.load(ckpt_path, map_location=device)

    # --- 修复 1: 处理分裂的分类头 (Split Head to Unified Head) ---
    # 如果存档里是 fc1 和 fc2 (增量学习常见结构)，我们将它们拼起来变成 fc
    if 'head.fc1.weight' in state_dict and 'head.fc2.weight' in state_dict:
        # print("Detected Split Head (fc1/fc2), merging into single fc...")
        w1 = state_dict.pop('head.fc1.weight')
        w2 = state_dict.pop('head.fc2.weight')
        state_dict['head.fc.weight'] = torch.cat([w1, w2], dim=0)

        # 如果有 bias 也拼起来
        if 'head.fc1.bias' in state_dict and 'head.fc2.bias' in state_dict:
            b1 = state_dict.pop('head.fc1.bias')
            b2 = state_dict.pop('head.fc2.bias')
            state_dict['head.fc.bias'] = torch.cat([b1, b2], dim=0)

    # --- 修复 2: 处理 Cosine Head 的 sigma ---
    # 我们只做特征提取，不需要 sigma，直接删掉以防止报错
    if 'head.sigma' in state_dict:
        # print("Detected Cosine Head sigma, removing it for compatibility...")
        state_dict.pop('head.sigma')

    # --- 修复 3: 补充缺失的 Bias (如果模型需要但存档里没有) ---
    # Cosine Head 通常没有 bias，但 Linear Head 需要。如果缺失，我们填 0。
    if 'head.fc.weight' in state_dict and 'head.fc.bias' not in state_dict:
        # 检查代码中的模型是否需要 bias
        if hasattr(model.head.fc, 'bias') and model.head.fc.bias is not None:
            # print("Filling missing bias with zeros...")
            out_dim = state_dict['head.fc.weight'].shape[0]
            state_dict['head.fc.bias'] = torch.zeros(out_dim).to(device)

    # --- 修复 4: 调整维度 (Resizing) ---
    if 'head.fc.weight' in state_dict:
        saved_weight = state_dict['head.fc.weight']
        saved_out_features = saved_weight.shape[0]
        saved_in_features = saved_weight.shape[1]

        current_out = model.head.fc.out_features

        # 如果维度不一样，强制修改代码中的模型结构
        if current_out != saved_out_features:
            print(
                f"Warning: Model head mismatch! Resizing code model from {current_out} to {saved_out_features} to match checkpoint.")
            model.head.fc = nn.Linear(saved_in_features, saved_out_features)

    # 2. 安全加载
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def extract_all_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            f_map = model.feature_map(x)
            f_flat = f_map.view(f_map.size(0), -1)
            features_list.append(f_flat.cpu().numpy())
            labels_list.append(y.numpy())
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


def plot_tsne_comparison_separate(feat_t, feat_b, feat_o, labels, save_dir, task_str,
                                  show_teacher=True, show_baseline=True, show_ours=True,
                                  show_legend=False):  # Added show_legend param
    """
    Plots and saves separate images for Baseline and Ours.
    """
    print("Computing t-SNE... Please wait.")

    # 1. Dynamic Stacking
    feats_to_stack = []
    if show_teacher: feats_to_stack.append(feat_t)
    if show_baseline: feats_to_stack.append(feat_b)
    if show_ours: feats_to_stack.append(feat_o)

    if not feats_to_stack:
        print("Error: Nothing to plot!")
        return

    combined_feats = np.vstack(feats_to_stack)

    # 2. Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
    embedded = tsne.fit_transform(combined_feats)

    # 3. Dynamic Unpacking
    cursor = 0
    tsne_t, tsne_b, tsne_o = None, None, None

    def get_slice(feat_array):
        nonlocal cursor
        n = feat_array.shape[0]
        res = embedded[cursor: cursor + n]
        cursor += n
        return res

    if show_teacher: tsne_t = get_slice(feat_t)
    if show_baseline: tsne_b = get_slice(feat_b)
    if show_ours: tsne_o = get_slice(feat_o)

    # ---------------------------------------------------------
    # Helper function to plot ONE specific configuration
    # ---------------------------------------------------------
    def save_and_show_single_plot(foreground_tsne, name_suffix, title_text):
        # Create a single figure
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)

        # Colormap
        if hasattr(matplotlib, 'colormaps'):
            cmap = matplotlib.colormaps['tab10'] if n_classes <= 10 else matplotlib.colormaps['tab20']
        else:
            cmap = cm.get_cmap('tab10') if n_classes <= 10 else cm.get_cmap('tab20')

        # A. Background: Teacher
        if show_teacher and tsne_t is not None:
            for i, cls in enumerate(unique_classes):
                idx = np.where(labels == cls)[0]
                color = cmap(i)
                alpha_val = 0.2 if foreground_tsne is not None else 0.6
                ax.scatter(tsne_t[idx, 0], tsne_t[idx, 1], color=color, marker='o',
                           alpha=alpha_val, s=150, label='_nolegend_')

        # B. Foreground: Student
        if foreground_tsne is not None:
            for i, cls in enumerate(unique_classes):
                idx = np.where(labels == cls)[0]
                color = cmap(i)
                ax.scatter(foreground_tsne[idx, 0], foreground_tsne[idx, 1],
                           color=color, marker='o',
                           alpha=0.9, s=40, edgecolors='white', linewidth=0.5,
                           label=f'Class {cls}')

        # Title
        title_suffix = " vs Teacher" if (show_teacher and foreground_tsne is not None) else ""
        ax.set_title(f"{title_text}{title_suffix}", fontsize=20, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Legend Logic (Controlled by show_legend flag) ---
        if show_legend:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=f'Class {int(cls)}',
                       markerfacecolor=cmap(i), markersize=12)
                for i, cls in enumerate(unique_classes)
            ]

            if show_teacher:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Teacher (Back)',
                                              markerfacecolor='gray', markersize=15, alpha=0.3))

            if foreground_tsne is not None:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Student (Front)',
                                              markerfacecolor='gray', markersize=8, alpha=0.9))

            ax.legend(handles=legend_elements,
                      loc='upper right',
                      fontsize=12,
                      title="Legend",
                      frameon=True,
                      framealpha=0.9,
                      facecolor='white',
                      edgecolor='black')

        plt.tight_layout()

        # Save file
        config_str = f"T{int(show_teacher)}"
        filename = f'tsne_tasks{task_str}_{name_suffix}_{config_str}.png'
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"Saved: {full_path}")

        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not show plot (likely no GUI). Error: {e}")

        plt.close(fig)

        # 4. Execute Saving

    if show_baseline:
        save_and_show_single_plot(tsne_b, "Baseline", "(a) Baseline (DT2W)")

    if show_ours:
        save_and_show_single_plot(tsne_o, "Ours", "(b) HiDe (Ours)")

    if not show_baseline and not show_ours and show_teacher:
        save_and_show_single_plot(None, "TeacherOnly", "Teacher Distribution")


# ==========================================
# Configuration & Main
# ==========================================
def create_model_args(data='uwave', encoder='CNN', head='Linear', norm='BN',
                      feature_dim=128, n_layers=4, dropout=0, device='cuda',
                      agent='Offline', input_norm='IN'):
    args = SimpleNamespace(
        data=data, encoder=encoder, head=head, norm=norm,
        feature_dim=feature_dim, n_layers=n_layers, dropout=dropout,
        device=device, agent=agent, input_norm=input_norm, stream_split='exp'
    )
    return args


def main():
    # ==========================
    # 0. User Configuration
    # ==========================
    TASKS_TO_VISUALIZE = [0, 1, 2, 3]

    # --- Toggles ---
    SHOW_TEACHER = False
    SHOW_BASELINE = False
    SHOW_OURS = True

    # [New Config] Show Legend?
    SHOW_LEGEND = False  # Set to False to hide legend

    # Experiment Paths
    base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_uwave')
    ours_exp_path = 'Dkfd_BN_Jan-21-14-54-04'
    # ours_exp_path = 'Dkfd_BN_Jan-21-14-54-04'
    baseline_exp_path = 'DT2W_BN_Dec-07-23-32-49'

    teacher_ckpt = os.path.join(base_result_path, baseline_exp_path, 'teacher_after_task0_r0.pt')
    ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'ckpt_r3.pt')
    baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'ckpt_r3.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    seed_fixer(0)

    # ==========================
    # 1. Data Preparation
    # ==========================
    cls_order = [5, 0, 6, 7, 2, 4, 1, 3]
    task_stream = IncrementalTaskStream(data='uwave', scenario='class', cls_order=cls_order, split='exp')
    task_stream.setup(load_subject=False)

    x_list, y_list = [], []
    for t_idx in TASKS_TO_VISUALIZE:
        if t_idx < len(task_stream.tasks):
            xt, yt = task_stream.tasks[t_idx][2]
            x_list.append(xt)
            y_list.append(yt)
            print(f"Added Task {t_idx} data: {xt.shape}")
        else:
            print(f"Warning: Task {t_idx} out of range, skipped.")

    x_test = np.concatenate(x_list, axis=0)
    y_test = np.concatenate(y_list, axis=0)
    print(f"Total Test Data Shape: {x_test.shape}")

    test_loader = Dataloader_from_numpy(x_test, y_test, batch_size=64, shuffle=False)

    # ==========================
    # 2. Model Loading & Feature Extraction
    # ==========================

    # --- Teacher ---
    feat_t, labels = None, None
    if SHOW_TEACHER:
        print(">>> Loading Teacher...")
        args_t = create_model_args(data='uwave', device=device, agent='Dkfd', head='Linear')
        teacher_model = setup_model(args_t)
        teacher_model = load_checkpoint_robust(teacher_model, teacher_ckpt, device)
        feat_t, labels = extract_all_features(teacher_model, test_loader, device)

    # --- Baseline ---
    feat_b = None
    if SHOW_BASELINE:
        print(">>> Loading Baseline...")
        args_b = create_model_args(data='uwave', device=device, agent='DT2W', head='Linear')
        baseline_model = setup_model(args_b)
        baseline_model = load_checkpoint_robust(baseline_model, baseline_ckpt, device)
        feat_b, labels_b = extract_all_features(baseline_model, test_loader, device)
        if labels is None: labels = labels_b

    # --- Ours ---
    feat_o = None
    if SHOW_OURS:
        print(">>> Loading Ours...")
        args_o = create_model_args(data='uwave', device=device, agent='Dkfd', head='Linear')
        ours_model = setup_model(args_o)
        ours_model = load_checkpoint_robust(ours_model, ours_ckpt, device)
        feat_o, labels_o = extract_all_features(ours_model, test_loader, device)
        if labels is None: labels = labels_o

    # ==========================
    # 3. Visualization
    # ==========================
    save_dir = os.path.join(base_result_path, ours_exp_path, 'tsne_visualization')
    os.makedirs(save_dir, exist_ok=True)

    task_str = "-".join(map(str, TASKS_TO_VISUALIZE))

    plot_tsne_comparison_separate(feat_t, feat_b, feat_o, labels, save_dir, task_str,
                                  show_teacher=SHOW_TEACHER,
                                  show_baseline=SHOW_BASELINE,
                                  show_ours=SHOW_OURS,
                                  show_legend=SHOW_LEGEND)


if __name__ == '__main__':
    main()

# # -*- coding: UTF-8 -*-
# """
# Feature Alignment t-SNE Visualization Script (UWave Fixed Version)
# Features:
# 1. Adaptive model loading.
# 2. Configurable task selection.
# 3. Toggle visibility for Teacher, Baseline, and Ours independently.
# 4. [Fix] Can now visualize Teacher alone.
# """
#
# import os
# import sys
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
#
# # Force Agg backend to prevent server errors
# # matplotlib.use('Agg')
# from sklearn.manifold import TSNE
# from types import SimpleNamespace
# import matplotlib.cm as cm
#
# # Setup Paths
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(project_root)
# sys.path.insert(0, project_root)
#
# from models.base import setup_model
# from utils.stream import IncrementalTaskStream
# from utils.data import Dataloader_from_numpy
# from utils.utils import seed_fixer
#
#
# # ==========================================
# # Core Functions
# # ==========================================
#
# def load_checkpoint_robust(model, ckpt_path, device):
#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
#     state_dict = torch.load(ckpt_path, map_location=device)
#     if 'head.fc.weight' in state_dict:
#         saved_weight = state_dict['head.fc.weight']
#         saved_out_features = saved_weight.shape[0]
#         saved_in_features = saved_weight.shape[1]
#         current_out = model.head.fc.out_features
#         if current_out != saved_out_features:
#             print(
#                 f"Warning: Model head mismatch! Resizing code model from {current_out} to {saved_out_features} to match checkpoint.")
#             model.head.fc = nn.Linear(saved_in_features, saved_out_features)
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model
#
#
# def extract_all_features(model, dataloader, device):
#     model.eval()
#     features_list = []
#     labels_list = []
#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.to(device)
#             f_map = model.feature_map(x)
#             f_flat = f_map.view(f_map.size(0), -1)
#             features_list.append(f_flat.cpu().numpy())
#             labels_list.append(y.numpy())
#     return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)
#
#
# def plot_tsne_comparison_by_class(feat_t, feat_b, feat_o, labels, save_path,
#                                   show_teacher=True, show_baseline=True, show_ours=True):
#     print("Computing t-SNE... Please wait.")
#
#     # 1. Dynamic Stacking
#     feats_to_stack = []
#     if show_teacher: feats_to_stack.append(feat_t)
#     if show_baseline: feats_to_stack.append(feat_b)
#     if show_ours: feats_to_stack.append(feat_o)
#
#     if not feats_to_stack:
#         print("Error: Nothing to plot!")
#         return
#
#     combined_feats = np.vstack(feats_to_stack)
#
#     # 2. Run t-SNE
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
#     embedded = tsne.fit_transform(combined_feats)
#
#     # 3. Dynamic Unpacking
#     cursor = 0
#     tsne_t, tsne_b, tsne_o = None, None, None
#
#     # Helper to slice embedded array
#     def get_slice(feat_array):
#         nonlocal cursor
#         n = feat_array.shape[0]
#         res = embedded[cursor: cursor + n]
#         cursor += n
#         return res
#
#     if show_teacher: tsne_t = get_slice(feat_t)
#     if show_baseline: tsne_b = get_slice(feat_b)
#     if show_ours: tsne_o = get_slice(feat_o)
#
#     # 4. Plotting Layout
#     # Determine how many subplots needed
#     plots_needed = sum([show_baseline, show_ours])
#
#     # 【修复点 1】: 如果学生都不展示，但 Teacher 展示，我们也需要 1 个子图
#     if plots_needed == 0 and show_teacher:
#         plots_needed = 1
#
#     if plots_needed == 0:
#         print("Error: No plots selected.")
#         return
#
#     fig, axes = plt.subplots(1, plots_needed, figsize=(9 * plots_needed, 8), dpi=300)
#     # Ensure axes is always iterable
#     if plots_needed == 1: axes = [axes]
#
#     unique_classes = np.unique(labels)
#     n_classes = len(unique_classes)
#
#     # Colormap
#     if hasattr(matplotlib, 'colormaps'):
#         cmap = matplotlib.colormaps['tab10'] if n_classes <= 10 else matplotlib.colormaps['tab20']
#     else:
#         cmap = cm.get_cmap('tab10') if n_classes <= 10 else cm.get_cmap('tab20')
#
#     # Helper function to plot a single panel
#     def plot_single_panel(ax, foreground_tsne, title_name):
#         # Background: Teacher
#         if show_teacher and tsne_t is not None:
#             for i, cls in enumerate(unique_classes):
#                 idx = np.where(labels == cls)[0]
#                 color = cmap(i)
#                 # Teacher alpha depends on whether there is a foreground
#                 alpha_val = 0.2 if foreground_tsne is not None else 0.6
#                 ax.scatter(tsne_t[idx, 0], tsne_t[idx, 1], color=color, alpha=alpha_val, s=100, label='_nolegend_')
#
#         # Foreground: Student (Baseline or Ours)
#         if foreground_tsne is not None:
#             for i, cls in enumerate(unique_classes):
#                 idx = np.where(labels == cls)[0]
#                 color = cmap(i)
#                 ax.scatter(foreground_tsne[idx, 0], foreground_tsne[idx, 1], color=color, alpha=0.9, s=20,
#                            edgecolors='white', linewidth=0.5, label=f'Class {cls}')
#
#         title_suffix = " vs Teacher" if (show_teacher and foreground_tsne is not None) else ""
#         ax.set_title(f"{title_name}{title_suffix}", fontsize=16, fontweight='bold')
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     # 【修复点 2】: 执行绘图逻辑
#     ax_idx = 0
#     has_plotted = False
#
#     if show_baseline:
#         plot_single_panel(axes[ax_idx], tsne_b, "(a) Baseline (DT2W)")
#         ax_idx += 1
#         has_plotted = True
#
#     if show_ours:
#         plot_single_panel(axes[ax_idx], tsne_o, "(b) HiDe (Ours)")
#         ax_idx += 1
#         has_plotted = True
#
#     # 如果两个学生都没画，但要求画 Teacher，则画单独的 Teacher 图
#     if not has_plotted and show_teacher:
#         plot_single_panel(axes[0], None, "Teacher Distribution")
#
#     # Legend
#     from matplotlib.lines import Line2D
#     legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Class {int(cls)}',
#                               markerfacecolor=cmap(i), markersize=10) for i, cls in enumerate(unique_classes)]
#
#     if show_teacher:
#         legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Teacher (Background)',
#                                       markerfacecolor='gray', markersize=15, alpha=0.3))
#
#     if show_baseline or show_ours:
#         legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Student (Foreground)',
#                                       markerfacecolor='gray', markersize=8, alpha=0.9))
#
#     fig.legend(handles=legend_elements, loc='lower center', ncol=min(len(legend_elements), 6),
#                bbox_to_anchor=(0.5, -0.05), fontsize=12)
#     plt.tight_layout()
#
#     plt.savefig(save_path, bbox_inches='tight')
#     print(f"Visualization saved to: {save_path}")
#
#
# # ==========================================
# # Configuration & Main
# # ==========================================
# def create_model_args(data='uwave', encoder='CNN', head='Linear', norm='BN',
#                       feature_dim=128, n_layers=4, dropout=0, device='cuda',
#                       agent='Offline', input_norm='IN'):
#     args = SimpleNamespace(
#         data=data, encoder=encoder, head=head, norm=norm,
#         feature_dim=feature_dim, n_layers=n_layers, dropout=dropout,
#         device=device, agent=agent, input_norm=input_norm, stream_split='exp'
#     )
#     return args
#
#
# def main():
#     # ==========================
#     # 0. User Configuration
#     # ==========================
#     TASKS_TO_VISUALIZE = [0, 1]
#
#     # --- Toggles ---
#     SHOW_TEACHER = True
#     SHOW_BASELINE = True
#     SHOW_OURS = True
#
#     # Experiment Paths
#     base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_uwave')
#     ours_exp_path = 'Dkfd_BN_Dec-29-22-09-35-judge'
#     baseline_exp_path = 'DT2W_BN_Dec-29-21-31-38'
#
#     teacher_ckpt = os.path.join(base_result_path, ours_exp_path, 'teacher_after_task0_r0.pt')
#     ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'final_model_r0.pt')
#     baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'final_model_r0.pt')
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     seed_fixer(0)
#
#     # ==========================
#     # 1. Data Preparation
#     # ==========================
#     cls_order = [2, 1, 6, 0, 3, 4, 5, 7]
#     task_stream = IncrementalTaskStream(data='uwave', scenario='class', cls_order=cls_order, split='exp')
#     task_stream.setup(load_subject=False)
#
#     x_list, y_list = [], []
#     for t_idx in TASKS_TO_VISUALIZE:
#         if t_idx < len(task_stream.tasks):
#             xt, yt = task_stream.tasks[t_idx][2]
#             x_list.append(xt)
#             y_list.append(yt)
#             print(f"Added Task {t_idx} data: {xt.shape}")
#         else:
#             print(f"Warning: Task {t_idx} out of range, skipped.")
#
#     x_test = np.concatenate(x_list, axis=0)
#     y_test = np.concatenate(y_list, axis=0)
#     print(f"Total Test Data Shape: {x_test.shape}")
#
#     test_loader = Dataloader_from_numpy(x_test, y_test, batch_size=64, shuffle=False)
#
#     # ==========================
#     # 2. Model Loading & Feature Extraction
#     # ==========================
#
#     # --- Teacher ---
#     feat_t, labels = None, None
#     if SHOW_TEACHER:
#         print(">>> Loading Teacher...")
#         args_t = create_model_args(data='uwave', device=device, agent='Dkfd', head='Linear')
#         teacher_model = setup_model(args_t)
#         teacher_model = load_checkpoint_robust(teacher_model, teacher_ckpt, device)
#         feat_t, labels = extract_all_features(teacher_model, test_loader, device)
#
#     # --- Baseline ---
#     feat_b = None
#     if SHOW_BASELINE:
#         print(">>> Loading Baseline...")
#         args_b = create_model_args(data='uwave', device=device, agent='DT2W', head='Linear')
#         baseline_model = setup_model(args_b)
#         baseline_model = load_checkpoint_robust(baseline_model, baseline_ckpt, device)
#         feat_b, labels_b = extract_all_features(baseline_model, test_loader, device)
#         if labels is None: labels = labels_b
#
#     # --- Ours ---
#     feat_o = None
#     if SHOW_OURS:
#         print(">>> Loading Ours...")
#         args_o = create_model_args(data='uwave', device=device, agent='Dkfd', head='Linear')
#         ours_model = setup_model(args_o)
#         ours_model = load_checkpoint_robust(ours_model, ours_ckpt, device)
#         feat_o, labels_o = extract_all_features(ours_model, test_loader, device)
#         if labels is None: labels = labels_o
#
#     # ==========================
#     # 3. Visualization
#     # ==========================
#     save_dir = os.path.join(base_result_path, ours_exp_path, 'tsne_visualization')
#     os.makedirs(save_dir, exist_ok=True)
#
#     task_str = "-".join(map(str, TASKS_TO_VISUALIZE))
#
#     # Construct meaningful filename based on configs
#     config_str = f"T{int(SHOW_TEACHER)}_B{int(SHOW_BASELINE)}_O{int(SHOW_OURS)}"
#     filename = f'tsne_tasks{task_str}_{config_str}.png'
#     save_path = os.path.join(save_dir, filename)
#
#     plot_tsne_comparison_by_class(feat_t, feat_b, feat_o, labels, save_path,
#                                   show_teacher=SHOW_TEACHER,
#                                   show_baseline=SHOW_BASELINE,
#                                   show_ours=SHOW_OURS)
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()


# # -*- coding: UTF-8 -*-
# """
# Feature Alignment t-SNE Visualization Script (UWave Fixed Version)
# Features:
# 1. Adaptive model loading (Fixes Head dimension mismatch).
# 2. Configurable task selection (Visualize specific tasks).
# 3. Configurable Teacher display (Toggle background).
# """
#
# import os
# import sys
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from types import SimpleNamespace
# import matplotlib.cm as cm
#
# # Setup Paths
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(project_root)
# sys.path.insert(0, project_root)
#
# from models.base import setup_model
# from utils.stream import IncrementalTaskStream
# from utils.data import Dataloader_from_numpy
# from utils.utils import seed_fixer
#
#
# # ==========================================
# # Core Functions
# # ==========================================
#
# def load_checkpoint_robust(model, ckpt_path, device):
#     """
#     Adaptive weight loading: Resizes model head if dimension mismatch occurs.
#     """
#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
#
#     # 1. Load state_dict
#     state_dict = torch.load(ckpt_path, map_location=device)
#
#     # 2. Check Head dimension
#     if 'head.fc.weight' in state_dict:
#         saved_weight = state_dict['head.fc.weight']
#         saved_out_features = saved_weight.shape[0]
#         saved_in_features = saved_weight.shape[1]
#
#         current_out = model.head.fc.out_features
#
#         # 3. Resize model if needed
#         if current_out != saved_out_features:
#             print(
#                 f"Warning: Model head mismatch! Resizing code model from {current_out} to {saved_out_features} to match checkpoint.")
#             model.head.fc = nn.Linear(saved_in_features, saved_out_features)
#
#     # 4. Safe Load
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model
#
#
# def extract_all_features(model, dataloader, device):
#     """
#     Extract features and labels from the entire dataloader.
#     """
#     model.eval()
#     features_list = []
#     labels_list = []
#
#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.to(device)
#             f_map = model.feature_map(x)
#             f_flat = f_map.view(f_map.size(0), -1)
#             features_list.append(f_flat.cpu().numpy())
#             labels_list.append(y.numpy())
#
#     return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)
#
#
# def plot_tsne_comparison_by_class(feat_t, feat_b, feat_o, labels, save_path, show_teacher=True):
#     print("Computing t-SNE... Please wait.")
#
#     # 1. Combine features based on config
#     # If show_teacher is False, we use dummy zeros for feat_t to keep logic simple,
#     # but practically we just slice the results later.
#     # To keep t-SNE space consistent, we usually include Teacher even if not plotting,
#     # but here we follow user request strictly.
#
#     if show_teacher:
#         combined_feats = np.vstack([feat_t, feat_b, feat_o])
#     else:
#         # If teacher is hidden, we perform t-SNE only on Baseline and Ours
#         combined_feats = np.vstack([feat_b, feat_o])
#
#     # 2. Run t-SNE
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
#     embedded = tsne.fit_transform(combined_feats)
#
#     # 3. Split back
#     if show_teacher:
#         n = feat_t.shape[0]
#         tsne_t = embedded[:n]
#         tsne_b = embedded[n:2 * n]
#         tsne_o = embedded[2 * n:]
#     else:
#         n = feat_b.shape[0]  # Note: here n is size of baseline (same as ours)
#         tsne_t = None
#         tsne_b = embedded[:n]
#         tsne_o = embedded[n:]
#
#     # 4. Plotting
#     fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
#
#     unique_classes = np.unique(labels)
#     n_classes = len(unique_classes)
#
#     # Colormap compatibility
#     import matplotlib
#     if hasattr(matplotlib, 'colormaps'):
#         cmap = matplotlib.colormaps['tab10'] if n_classes <= 10 else matplotlib.colormaps['tab20']
#     else:
#         cmap = cm.get_cmap('tab10') if n_classes <= 10 else cm.get_cmap('tab20')
#
#     # --- Subplot 1: Baseline ---
#     ax1 = axes[0]
#     # Teacher (Background)
#     if show_teacher and tsne_t is not None:
#         for i, cls in enumerate(unique_classes):
#             idx = np.where(labels == cls)[0]
#             color = cmap(i)
#             ax1.scatter(tsne_t[idx, 0], tsne_t[idx, 1], color=color, alpha=0.2, s=100, label='_nolegend_')
#
#     # Baseline (Foreground)
#     for i, cls in enumerate(unique_classes):
#         idx = np.where(labels == cls)[0]
#         color = cmap(i)
#         ax1.scatter(tsne_b[idx, 0], tsne_b[idx, 1], color=color, alpha=0.9, s=20, edgecolors='white', linewidth=0.5,
#                     label=f'Class {cls}')
#
#     title_suffix = " vs Teacher" if show_teacher else ""
#     ax1.set_title(f"(a) Baseline (DT2W){title_suffix}", fontsize=16, fontweight='bold')
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#
#     # --- Subplot 2: Ours ---
#     ax2 = axes[1]
#     # Teacher (Background)
#     if show_teacher and tsne_t is not None:
#         for i, cls in enumerate(unique_classes):
#             idx = np.where(labels == cls)[0]
#             color = cmap(i)
#             ax2.scatter(tsne_t[idx, 0], tsne_t[idx, 1], color=color, alpha=0.2, s=100, label='_nolegend_')
#
#     # Ours (Foreground)
#     for i, cls in enumerate(unique_classes):
#         idx = np.where(labels == cls)[0]
#         color = cmap(i)
#         ax2.scatter(tsne_o[idx, 0], tsne_o[idx, 1], color=color, alpha=0.9, s=20, edgecolors='white', linewidth=0.5,
#                     label=f'Class {cls}')
#
#     ax2.set_title(f"(b) HiDe (Ours){title_suffix}", fontsize=16, fontweight='bold')
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#
#     # Legend
#     from matplotlib.lines import Line2D
#     legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Class {int(cls)}',
#                               markerfacecolor=cmap(i), markersize=10) for i, cls in enumerate(unique_classes)]
#
#     if show_teacher:
#         legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Teacher (Background)',
#                                       markerfacecolor='gray', markersize=15, alpha=0.3))
#
#     legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Student (Foreground)',
#                                   markerfacecolor='gray', markersize=8, alpha=0.9))
#
#     fig.legend(handles=legend_elements, loc='lower center', ncol=min(len(legend_elements), 6),
#                bbox_to_anchor=(0.5, -0.05),
#                fontsize=12)
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight')
#     print(f"Visualization saved to: {save_path}")
#
#     # Try to show (will likely fail on server, handled by try-except)
#     try:
#         plt.show()
#     except Exception:
#         pass
#
#
# # ==========================================
# # Configuration & Main
# # ==========================================
# def create_model_args(data='uwave', encoder='CNN', head='Linear', norm='BN',
#                       feature_dim=128, n_layers=4, dropout=0, device='cuda',
#                       agent='Offline', input_norm='IN'):
#     args = SimpleNamespace(
#         data=data, encoder=encoder, head=head, norm=norm,
#         feature_dim=feature_dim, n_layers=n_layers, dropout=dropout,
#         device=device, agent=agent, input_norm=input_norm, stream_split='exp'
#     )
#     return args
#
#
# def main():
#     # ==========================
#     # 0. User Configuration
#     # ==========================
#     # Define which tasks to visualize by index (e.g., [0] for task0 only, [0, 1] for task0+1)
#     TASKS_TO_VISUALIZE = [0]  # Change as needed
#
#     # Toggle Teacher display (True/False)
#     SHOW_TEACHER = True
#
#     # Experiment Paths (UWave)
#     base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_uwave')
#     ours_exp_path = 'Dkfd_BN_Dec-29-22-09-35-judge'
#     baseline_exp_path = 'DT2W_BN_Dec-29-21-31-38'
#
#     teacher_ckpt = os.path.join(base_result_path, ours_exp_path, 'teacher_after_task0_r0.pt')
#     ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'final_model_r0.pt')
#     baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'final_model_r0.pt')
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     seed_fixer(0)
#
#     # ==========================
#     # 1. Data Preparation
#     # ==========================
#     cls_order = [2, 1, 6, 0, 3, 4, 5, 7]
#     task_stream = IncrementalTaskStream(data='uwave', scenario='class', cls_order=cls_order, split='exp')
#     task_stream.setup(load_subject=False)
#
#     x_list, y_list = [], []
#     for t_idx in TASKS_TO_VISUALIZE:
#         if t_idx < len(task_stream.tasks):
#             xt, yt = task_stream.tasks[t_idx][2]
#             x_list.append(xt)
#             y_list.append(yt)
#             print(f"Added Task {t_idx} data: {xt.shape}")
#         else:
#             print(f"Warning: Task {t_idx} out of range, skipped.")
#
#     x_test = np.concatenate(x_list, axis=0)
#     y_test = np.concatenate(y_list, axis=0)
#     print(f"Total Test Data Shape: {x_test.shape}")
#
#     test_loader = Dataloader_from_numpy(x_test, y_test, batch_size=64, shuffle=False)
#
#     # ==========================
#     # 2. Model Loading & Feature Extraction
#     # ==========================
#
#     # --- Teacher Model ---
#     feat_t = None
#     if SHOW_TEACHER:
#         print(">>> Loading Teacher...")
#         args_t = create_model_args(data='uwave', device=device, agent='Dkfd', head='Linear')
#         teacher_model = setup_model(args_t)
#         teacher_model = load_checkpoint_robust(teacher_model, teacher_ckpt, device)
#         feat_t, labels = extract_all_features(teacher_model, test_loader, device)
#     else:
#         # If not showing teacher, we still need labels from somewhere.
#         # We can get them from baseline extraction.
#         labels = None
#
#     # --- Baseline Model ---
#     print(">>> Loading Baseline...")
#     args_b = create_model_args(data='uwave', device=device, agent='DT2W', head='Linear')
#     baseline_model = setup_model(args_b)
#     baseline_model = load_checkpoint_robust(baseline_model, baseline_ckpt, device)
#     feat_b, labels_b = extract_all_features(baseline_model, test_loader, device)
#
#     if labels is None: labels = labels_b
#
#     # --- Ours Model ---
#     print(">>> Loading Ours...")
#     args_o = create_model_args(data='uwave', device=device, agent='Dkfd', head='Linear')
#     ours_model = setup_model(args_o)
#     ours_model = load_checkpoint_robust(ours_model, ours_ckpt, device)
#     feat_o, _ = extract_all_features(ours_model, test_loader, device)
#
#     # ==========================
#     # 3. Visualization
#     # ==========================
#     save_dir = os.path.join(base_result_path, ours_exp_path, 'tsne_visualization')
#     os.makedirs(save_dir, exist_ok=True)
#
#     task_str = "-".join(map(str, TASKS_TO_VISUALIZE))
#     filename = f'tsne_tasks{task_str}_{"with" if SHOW_TEACHER else "no"}_teacher.png'
#     save_path = os.path.join(save_dir, filename)
#
#     plot_tsne_comparison_by_class(feat_t, feat_b, feat_o, labels, save_path, show_teacher=SHOW_TEACHER)
#
#
# if __name__ == '__main__':
#     main()



# # -*- coding: UTF-8 -*-
# """
# 特征对齐 t-SNE 可视化脚本 (UWave 修复版)
# 修复内容：增加自适应模型加载，解决 Head 维度不匹配报错
# """
#
# import os
# import sys
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from types import SimpleNamespace
# import matplotlib.cm as cm
#
# # 获取项目根目录并切换工作目录
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(project_root)
# sys.path.insert(0, project_root)
#
# from models.base import setup_model
# from utils.stream import IncrementalTaskStream
# from utils.data import Dataloader_from_numpy
# from utils.utils import seed_fixer
#
#
# # ==========================================
# # 核心功能函数
# # ==========================================
#
# def load_checkpoint_robust(model, ckpt_path, device):
#     """
#     自适应加载权重：如果存档的分类头维度和代码不一致，强制修改代码中的模型以匹配存档。
#     """
#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
#
#     # 1. 加载 state_dict
#     state_dict = torch.load(ckpt_path, map_location=device)
#
#     # 2. 检查分类头 (Head) 维度
#     # 假设分类头名字是 'head.fc.weight' (根据报错信息推断)
#     if 'head.fc.weight' in state_dict:
#         saved_weight = state_dict['head.fc.weight']
#         saved_out_features = saved_weight.shape[0]  # 存档中的类别数 (比如 4)
#         saved_in_features = saved_weight.shape[1]  # 输入维度 (比如 128)
#
#         current_out = model.head.fc.out_features
#
#         # 3. 如果维度不匹配，动态调整模型结构
#         if current_out != saved_out_features:
#             print(
#                 f"Warning: Model head mismatch! Resizing code model from {current_out} to {saved_out_features} to match checkpoint.")
#             model.head.fc = nn.Linear(saved_in_features, saved_out_features)
#
#     # 4. 安全加载
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model
#
#
# def extract_all_features(model, dataloader, device):
#     """
#     提取整个数据集的特征和标签
#     """
#     model.eval()
#     features_list = []
#     labels_list = []
#
#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.to(device)
#             f_map = model.feature_map(x)
#             f_flat = f_map.view(f_map.size(0), -1)
#             features_list.append(f_flat.cpu().numpy())
#             labels_list.append(y.numpy())
#
#     return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)
#
#
# def plot_tsne_comparison_by_class(feat_t, feat_b, feat_o, labels, save_path):
#     print("Computing t-SNE... Please wait.")
#
#     # 1. 组合特征
#     combined_feats = np.vstack([feat_t, feat_b, feat_o])
#
#     # 2. t-SNE
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
#     embedded = tsne.fit_transform(combined_feats)
#
#     # 3. 拆分
#     n = feat_t.shape[0]
#     tsne_t = embedded[:n]
#     tsne_b = embedded[n:2 * n]
#     tsne_o = embedded[2 * n:]
#
#     # 4. 绘图
#     fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
#
#     unique_classes = np.unique(labels)
#     n_classes = len(unique_classes)
#
#     # 颜色映射兼容写法
#     import matplotlib
#     if hasattr(matplotlib, 'colormaps'):
#         cmap = matplotlib.colormaps['tab10'] if n_classes <= 10 else matplotlib.colormaps['tab20']
#     else:
#         cmap = cm.get_cmap('tab10') if n_classes <= 10 else cm.get_cmap('tab20')
#
#     # --- Subplot 1: Baseline vs Teacher ---
#     ax1 = axes[0]
#     # Teacher (背景)
#     for i, cls in enumerate(unique_classes):
#         idx = np.where(labels == cls)[0]
#         color = cmap(i)
#         ax1.scatter(tsne_t[idx, 0], tsne_t[idx, 1], color=color, alpha=0.2, s=100, label='_nolegend_')
#     # Baseline (前景)
#     for i, cls in enumerate(unique_classes):
#         idx = np.where(labels == cls)[0]
#         color = cmap(i)
#         ax1.scatter(tsne_b[idx, 0], tsne_b[idx, 1], color=color, alpha=0.9, s=20, edgecolors='white', linewidth=0.5,
#                     label=f'Class {cls}')
#
#     ax1.set_title("(a) Baseline (DT2W) vs Teacher", fontsize=16, fontweight='bold')
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#
#     # --- Subplot 2: Ours vs Teacher ---
#     ax2 = axes[1]
#     # Teacher (背景)
#     for i, cls in enumerate(unique_classes):
#         idx = np.where(labels == cls)[0]
#         color = cmap(i)
#         ax2.scatter(tsne_t[idx, 0], tsne_t[idx, 1], color=color, alpha=0.2, s=100, label='_nolegend_')
#     # Ours (前景)
#     for i, cls in enumerate(unique_classes):
#         idx = np.where(labels == cls)[0]
#         color = cmap(i)
#         ax2.scatter(tsne_o[idx, 0], tsne_o[idx, 1], color=color, alpha=0.9, s=20, edgecolors='white', linewidth=0.5,
#                     label=f'Class {cls}')
#
#     ax2.set_title("(b) HiDe (Ours) vs Teacher", fontsize=16, fontweight='bold')
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#
#     # 图例
#     from matplotlib.lines import Line2D
#     legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Class {int(cls)}',
#                               markerfacecolor=cmap(i), markersize=10) for i, cls in enumerate(unique_classes)]
#
#     legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Teacher (Background)',
#                                   markerfacecolor='gray', markersize=15, alpha=0.3))
#     legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Student (Foreground)',
#                                   markerfacecolor='gray', markersize=8, alpha=0.9))
#
#     fig.legend(handles=legend_elements, loc='lower center', ncol=len(unique_classes) + 2, bbox_to_anchor=(0.5, -0.05),
#                fontsize=12)
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight')
#     print(f"Visualization saved to: {save_path}")
#
#
# # ==========================================
# # 配置部分
# # ==========================================
# def create_model_args(data='uwave', encoder='CNN', head='Linear', norm='BN',
#                       feature_dim=128, n_layers=4, dropout=0, device='cuda',
#                       agent='Offline', input_norm='IN'):
#     args = SimpleNamespace(
#         data=data, encoder=encoder, head=head, norm=norm,
#         feature_dim=feature_dim, n_layers=n_layers, dropout=dropout,
#         device=device, agent=agent, input_norm=input_norm, stream_split='exp'
#     )
#     return args
#
#
# def main():
#     # 1. 实验路径配置 (UWave 版本)
#     base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_uwave')
#
#     ours_exp_path = 'Dkfd_BN_Dec-29-22-09-35-judge'
#     baseline_exp_path = 'DT2W_BN_Dec-29-21-31-38'
#
#     teacher_ckpt = os.path.join(base_result_path, ours_exp_path, 'teacher_after_task0_r0.pt')
#     ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'final_model_r0.pt')
#     baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'final_model_r0.pt')
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     seed_fixer(0)
#
#     # 2. 数据准备：加载 Task 0 (旧类)
#     cls_order = [2, 1, 6, 0, 3, 4, 5, 7]
#     task_stream = IncrementalTaskStream(data='uwave', scenario='class', cls_order=cls_order, split='exp')
#     task_stream.setup(load_subject=False)
#
#     x_test, y_test = task_stream.tasks[0][2]
#     print(f"Loading Task 0 Test Data (UWave): {x_test.shape}")
#     test_loader = Dataloader_from_numpy(x_test, y_test, batch_size=64, shuffle=False)
#
#     # 3. 加载模型 & 提取特征
#     # 注意：不再手动指定 head 维度，全部交给 load_checkpoint_robust 处理
#
#     # --- Teacher Model ---
#     print(">>> Loading Teacher...")
#     args_t = create_model_args(data='uwave', device=device, agent='Dkfd', head='Linear')
#     teacher_model = setup_model(args_t)
#     teacher_model = load_checkpoint_robust(teacher_model, teacher_ckpt, device)
#     feat_t, labels = extract_all_features(teacher_model, test_loader, device)
#
#     # --- Baseline Model ---
#     print(">>> Loading Baseline...")
#     args_b = create_model_args(data='uwave', device=device, agent='DT2W', head='Linear')
#     baseline_model = setup_model(args_b)
#     baseline_model = load_checkpoint_robust(baseline_model, baseline_ckpt, device)
#     feat_b, _ = extract_all_features(baseline_model, test_loader, device)
#
#     # --- Ours Model ---
#     print(">>> Loading Ours...")
#     args_o = create_model_args(data='uwave', device=device, agent='Dkfd', head='Linear')
#     ours_model = setup_model(args_o)
#     ours_model = load_checkpoint_robust(ours_model, ours_ckpt, device)
#     feat_o, _ = extract_all_features(ours_model, test_loader, device)
#
#     # 4. 绘制并保存
#     save_dir = os.path.join(base_result_path, ours_exp_path, 'tsne_visualization')
#     os.makedirs(save_dir, exist_ok=True)
#
#     save_path = os.path.join(save_dir, 'tsne_uwave_class_alignment_v2.png')
#
#     plot_tsne_comparison_by_class(feat_t, feat_b, feat_o, labels, save_path)
#     plt.show()
#
# if __name__ == '__main__':
#     main()



# """
# 特征对齐 t-SNE 可视化脚本 (方案一：流形分布对比)
# 功能：
# 1. 提取 Task 1 (旧类) 测试集的所有样本特征。
# 2. 对比 Frozen Teacher (只学过 Task1) 与 Student (学完 Task2 后) 的特征分布。
# 3. 使用 t-SNE 将 Teacher, Baseline, Ours 投影到同一 2D 空间。
# 4. 绘制对比图：证明 HiDe 能让 Student 特征与 Teacher 特征完美重叠 (Overlap)，而 Baseline 有间距 (Gap)。
# """
#
# import os
# import sys
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from types import SimpleNamespace
#
# # 获取项目根目录并切换工作目录
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(project_root)
# sys.path.insert(0, project_root)
#
# from models.base import setup_model
# from utils.stream import IncrementalTaskStream
# from utils.data import Dataloader_from_numpy
# from utils.utils import seed_fixer
#
# # ==========================================
# # 核心功能函数
# # ==========================================
#
# def extract_all_features(model, dataloader, device):
#     """
#     提取整个数据集的特征
#     Returns: numpy array [N, Feature_Dim]
#     """
#     model.eval()
#     features_list = []
#
#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.to(device)
#             # 获取特征图 [B, C, L]
#             f_map = model.feature_map(x)
#
#             # 展平处理: [B, C, L] -> [B, C*L]
#             # 注意：对于形状对齐，保留时序维度的展平通常比 Global Average Pooling 更能体现细节差异
#             # 如果维度过大导致 t-SNE 极慢，可以考虑先做 PCA 或 Pooling，
#             # 但为了体现 "Shape" 对齐，直接展平通常效果最好。
#             f_flat = f_map.view(f_map.size(0), -1)
#
#             features_list.append(f_flat.cpu().numpy())
#
#     return np.concatenate(features_list, axis=0)
#
# def plot_tsne_comparison(feat_t, feat_b, feat_o, save_path):
#     """
#     绘制 t-SNE 对比图
#     feat_t: Teacher Features
#     feat_b: Baseline Features
#     feat_o: Ours Features
#     """
#     print("Computing t-SNE (this may take a moment)...")
#
#     # 1. 组合所有特征进行统一 t-SNE，确保在同一个坐标系下比较
#     # 形状: [N_total, Dim]
#     combined_feats = np.vstack([feat_t, feat_b, feat_o])
#
#     # 2. 运行 t-SNE
#     # perplexity: 邻居数量，通常 30-50
#     # init='pca': 通常能得到更稳定的结果
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
#     embedded = tsne.fit_transform(combined_feats)
#
#     # 3. 拆分回各自的数据
#     n_samples = feat_t.shape[0]
#     tsne_t = embedded[:n_samples]
#     tsne_b = embedded[n_samples:2*n_samples]
#     tsne_o = embedded[2*n_samples:]
#
#     # 4. 绘图配置
#     fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
#
#     # 通用样式设置
#     # Teacher: 空心圆，灰色，代表"背景/基准"
#     style_teacher = {'c': 'none', 'edgecolors': 'gray', 'alpha': 0.6, 's': 40, 'linewidths': 0.8, 'label': 'Teacher (Task 1 Model)'}
#
#     # Student: 实心点，颜色鲜艳
#     style_student_b = {'c': '#D62728', 'alpha': 0.7, 's': 15, 'label': 'Student (Baseline)'} # Red
#     style_student_o = {'c': '#2CA02C', 'alpha': 0.7, 's': 15, 'label': 'Student (Ours)'}     # Green
#
#     # --- Subplot 1: Baseline vs Teacher ---
#     ax1 = axes[0]
#     # 先画 Teacher (垫底)
#     ax1.scatter(tsne_t[:, 0], tsne_t[:, 1], **style_teacher)
#     # 再画 Student
#     ax1.scatter(tsne_b[:, 0], tsne_b[:, 1], **style_student_b)
#
#     ax1.set_title("(a) Baseline (DT2W) vs Teacher", fontsize=16, fontweight='bold', y=1.02)
#     ax1.legend(loc='upper right', fontsize=10)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     # 添加边框说明
#     for spine in ax1.spines.values():
#         spine.set_edgecolor('black')
#         spine.set_linewidth(1.5)
#
#     # --- Subplot 2: Ours vs Teacher ---
#     ax2 = axes[1]
#     # 先画 Teacher
#     ax2.scatter(tsne_t[:, 0], tsne_t[:, 1], **style_teacher)
#     # 再画 Student
#     ax2.scatter(tsne_o[:, 0], tsne_o[:, 1], **style_student_o)
#
#     ax2.set_title("(b) HiDe (Ours) vs Teacher", fontsize=16, fontweight='bold', y=1.02)
#     ax2.legend(loc='upper right', fontsize=10)
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     for spine in ax2.spines.values():
#         spine.set_edgecolor('black')
#         spine.set_linewidth(1.5)
#
#     # 整体布局
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#         print(f"-> Saved t-SNE visualization to: {save_path}")
#
#     plt.show()
#
# # ==========================================
# # 参数配置与主程序
# # ==========================================
# def create_model_args(data='uwave', encoder='CNN', head='Linear', norm='BN',
#                       feature_dim=128, n_layers=4, dropout=0, device='cuda',
#                       agent='Offline', input_norm='IN'):
#     args = SimpleNamespace(
#         data=data, encoder=encoder, head=head, norm=norm,
#         feature_dim=feature_dim, n_layers=n_layers, dropout=dropout,
#         device=device, agent=agent, input_norm=input_norm, stream_split='exp'
#     )
#     return args
#
# def main():
#     # 1. 实验路径配置 (请根据实际情况修改)
#     base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_uwave')
#
#     # 你的实验文件夹
#     ours_exp_path = 'Dkfd_BN_Dec-29-22-09-35-judge'
#     baseline_exp_path = 'DT2W_BN_Dec-29-21-31-38'
#
#     # Teacher 路径 (通常是 Task 0 结束后的模型)
#     teacher_ckpt = os.path.join(base_result_path, 'Dkfd_BN_Dec-29-22-09-35-judge', 'teacher_after_task0_r0.pt')
#
#     # Student 路径 (Task 1 结束后的模型，即学完第二个任务)
#     # 注意：这里需要加载的是训练完 Task 1 后的模型。
#     # 假设 final_model_r0.pt 是整个流程结束的模型，如果只有 2 个任务，那就是它。
#     # 如果有多个任务，最好加载 task1 的 checkpoint。这里暂时假设是 final model。
#     ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'final_model_r0.pt')
#     baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'final_model_r0.pt')
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     seed_fixer(0)
#
#     # 检查文件是否存在
#     for path, name in [(teacher_ckpt, 'Teacher'), (ours_ckpt, 'Ours'), (baseline_ckpt, 'Baseline')]:
#         if not os.path.exists(path):
#             print(f"Error: {name} ckpt not found: {path}")
#             # return # 暂时注释，方便调试，实际运行时请取消注释
#
#     # 2. 数据准备：加载 Task 0 (旧类) 的测试集
#     # 我们要看的是：旧类样本在不同模型中的分布
#     cls_order = [2, 1, 6, 0, 3, 4, 5, 7] # 请确保这个顺序与你训练时的一致！
#     task_stream = IncrementalTaskStream(data='uwave', scenario='class', cls_order=cls_order, split='exp')
#     task_stream.setup(load_subject=False)
#
#     # 获取 Task 0 (Old Classes) 的测试数据
#     x_test, y_test = task_stream.tasks[0][2]
#     print(f"Loading Task 0 Test Data: {x_test.shape}")
#     test_loader = Dataloader_from_numpy(x_test, y_test, batch_size=64, shuffle=False)
#
#     # 3. 加载模型 & 提取特征
#     # --- Teacher Model ---
#     args_t = create_model_args(device=device, agent='Dkfd', head='Linear')
#     teacher_model = setup_model(args_t)
#     teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
#     teacher_model.to(device)
#     print("Extracting Teacher Features...")
#     feat_t = extract_all_features(teacher_model, test_loader, device)
#
#     # --- Baseline Model (DT2W) ---
#     args_b = create_model_args(device=device, agent='DT2W', head='Linear')
#     baseline_model = setup_model(args_b)
#     # 更新 Head 以匹配权重加载 (通常 Final Model 包含所有任务的 Head)
#     baseline_model.update_head(n_new=2, task_now=1)
#     baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
#     baseline_model.to(device)
#     print("Extracting Baseline Features...")
#     feat_b = extract_all_features(baseline_model, test_loader, device)
#
#     # --- Ours Model (HiDe) ---
#     args_o = create_model_args(device=device, agent='Dkfd', head='Linear')
#     ours_model = setup_model(args_o)
#     ours_model.update_head(n_new=2, task_now=1)
#     ours_model.load_state_dict(torch.load(ours_ckpt, map_location=device))
#     ours_model.to(device)
#     print("Extracting Ours Features...")
#     feat_o = extract_all_features(ours_model, test_loader, device)
#
#     print(f"Features Shape -> Teacher: {feat_t.shape}, Baseline: {feat_b.shape}, Ours: {feat_o.shape}")
#
#     # 4. 绘制并保存
#     save_dir = os.path.join(base_result_path, ours_exp_path, 'tsne_visualization')
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, 'tsne_alignment_comparison.png') # 也可以存为 .pdf
#
#     plot_tsne_comparison(feat_t, feat_b, feat_o, save_path)
#
# if __name__ == '__main__':
#     main()


# # -*- coding: UTF-8 -*-
# import os
# import sys
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from types import SimpleNamespace
#
# # 环境配置
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(project_root)
# sys.path.insert(0, project_root)
#
# from models.base import setup_model
# from utils.stream import IncrementalTaskStream
# from utils.data import Dataloader_from_numpy
# from utils.utils import seed_fixer
#
#
# def extract_all_features(model, dataloader, device, use_gap=True):
#     """
#     提取特征。
#     use_gap=True: 使用全局平均池化，计算快，适合展示语义聚类。
#     use_gap=False: 使用全特征展开，计算慢，但能体现更细微的时序形状差异。
#     """
#     model.eval()
#     features_list = []
#     with torch.no_grad():
#         for x, _ in dataloader:
#             x = x.to(device)
#             f_map = model.feature_map(x)  # [B, 128, 128]
#
#             if use_gap:
#                 # 降维到 [B, 128]
#                 feat = torch.mean(f_map, dim=-1)
#             else:
#                 # 展开到 [B, 16384]，解决 view 报错
#                 feat = f_map.reshape(f_map.size(0), -1)
#
#             features_list.append(feat.cpu().numpy())
#     return np.concatenate(features_list, axis=0)
#
#
# def plot_tsne_comparison(feat_t, feat_b, feat_o, save_path):
#     print("Computing t-SNE... Please wait.")
#     combined_feats = np.vstack([feat_t, feat_b, feat_o])
#
#     # 执行 t-SNE
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
#     embedded = tsne.fit_transform(combined_feats)
#
#     n = feat_t.shape[0]
#     tsne_t, tsne_b, tsne_o = embedded[:n], embedded[n:2 * n], embedded[2 * n:]
#
#     fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
#
#     # 绘图风格
#     t_style = {'c': 'none', 'edgecolors': 'gray', 'alpha': 0.3, 's': 40, 'label': 'Teacher (Base)'}
#     b_style = {'c': '#D62728', 'alpha': 0.6, 's': 15, 'label': 'Baseline (DT2W)'}
#     o_style = {'c': '#2CA02C', 'alpha': 0.6, 's': 15, 'label': 'Ours (HiDe)'}
#
#     # 左图: Baseline
#     axes[0].scatter(tsne_t[:, 0], tsne_t[:, 1], **t_style)
#     axes[0].scatter(tsne_b[:, 0], tsne_b[:, 1], **b_style)
#     axes[0].set_title("Baseline vs Teacher (Task 0 Classes)", fontsize=14, fontweight='bold')
#
#     # 右图: Ours
#     axes[1].scatter(tsne_t[:, 0], tsne_t[:, 1], **t_style)
#     axes[1].scatter(tsne_o[:, 0], tsne_o[:, 1], **o_style)
#     axes[1].set_title("Ours vs Teacher (Task 0 Classes)", fontsize=14, fontweight='bold')
#
#     for ax in axes:
#         ax.legend()
#         ax.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(save_path)
#     print(f"Visualization saved to: {save_path}")
#
#
# def load_model_dynamic(ckpt_path, agent_type, device):
#     args = SimpleNamespace(data='har', encoder='CNN', head='Linear', norm='LN',
#                            feature_dim=128, n_layers=4, dropout=0, device=device,
#                            agent=agent_type, input_norm='LN', stream_split='exp')
#     model = setup_model(args)
#     sd = torch.load(ckpt_path, map_location=device)
#     if 'head.fc.weight' in sd:
#         model.head.fc = torch.nn.Linear(model.head.fc.in_features, sd['head.fc.weight'].shape[0])
#     model.to(device).load_state_dict(sd)
#     return model.eval()
#
#
# def main():
#     base_path = 'result/tune_and_exp/CNN_har'
#     ours_exp = 'Dkfd_LN_Jan-10-17-28-50-judge'
#     base_exp = 'DT2W_LN_Jan-10-20-57-02'
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     seed_fixer(0)
#
#     # 1. 路径
#     t_path = os.path.join(base_path, ours_exp, 'teacher_after_task0_r0.pt')
#     o_path = os.path.join(base_path, ours_exp, 'final_model_r0.pt')
#     b_path = os.path.join(base_path, base_exp, 'final_model_r0.pt')
#
#     # 2. 数据 (只加载 Task 0)
#     cls_order = [2, 1, 5, 0, 4, 3]
#     stream = IncrementalTaskStream(data='har', scenario='class', cls_order=cls_order, split='exp')
#     stream.setup(load_subject=False)
#     x_t0, y_t0 = stream.tasks[0][2]  # Task 0 的测试集
#     loader = Dataloader_from_numpy(x_t0, y_t0, batch_size=64, shuffle=False)
#
#     # 3. 提特征
#     print("Loading models and extracting features...")
#     # 这里 use_gap=True 能让你快速看到结果
#     f_t = extract_all_features(load_model_dynamic(t_path, 'Dkfd', device), loader, device, use_gap=True)
#     f_b = extract_all_features(load_model_dynamic(b_path, 'DT2W', device), loader, device, use_gap=True)
#     f_o = extract_all_features(load_model_dynamic(o_path, 'Dkfd', device), loader, device, use_gap=True)
#
#     # 4. 绘图
#     plot_tsne_comparison(f_t, f_b, f_o, os.path.join(base_path, ours_exp, "task0_tsne.png"))
#
#
# if __name__ == '__main__':
#     main()


# # -*- coding: UTF-8 -*-
# import os
# import sys
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from types import SimpleNamespace
# import matplotlib.cm as cm  # 引入颜色映射
#
# # 环境配置
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(project_root)
# sys.path.insert(0, project_root)
#
# from models.base import setup_model
# from utils.stream import IncrementalTaskStream
# from utils.data import Dataloader_from_numpy
# from utils.utils import seed_fixer
#
#
# def extract_all_features(model, dataloader, device, use_gap=True):
#     """
#     提取特征并返回对应的标签
#     """
#     model.eval()
#     features_list = []
#     labels_list = []  # 新增：存储标签
#
#     with torch.no_grad():
#         for x, y in dataloader:  # 这里解包 x, y
#             x = x.to(device)
#             f_map = model.feature_map(x)
#
#             if use_gap:
#                 feat = torch.mean(f_map, dim=-1)
#             else:
#                 feat = f_map.reshape(f_map.size(0), -1)
#
#             features_list.append(feat.cpu().numpy())
#             labels_list.append(y.numpy())  # 收集标签
#
#     return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)
#
# def plot_tsne_comparison(feat_t, feat_b, feat_o, labels, save_path):
#     print("Computing t-SNE... Please wait.")
#
#     # 确保标签是整数
#     labels = labels.astype(int)
#     unique_classes = np.unique(labels)
#     n_classes = len(unique_classes)
#
#     # 组合所有特征进行 t-SNE 映射
#     combined_feats = np.vstack([feat_t, feat_b, feat_o])
#
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
#     embedded = tsne.fit_transform(combined_feats)
#
#     n = feat_t.shape[0]
#     tsne_t = embedded[:n]  # Teacher
#     tsne_b = embedded[n:2 * n]  # Baseline
#     tsne_o = embedded[2 * n:]  # Ours
#
#     # 设置绘图
#     fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
#
#     # 【修改点开始】获取颜色映射 (兼容 Matplotlib 3.7+)
#     import matplotlib  # 确保在函数内或文件头导入了 matplotlib
#     if hasattr(matplotlib, 'colormaps'):
#         cmap = matplotlib.colormaps['tab10'] if n_classes <= 10 else matplotlib.colormaps['tab20']
#     else:
#         cmap = cm.get_cmap('tab10') if n_classes <= 10 else cm.get_cmap('tab20')
#     # 【修改点结束】
#
#     # Teacher 的样式 (作为灰色背景参考)
#     t_style = {'c': 'lightgray', 'edgecolors': 'none', 'alpha': 0.3, 's': 60, 'label': 'Teacher (Ref)', 'zorder': 0}
#
#     # 子图 1: Baseline
#     axes[0].scatter(tsne_t[:, 0], tsne_t[:, 1], **t_style)
#     for i, cls in enumerate(unique_classes):
#         idx = np.where(labels == cls)[0]
#         color = cmap(i / (n_classes - 1) if n_classes > 1 else 0)
#         axes[0].scatter(tsne_b[idx, 0], tsne_b[idx, 1], c=[color], alpha=0.7, s=20, label=f'Class {cls}', zorder=1)
#     axes[0].set_title("Baseline (Colored) vs Teacher (Gray)", fontsize=16, fontweight='bold')
#
#     # 子图 2: Ours
#     axes[1].scatter(tsne_t[:, 0], tsne_t[:, 1], **t_style)
#     for i, cls in enumerate(unique_classes):
#         idx = np.where(labels == cls)[0]
#         color = cmap(i / (n_classes - 1) if n_classes > 1 else 0)
#         axes[1].scatter(tsne_o[idx, 0], tsne_o[idx, 1], c=[color], alpha=0.7, s=20, label=f'Class {cls}', zorder=1)
#     axes[1].set_title("Ours (Colored) vs Teacher (Gray)", fontsize=16, fontweight='bold')
#
#     # 通用设置
#     for ax in axes:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         handles, labels_txt = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels_txt, handles))
#         ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10, markerscale=1.5)
#
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(save_path, bbox_inches='tight')
#     print(f"Visualization saved to: {save_path}")
#
# def load_model_dynamic(ckpt_path, agent_type, device):
#     args = SimpleNamespace(data='har', encoder='CNN', head='Linear', norm='LN',
#                            feature_dim=128, n_layers=4, dropout=0, device=device,
#                            agent=agent_type, input_norm='LN', stream_split='exp')
#     model = setup_model(args)
#     sd = torch.load(ckpt_path, map_location=device)
#
#     # 处理 Head 维度不匹配的兼容性代码
#     if 'head.fc.weight' in sd:
#         current_dim = model.head.fc.in_features
#         saved_dim = sd['head.fc.weight'].shape[0]
#         if model.head.fc.out_features != saved_dim:
#             model.head.fc = torch.nn.Linear(current_dim, saved_dim)
#
#     model.to(device).load_state_dict(sd)
#     return model.eval()
#
#
# def main():
#     base_path = 'result/tune_and_exp/CNN_har'
#     ours_exp = 'Dkfd_LN_Jan-10-17-28-50-judge'
#     base_exp = 'DT2W_LN_Jan-10-20-57-02'
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     seed_fixer(0)
#
#     # 1. 路径
#     t_path = os.path.join(base_path, ours_exp, 'teacher_after_task0_r0.pt')
#     o_path = os.path.join(base_path, ours_exp, 'final_model_r0.pt')
#     b_path = os.path.join(base_path, base_exp, 'final_model_r0.pt')
#
#     # 2. 数据 (只加载 Task 0)
#     cls_order = [2, 1, 5, 0, 4, 3]
#     stream = IncrementalTaskStream(data='har', scenario='class', cls_order=cls_order, split='exp')
#     stream.setup(load_subject=False)
#     x_t0, y_t0 = stream.tasks[0][2]  # Task 0 的测试集
#
#     # 注意：shuffle=False 保证我们提取出的 labels 和 features 顺序是一一对应的
#     loader = Dataloader_from_numpy(x_t0, y_t0, batch_size=64, shuffle=False)
#
#     # 3. 提特征 (注意接收两个返回值)
#     print("Loading models and extracting features...")
#
#     # 这里的 labels 其实只需要提取一次，因为所有模型用的数据是一样的
#     f_t, labels = extract_all_features(load_model_dynamic(t_path, 'Dkfd', device), loader, device, use_gap=True)
#     f_b, _ = extract_all_features(load_model_dynamic(b_path, 'DT2W', device), loader, device, use_gap=True)
#     f_o, _ = extract_all_features(load_model_dynamic(o_path, 'Dkfd', device), loader, device, use_gap=True)
#
#     # 4. 绘图 (传入 labels)
#     save_file = os.path.join(base_path, ours_exp, "task0_tsne_by_class.png")
#     plot_tsne_comparison(f_t, f_b, f_o, labels, save_file)
#
#
# if __name__ == '__main__':
#     main()