# # -*- coding: UTF-8 -*-
# """
# 频域蒸馏效果可视化脚本 (2行5列版 - 终极版)
# 对比 Teacher、Baseline(DT2W)、Ours(Dkfd) 在【原始、DC、含DC低频、不含DC低频、高频】上的对齐效果
# """
# import random
#
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import sys
# import os
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
# from utils.utils import TriBandFDFilter, seed_fixer
#
#
# def get_real_feature_data(teacher_model, baseline_model, ours_model, dataloader, device, fd_filter, top_k=10):
#     """
#     提取特征，筛选策略：
#     1. 确保高频有波动 (Amp_High > Threshold)
#     2. 在此基础上，寻找低频能量占比最大的样本 (Maximize Low/High Ratio)
#     """
#     teacher_model.eval()
#     baseline_model.eval()
#     ours_model.eval()
#
#     all_candidates = []
#
#     print(f"Scanning samples to find 'Low >> High' but 'High is Active' cases...")
#
#     with torch.no_grad():
#         for batch_idx, (x, y) in enumerate(dataloader):
#             x = x.to(device)
#             # 获取 Teacher 特征
#             t_fmap = teacher_model.feature_map(x)
#
#             # 解耦：获取 DC, Low, High
#             # 注意：这里需要在 batch 循环里做一次完整解耦，才能算出幅度
#             t_dc, t_low_only, t_high = fd_filter(t_fmap)
#
#             # 计算幅度 (Amplitude) - 使用平均绝对值
#             # [B, C]
#             amp_low = torch.mean(torch.abs(t_low_only), dim=-1)
#             amp_high = torch.mean(torch.abs(t_high), dim=-1)
#
#             B, C = t_fmap.shape[0], t_fmap.shape[1]
#             for b in range(B):
#                 for c in range(C):
#                     a_low = amp_low[b, c].item()
#                     a_high = amp_high[b, c].item()
#
#                     # 记录所有候选者
#                     all_candidates.append({
#                         'amp_low': a_low,
#                         'amp_high': a_high,
#                         'ratio': a_low / (a_high + 1e-8),  # 避免除以0
#                         'x_sample': x[b].unsqueeze(0).clone(),
#                         'c_idx': c
#                     })
#
#     # ================= 核心筛选逻辑修改 =================
#
#     # 1. 计算高频能量的阈值 (例如中位数)，排除那些高频完全是死线的样本
#     high_amps = [c['amp_high'] for c in all_candidates]
#     # 设阈值为所有样本高频能量的 Top 60% (即排除掉最弱的 40%)
#     # 这样保证选出来的样本，高频是有“动静”的
#     threshold_high = np.percentile(high_amps, 40)
#
#     print(f"High-Freq Threshold (Top 60% activity): {threshold_high:.4f}")
#
#     # 2. 过滤掉高频太弱的样本
#     active_candidates = [c for c in all_candidates if c['amp_high'] > threshold_high]
#     print(f"Candidates after filtering silent high-freq: {len(active_candidates)} / {len(all_candidates)}")
#
#     # 3. 在剩下的样本中，按 Low/High 比率排序 (找 Low 最强势的)
#     active_candidates.sort(key=lambda x: x['ratio'], reverse=True)
#
#     # 4. 取 Top K
#     top_candidates = active_candidates[:top_k]
#     # ==================================================
#
#     print(f"Selected top {top_k} samples fitting the criteria.")
#
#     results = []
#     for i, cand in enumerate(top_candidates):
#         x_sample = cand['x_sample'].to(device)
#         c_idx = cand['c_idx']
#
#         # Forward
#         t_fmap = teacher_model.feature_map(x_sample)
#         b_fmap = baseline_model.feature_map(x_sample)
#         o_fmap = ours_model.feature_map(x_sample)
#
#         # Decouple
#         t_dc, t_low_only, t_high = fd_filter(t_fmap)
#         b_dc, b_low_only, b_high = fd_filter(b_fmap)
#         o_dc, o_low_only, o_high = fd_filter(o_fmap)
#
#         # Merge DC for Trend
#         t_low_dc = t_dc + t_low_only
#         b_low_dc = b_dc + b_low_only
#         o_low_dc = o_dc + o_low_only
#
#         def to_np(tensor): return tensor[0, c_idx, :].detach().cpu().numpy()
#
#         # 打印信息验证
#         print(f"Rank {i + 1}: Ratio={cand['ratio']:.1f} | Low={cand['amp_low']:.4f} >> High={cand['amp_high']:.4f}")
#
#         results.append({
#             'rank': i + 1,
#             'std': cand['amp_high'],  # 这里存 amp_high 方便参考
#             'channel': c_idx,
#             # Raw
#             't_raw': to_np(t_fmap), 'b_raw': to_np(b_fmap), 'o_raw': to_np(o_fmap),
#             # DC Component
#             't_dc': to_np(t_dc), 'b_dc': to_np(b_dc), 'o_dc': to_np(o_dc),
#             # Low + DC (Trend)
#             't_low_dc': to_np(t_low_dc), 'b_low_dc': to_np(b_low_dc), 'o_low_dc': to_np(o_low_dc),
#             # Low Only (No DC)
#             't_low_only': to_np(t_low_only), 'b_low_only': to_np(b_low_only), 'o_low_only': to_np(o_low_only),
#             # High (Detail)
#             't_high': to_np(t_high), 'b_high': to_np(b_high), 'o_high': to_np(o_high)
#         })
#
#     return results
# def plot_visualization_2x5(data_dict, save_path=None, title_suffix=""):
#     """
#     绘制 2x5 的对比图：
#     Columns: [Raw, DC, Low+DC, Low-Only, High]
#     """
#     t_raw = data_dict['t_raw']
#     t_axis = np.arange(len(t_raw))
#
#     # 配色
#     c_teacher = 'black'
#     c_baseline = '#D62728'  # Red
#     c_ours = '#2CA02C'  # Green
#
#     # 初始化 2x5 画布 (宽度增加)
#     fig, axes = plt.subplots(2, 5, figsize=(30, 8), dpi=150)
#     plt.rcParams['font.family'] = 'serif'
#
#     # --- 统一 Y 轴范围 ---
#     def get_ylim(keys):
#         vals = np.concatenate([data_dict[k] for k in keys])
#         ymin, ymax = np.min(vals), np.max(vals)
#         # 如果是 DC 这种平直线，范围可能很小，防止除以零或太扁
#         if ymax - ymin < 1e-6:
#             margin = 0.5
#         else:
#             margin = (ymax - ymin) * 0.1
#         return (ymin - margin, ymax + margin)
#
#     ylim_raw = get_ylim(['t_raw', 'b_raw', 'o_raw'])
#     ylim_dc = get_ylim(['t_dc', 'b_dc', 'o_dc'])
#     ylim_low_dc = get_ylim(['t_low_dc', 'b_low_dc', 'o_low_dc'])
#     ylim_low_only = get_ylim(['t_low_only', 'b_low_only', 'o_low_only'])
#     ylim_high = get_ylim(['t_high', 'b_high', 'o_high'])
#
#     # 定义绘图辅助函数
#     def plot_col(row_idx, col_idx, t_data, s_data, s_color, s_label, title, ylim):
#         ax = axes[row_idx, col_idx]
#         ax.plot(t_axis, t_data, color=c_teacher, linewidth=3, alpha=0.4, label='Teacher')
#         ax.plot(t_axis, s_data, color=s_color, linestyle='--', linewidth=2, label=s_label)
#         ax.set_title(title, fontsize=11, fontweight='bold')
#         ax.set_ylim(ylim)
#         ax.legend(loc='upper right', fontsize=9)
#         ax.grid(True, linestyle=':', alpha=0.5)
#         if row_idx == 1: ax.set_xlabel("Time Step")
#
#     # ==================== Row 1: Baseline ====================
#     # (1,1) Raw
#     plot_col(0, 0, t_raw, data_dict['b_raw'], c_baseline, 'Baseline', "(a1) Baseline: Raw", ylim_raw)
#     # (1,2) DC
#     plot_col(0, 1, data_dict['t_dc'], data_dict['b_dc'], c_baseline, 'Baseline', "(a2) Baseline: DC (Mean)", ylim_dc)
#     # (1,3) Low + DC
#     plot_col(0, 2, data_dict['t_low_dc'], data_dict['b_low_dc'], c_baseline, 'Baseline', "(a3) Baseline: Low+DC",
#              ylim_low_dc)
#     # (1,4) Low Only
#     plot_col(0, 3, data_dict['t_low_only'], data_dict['b_low_only'], c_baseline, 'Baseline',
#              "(a4) Baseline: Low (No DC)", ylim_low_only)
#     # (1,5) High
#     plot_col(0, 4, data_dict['t_high'], data_dict['b_high'], c_baseline, 'Baseline', "(a5) Baseline: High", ylim_high)
#
#     # ==================== Row 2: Ours ====================
#     # (2,1) Raw
#     plot_col(1, 0, t_raw, data_dict['o_raw'], c_ours, 'Ours', "(b1) Ours: Raw", ylim_raw)
#     # (2,2) DC
#     plot_col(1, 1, data_dict['t_dc'], data_dict['o_dc'], c_ours, 'Ours', "(b2) Ours: DC (Mean)", ylim_dc)
#     # (2,3) Low + DC
#     plot_col(1, 2, data_dict['t_low_dc'], data_dict['o_low_dc'], c_ours, 'Ours', "(b3) Ours: Low+DC", ylim_low_dc)
#     # (2,4) Low Only
#     plot_col(1, 3, data_dict['t_low_only'], data_dict['o_low_only'], c_ours, 'Ours', "(b4) Ours: Low (No DC)",
#              ylim_low_only)
#     # (2,5) High
#     plot_col(1, 4, data_dict['t_high'], data_dict['o_high'], c_ours, 'Ours', "(b5) Ours: High", ylim_high)
#
#     plt.suptitle(f"Comprehensive Frequency Decomposition{title_suffix}", fontsize=16, y=0.98)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         print(f"  -> Saved: {save_path}")
#     plt.close(fig)
#
#
# # ==========================================
# # 辅助函数
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
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_uwave')
#
#     # === 请在这里修改你的实验文件夹名 ===
#     ours_exp_path = 'Dkfd_BN_Jan-05-11-17-22-judge'
#     baseline_exp_path = 'DT2W_BN_Dec-29-21-31-38'
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     seed_fixer(0)
#
#     # 路径检查
#     teacher_ckpt = os.path.join(base_result_path, 'Dkfd_BN_Dec-29-22-09-35-judge', 'teacher_after_task0_r0.pt')
#     ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'final_model_r0.pt')
#     baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'final_model_r0.pt')
#
#     for path, name in [(teacher_ckpt, 'Teacher'), (ours_ckpt, 'Ours'), (baseline_ckpt, 'Baseline')]:
#         if not os.path.exists(path):
#             print(f"Error: {name} ckpt not found: {path}")
#             return
#
#     # 加载模型
#     args_t = create_model_args(device=device, agent='Dkfd', head='Linear')
#     teacher_model = setup_model(args_t)
#     teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
#     teacher_model.eval()
#
#     args_o = create_model_args(device=device, agent='Dkfd', head='Linear')
#     ours_model = setup_model(args_o)
#     ours_model.update_head(n_new=2, task_now=1)
#     ours_model.to(device)
#     ours_model.load_state_dict(torch.load(ours_ckpt, map_location=device))
#     ours_model.eval()
#
#     args_b = create_model_args(device=device, agent='DT2W', head='Linear')
#     baseline_model = setup_model(args_b)
#     baseline_model.update_head(n_new=2, task_now=1)
#     baseline_model.to(device)
#     baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
#     baseline_model.eval()
#
#     # 数据准备
#     cls_order = [2, 1, 6, 0]
#     task_stream = IncrementalTaskStream(data='uwave', scenario='class', cls_order=cls_order, split='exp')
#     task_stream.setup(load_subject=False)
#     x_test, y_test = task_stream.tasks[0][2]
#     test_loader = Dataloader_from_numpy(x_test, y_test, batch_size=32, shuffle=False)
#
#     # 提取与绘图
#     fd_filter = TriBandFDFilter().to(device)
#     results = get_real_feature_data(
#         teacher_model, baseline_model, ours_model, test_loader, device, fd_filter, top_k=10
#     )
#
#     save_dir = os.path.join(base_result_path, ours_exp_path, 'freq_visualization_2x8')
#     os.makedirs(save_dir, exist_ok=True)
#
#     for data in results:
#         print(f"Plotting Rank {data['rank']}...")
#         save_path = os.path.join(save_dir, f'full_spec_rank{data["rank"]:02d}.png')
#
#         # 使用 2x5 绘图函数
#         plot_visualization_2x5(
#             data,
#             save_path=save_path,
#             title_suffix=f" (Rank {data['rank']}, Ch {data['channel']})"
#         )
#
#     print(f"Done. Images saved to {save_dir}")
#
#
# if __name__ == '__main__':
#     main()


# # -*- coding: UTF-8 -*-
# """
# 频域蒸馏效果可视化脚本 (2行5列版 - 终极版)
# 对比 Teacher、Baseline(DT2W)、Ours(Dkfd) 在【原始、DC、含DC低频、不含DC低频、高频】上的对齐效果
# """
# import random
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import sys
# import os
# from types import SimpleNamespace
# from matplotlib.ticker import MaxNLocator
#
# # 获取项目根目录并切换工作目录
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(project_root)
# sys.path.insert(0, project_root)
#
# from models.base import setup_model
# from utils.stream import IncrementalTaskStream
# from utils.data import Dataloader_from_numpy
# from utils.utils import TriBandFDFilter, seed_fixer
#
#
# def get_real_feature_data(teacher_model, baseline_model, ours_model, dataloader, device, fd_filter, top_k=10):
#     """
#     提取特征，筛选策略：
#     1. 确保高频有波动 (Amp_High > Threshold)
#     2. 在此基础上，寻找低频能量占比最大的样本 (Maximize Low/High Ratio)
#     """
#     teacher_model.eval()
#     baseline_model.eval()
#     ours_model.eval()
#
#     all_candidates = []
#
#     print(f"Scanning samples to find 'Low >> High' but 'High is Active' cases...")
#
#     with torch.no_grad():
#         for batch_idx, (x, y) in enumerate(dataloader):
#             x = x.to(device)
#             # 获取 Teacher 特征
#             t_fmap = teacher_model.feature_map(x)
#
#             # 解耦：获取 DC, Low, High
#             t_dc, t_low_only, t_high = fd_filter(t_fmap)
#
#             # 计算幅度 (Amplitude) - 使用平均绝对值
#             amp_low = torch.mean(torch.abs(t_low_only), dim=-1)
#             amp_high = torch.mean(torch.abs(t_high), dim=-1)
#
#             B, C = t_fmap.shape[0], t_fmap.shape[1]
#             for b in range(B):
#                 for c in range(C):
#                     a_low = amp_low[b, c].item()
#                     a_high = amp_high[b, c].item()
#
#                     # 记录所有候选者
#                     all_candidates.append({
#                         'amp_low': a_low,
#                         'amp_high': a_high,
#                         'ratio': a_low / (a_high + 1e-8),  # 避免除以0
#                         'x_sample': x[b].unsqueeze(0).clone(),
#                         'c_idx': c
#                     })
#
#     # ================= 核心筛选逻辑修改 =================
#
#     # 1. 计算高频能量的阈值 (例如中位数)，排除那些高频完全是死线的样本
#     high_amps = [c['amp_high'] for c in all_candidates]
#     threshold_high = np.percentile(high_amps, 40)
#
#     print(f"High-Freq Threshold (Top 60% activity): {threshold_high:.4f}")
#
#     # 2. 过滤掉高频太弱的样本
#     active_candidates = [c for c in all_candidates if c['amp_high'] > threshold_high]
#     print(f"Candidates after filtering silent high-freq: {len(active_candidates)} / {len(all_candidates)}")
#
#     # 3. 在剩下的样本中，按 Low/High 比率排序 (找 Low 最强势的)
#     active_candidates.sort(key=lambda x: x['ratio'], reverse=True)
#
#     # 4. 取 Top K
#     top_candidates = active_candidates[:top_k]
#     # ==================================================
#
#     print(f"Selected top {top_k} samples fitting the criteria.")
#
#     results = []
#     for i, cand in enumerate(top_candidates):
#         x_sample = cand['x_sample'].to(device)
#         c_idx = cand['c_idx']
#
#         # Forward
#         t_fmap = teacher_model.feature_map(x_sample)
#         b_fmap = baseline_model.feature_map(x_sample)
#         o_fmap = ours_model.feature_map(x_sample)
#
#         # Decouple
#         t_dc, t_low_only, t_high = fd_filter(t_fmap)
#         b_dc, b_low_only, b_high = fd_filter(b_fmap)
#         o_dc, o_low_only, o_high = fd_filter(o_fmap)
#
#         # Merge DC for Trend
#         t_low_dc = t_dc + t_low_only
#         b_low_dc = b_dc + b_low_only
#         o_low_dc = o_dc + o_low_only
#
#         def to_np(tensor): return tensor[0, c_idx, :].detach().cpu().numpy()
#
#         # 打印信息验证
#         print(f"Rank {i + 1}: Ratio={cand['ratio']:.1f} | Low={cand['amp_low']:.4f} >> High={cand['amp_high']:.4f}")
#
#         results.append({
#             'rank': i + 1,
#             'std': cand['amp_high'],  # 这里存 amp_high 方便参考
#             'channel': c_idx,
#             # Raw
#             't_raw': to_np(t_fmap), 'b_raw': to_np(b_fmap), 'o_raw': to_np(o_fmap),
#             # DC Component
#             't_dc': to_np(t_dc), 'b_dc': to_np(b_dc), 'o_dc': to_np(o_dc),
#             # Low + DC (Trend)
#             't_low_dc': to_np(t_low_dc), 'b_low_dc': to_np(b_low_dc), 'o_low_dc': to_np(o_low_dc),
#             # Low Only (No DC)
#             't_low_only': to_np(t_low_only), 'b_low_only': to_np(b_low_only), 'o_low_only': to_np(o_low_only),
#             # High (Detail)
#             't_high': to_np(t_high), 'b_high': to_np(b_high), 'o_high': to_np(o_high)
#         })
#
#     return results
#
#
# def plot_visualization_2x5(data_dict, save_path=None, title_suffix=""):
#     """
#     绘制 2x5 的对比图：
#     Columns: [Raw, DC, Low+DC, Low-Only, High]
#     """
#     t_raw = data_dict['t_raw']
#     t_axis = np.arange(len(t_raw))
#
#     # 配色
#     c_teacher = 'black'
#     c_baseline = '#D62728'  # Red
#     c_ours = '#2CA02C'  # Green
#
#     # Baseline 的 Label 名称
#     label_baseline = r'DT$^2$W'
#
#     # 初始化 2x5 画布
#     fig, axes = plt.subplots(2, 5, figsize=(30, 8), dpi=150)
#     plt.rcParams['font.family'] = 'serif'
#
#     # --- 统一 Y 轴范围 ---
#     def get_ylim(keys):
#         vals = np.concatenate([data_dict[k] for k in keys])
#         ymin, ymax = np.min(vals), np.max(vals)
#         if ymax - ymin < 1e-6:
#             margin = 0.5
#         else:
#             margin = (ymax - ymin) * 0.1
#         return (ymin - margin, ymax + margin)
#
#     ylim_raw = get_ylim(['t_raw', 'b_raw', 'o_raw'])
#     ylim_dc = get_ylim(['t_dc', 'b_dc', 'o_dc'])
#     ylim_low_dc = get_ylim(['t_low_dc', 'b_low_dc', 'o_low_dc'])
#     ylim_low_only = get_ylim(['t_low_only', 'b_low_only', 'o_low_only'])
#     ylim_high = get_ylim(['t_high', 'b_high', 'o_high'])
#
#     # 定义绘图辅助函数
#     def plot_col(row_idx, col_idx, t_data, s_data, s_color, s_label, title, ylim):
#         ax = axes[row_idx, col_idx]
#         ax.plot(t_axis, t_data, color=c_teacher, linewidth=3, alpha=0.4, label='Teacher')
#         ax.plot(t_axis, s_data, color=s_color, linestyle='--', linewidth=2, label=s_label)
#         ax.set_title(title, fontsize=11, fontweight='bold')
#         ax.set_ylim(ylim)
#         ax.legend(loc='upper right', fontsize=9)
#         ax.grid(True, linestyle=':', alpha=0.5)
#
#         # 强制 X 轴为整数
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#
#         # [修改点]：移除了 `if row_idx == 1:` 的限制
#         # 现在无论是 Baseline (Row 0) 还是 Ours (Row 1)，都会显示 x-label
#         ax.set_xlabel("Time Step")
#
#     # ==================== Row 1: Baseline (DT^2W) ====================
#     plot_col(0, 0, t_raw, data_dict['b_raw'], c_baseline, label_baseline, f"(a1) {label_baseline}: Raw", ylim_raw)
#     plot_col(0, 1, data_dict['t_dc'], data_dict['b_dc'], c_baseline, label_baseline, f"(a2) {label_baseline}: DC (Mean)", ylim_dc)
#     plot_col(0, 2, data_dict['t_low_dc'], data_dict['b_low_dc'], c_baseline, label_baseline, f"(a3) {label_baseline}: Low+DC", ylim_low_dc)
#     plot_col(0, 3, data_dict['t_low_only'], data_dict['b_low_only'], c_baseline, label_baseline, f"(a4) {label_baseline}: Low (No DC)", ylim_low_only)
#     plot_col(0, 4, data_dict['t_high'], data_dict['b_high'], c_baseline, label_baseline, f"(a5) {label_baseline}: High", ylim_high)
#
#     # ==================== Row 2: Ours ====================
#     plot_col(1, 0, t_raw, data_dict['o_raw'], c_ours, 'Ours', "(b1) Ours: Raw", ylim_raw)
#     plot_col(1, 1, data_dict['t_dc'], data_dict['o_dc'], c_ours, 'Ours', "(b2) Ours: DC (Mean)", ylim_dc)
#     plot_col(1, 2, data_dict['t_low_dc'], data_dict['o_low_dc'], c_ours, 'Ours', "(b3) Ours: Low+DC", ylim_low_dc)
#     plot_col(1, 3, data_dict['t_low_only'], data_dict['o_low_only'], c_ours, 'Ours', "(b4) Ours: Low (No DC)", ylim_low_only)
#     plot_col(1, 4, data_dict['t_high'], data_dict['o_high'], c_ours, 'Ours', "(b5) Ours: High", ylim_high)
#
#     plt.suptitle(f"Comprehensive Frequency Decomposition{title_suffix}", fontsize=16, y=0.98)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         print(f"  -> Saved: {save_path}")
#     plt.close(fig)
#
#
# # ==========================================
# # 辅助函数
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
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_uwave')
#
#     # === 请在这里修改你的实验文件夹名 ===
#     ours_exp_path = 'Dkfd_BN_Jan-05-11-17-22-judge'
#     baseline_exp_path = 'DT2W_BN_Dec-29-21-31-38'
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     seed_fixer(0)
#
#     # 路径检查
#     teacher_ckpt = os.path.join(base_result_path, 'Dkfd_BN_Dec-29-22-09-35-judge', 'teacher_after_task0_r0.pt')
#     ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'final_model_r0.pt')
#     baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'final_model_r0.pt')
#
#     for path, name in [(teacher_ckpt, 'Teacher'), (ours_ckpt, 'Ours'), (baseline_ckpt, 'Baseline')]:
#         if not os.path.exists(path):
#             print(f"Error: {name} ckpt not found: {path}")
#             return
#
#     # 加载模型
#     args_t = create_model_args(device=device, agent='Dkfd', head='Linear')
#     teacher_model = setup_model(args_t)
#     teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
#     teacher_model.eval()
#
#     args_o = create_model_args(device=device, agent='Dkfd', head='Linear')
#     ours_model = setup_model(args_o)
#     ours_model.update_head(n_new=2, task_now=1)
#     ours_model.to(device)
#     ours_model.load_state_dict(torch.load(ours_ckpt, map_location=device))
#     ours_model.eval()
#
#     args_b = create_model_args(device=device, agent='DT2W', head='Linear')
#     baseline_model = setup_model(args_b)
#     baseline_model.update_head(n_new=2, task_now=1)
#     baseline_model.to(device)
#     baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
#     baseline_model.eval()
#
#     # 数据准备
#     cls_order = [2, 1, 6, 0]
#     task_stream = IncrementalTaskStream(data='uwave', scenario='class', cls_order=cls_order, split='exp')
#     task_stream.setup(load_subject=False)
#     x_test, y_test = task_stream.tasks[0][2]
#     test_loader = Dataloader_from_numpy(x_test, y_test, batch_size=32, shuffle=False)
#
#     # 提取与绘图
#     fd_filter = TriBandFDFilter().to(device)
#     results = get_real_feature_data(
#         teacher_model, baseline_model, ours_model, test_loader, device, fd_filter, top_k=10
#     )
#
#     save_dir = os.path.join(base_result_path, ours_exp_path, 'freq_visualization_2x9')
#     os.makedirs(save_dir, exist_ok=True)
#
#     for data in results:
#         print(f"Plotting Rank {data['rank']}...")
#         save_path = os.path.join(save_dir, f'full_spec_rank{data["rank"]:02d}.png')
#
#         # 使用 2x5 绘图函数
#         plot_visualization_2x5(
#             data,
#             save_path=save_path,
#             title_suffix=f" (Rank {data['rank']}, Ch {data['channel']})"
#         )
#
#     print(f"Done. Images saved to {save_dir}")
#
#
# if __name__ == '__main__':
#     main()


# # -*- coding: UTF-8 -*-
# """
# 原始特征对齐可视化脚本 (独立图版)
# 功能：分别为 Baseline(DT2W) 和 Ours(Dkfd) 绘制与 Teacher 的原始特征对比图
# """
# import random
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import sys
# import os
# from types import SimpleNamespace
# from matplotlib.ticker import MaxNLocator
#
# # 获取项目根目录并切换工作目录
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(project_root)
# sys.path.insert(0, project_root)
#
# from models.base import setup_model
# from utils.stream import IncrementalTaskStream
# from utils.data import Dataloader_from_numpy
# from utils.utils import TriBandFDFilter, seed_fixer
#
#
# def get_real_feature_data(teacher_model, baseline_model, ours_model, dataloader, device, fd_filter, top_k=10):
#     """
#     提取特征，筛选策略：
#     1. 确保高频有波动 (Amp_High > Threshold)
#     2. 在此基础上，寻找低频能量占比最大的样本 (Maximize Low/High Ratio)
#     """
#     teacher_model.eval()
#     baseline_model.eval()
#     ours_model.eval()
#
#     all_candidates = []
#
#     print(f"Scanning samples to find 'Low >> High' but 'High is Active' cases...")
#
#     with torch.no_grad():
#         for batch_idx, (x, y) in enumerate(dataloader):
#             x = x.to(device)
#             # 获取 Teacher 特征
#             t_fmap = teacher_model.feature_map(x)
#
#             # 解耦用于筛选计算
#             t_dc, t_low_only, t_high = fd_filter(t_fmap)
#
#             # 计算幅度
#             amp_low = torch.mean(torch.abs(t_low_only), dim=-1)
#             amp_high = torch.mean(torch.abs(t_high), dim=-1)
#
#             B, C = t_fmap.shape[0], t_fmap.shape[1]
#             for b in range(B):
#                 for c in range(C):
#                     a_low = amp_low[b, c].item()
#                     a_high = amp_high[b, c].item()
#
#                     # 记录所有候选者
#                     all_candidates.append({
#                         'amp_low': a_low,
#                         'amp_high': a_high,
#                         'ratio': a_low / (a_high + 1e-8),
#                         'x_sample': x[b].unsqueeze(0).clone(),
#                         'c_idx': c
#                     })
#
#     # ================= 核心筛选逻辑 =================
#     high_amps = [c['amp_high'] for c in all_candidates]
#     threshold_high = np.percentile(high_amps, 40)
#     print(f"High-Freq Threshold (Top 60% activity): {threshold_high:.4f}")
#
#     active_candidates = [c for c in all_candidates if c['amp_high'] > threshold_high]
#     print(f"Candidates after filtering silent high-freq: {len(active_candidates)} / {len(all_candidates)}")
#
#     active_candidates.sort(key=lambda x: x['ratio'], reverse=True)
#     top_candidates = active_candidates[:top_k]
#     # ==================================================
#
#     print(f"Selected top {top_k} samples fitting the criteria.")
#
#     results = []
#     for i, cand in enumerate(top_candidates):
#         x_sample = cand['x_sample'].to(device)
#         c_idx = cand['c_idx']
#
#         # Forward
#         t_fmap = teacher_model.feature_map(x_sample)
#         b_fmap = baseline_model.feature_map(x_sample)
#         o_fmap = ours_model.feature_map(x_sample)
#
#         def to_np(tensor): return tensor[0, c_idx, :].detach().cpu().numpy()
#
#         results.append({
#             'rank': i + 1,
#             'channel': c_idx,
#             # 只保留 Raw 数据用于绘图
#             't_raw': to_np(t_fmap),
#             'b_raw': to_np(b_fmap),
#             'o_raw': to_np(o_fmap),
#         })
#
#     return results
#
#
# def plot_raw_contrast_separate(data_dict, save_dir):
#     """
#     [修改] 分别绘制两张图：
#     1. Baseline vs Teacher
#     2. Ours vs Teacher
#     注意：两张图共享相同的 Y 轴范围，以保证公平对比。
#     """
#     rank = data_dict['rank']
#     t_raw = data_dict['t_raw']
#     b_raw = data_dict['b_raw']
#     o_raw = data_dict['o_raw']
#
#     t_axis = np.arange(len(t_raw))
#
#     # 配色
#     c_teacher = 'black'
#     c_baseline = '#D62728'  # Red
#     c_ours = '#2CA02C'  # Green
#
#     # Label 名称
#     label_baseline = r'DT$^2$W'
#     label_ours = 'HiDe (Ours)'
#
#     # 1. 计算统一的 Y 轴范围 (基于三者数据的最大最小值)
#     all_vals = np.concatenate([t_raw, b_raw, o_raw])
#     ymin, ymax = np.min(all_vals), np.max(all_vals)
#     margin = (ymax - ymin) * 0.15  # 留一点边距
#     if margin == 0: margin = 0.5
#     ylim_shared = (ymin - margin, ymax + margin)
#
#     # 定义通用单图绘制函数
#     def draw_single_plot(s_data, s_color, s_label, file_suffix, title_text):
#         fig, ax = plt.subplots(figsize=(8, 5), dpi=300)  # 适合论文的尺寸
#         plt.rcParams['font.family'] = 'serif'
#
#         # 画 Teacher (作为背景基准)
#         ax.plot(t_axis, t_raw, color=c_teacher, linewidth=3.5, alpha=0.3, label='Teacher')
#         # 画 Student (Baseline 或 Ours)
#         ax.plot(t_axis, s_data, color=s_color, linestyle='--', linewidth=2.5, label=s_label)
#
#         # 样式设置
#         ax.set_title(title_text, fontsize=14, fontweight='bold', pad=10)
#         ax.set_ylim(ylim_shared)  # 关键：应用统一范围
#         ax.set_xlabel("Time Step", fontsize=12)
#         ax.set_ylabel("Feature Value", fontsize=12)
#
#         # 刻度设置
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         ax.grid(True, linestyle=':', alpha=0.6)
#
#         # 图例
#         ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9)
#
#         plt.tight_layout()
#
#         # 保存
#         fname = f"rank{rank:02d}_{file_suffix}.png"
#         full_path = os.path.join(save_dir, fname)
#         plt.savefig(full_path, bbox_inches='tight')
#         plt.close(fig)
#         print(f"  -> Saved: {fname}")
#
#     # 2. 绘制 Baseline 图
#     draw_single_plot(b_raw, c_baseline, label_baseline, "baseline", f"Baseline ({label_baseline}) vs Teacher")
#
#     # 3. 绘制 Ours 图
#     draw_single_plot(o_raw, c_ours, label_ours, "ours", f"{label_ours} vs Teacher")
#
#
# # ==========================================
# # 辅助函数
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
# # def main():
# #     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# #     base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_uwave')
# #
# #     # === 请在这里修改你的实验文件夹名 ===
# #     ours_exp_path = 'Dkfd_LN_Jan-10-17-28-50-judge'
# #     baseline_exp_path = 'DT2W_LN_Jan-10-20-57-02'
# #
# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #     print(f"Using device: {device}")
# #     seed_fixer(0)
# #
# #     # 路径检查
# #     teacher_ckpt = os.path.join(base_result_path, 'Dkfd_LN_Jan-10-17-28-50-judge', 'teacher_after_task0_r0.pt')
# #     ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'final_model_r0.pt')
# #     baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'final_model_r0.pt')
# #
# #     for path, name in [(teacher_ckpt, 'Teacher'), (ours_ckpt, 'Ours'), (baseline_ckpt, 'Baseline')]:
# #         if not os.path.exists(path):
# #             print(f"Error: {name} ckpt not found: {path}")
# #             return
# #
# #     # 加载模型
# #     args_t = create_model_args(device=device, agent='Dkfd', head='Linear')
# #     teacher_model = setup_model(args_t)
# #     teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
# #     teacher_model.eval()
# #
# #     args_o = create_model_args(device=device, agent='Dkfd', head='Linear')
# #     ours_model = setup_model(args_o)
# #     ours_model.update_head(n_new=2, task_now=1)
# #     ours_model.to(device)
# #     ours_model.load_state_dict(torch.load(ours_ckpt, map_location=device))
# #     ours_model.eval()
# #
# #     args_b = create_model_args(device=device, agent='DT2W', head='Linear')
# #     baseline_model = setup_model(args_b)
# #     baseline_model.update_head(n_new=2, task_now=1)
# #     baseline_model.to(device)
# #     baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
# #     baseline_model.eval()
# #
# #     # 数据准备
# #     cls_order = [2, 1, 6, 0]
# #     task_stream = IncrementalTaskStream(data='uwave', scenario='class', cls_order=cls_order, split='exp')
# #     task_stream.setup(load_subject=False)
# #     x_test, y_test = task_stream.tasks[0][2]
# #     test_loader = Dataloader_from_numpy(x_test, y_test, batch_size=32, shuffle=False)
# #
# #     # 提取
# #     fd_filter = TriBandFDFilter().to(device)
# #     results = get_real_feature_data(
# #         teacher_model, baseline_model, ours_model, test_loader, device, fd_filter, top_k=10
# #     )
# #
# #     # 保存路径
# #     save_dir = os.path.join(base_result_path, ours_exp_path, 'raw_feature_contrast')
# #     os.makedirs(save_dir, exist_ok=True)
# #
# #     # 绘图
# #     for data in results:
# #         print(f"Plotting Rank {data['rank']}...")
# #         plot_raw_contrast_separate(data, save_dir)
# #
# #     print(f"Done. Images saved to {save_dir}")
#
#
# # ==========================================
# # 修改后的 main 函数
# # ==========================================
# def main():
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     # 1. 修改数据集名称为 har
#     base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_har')
#
#     # === 请确保这里填写的文件夹名与你 HAR 实验的结果一致 ===
#     ours_exp_path = 'Dkfd_LN_Jan-10-17-28-50-judge'
#     baseline_exp_path = 'DT2W_LN_Jan-10-20-57-02'
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     seed_fixer(0)
#
#     # 路径检查
#     teacher_ckpt = os.path.join(base_result_path, ours_exp_path, 'teacher_after_task0_r0.pt')
#     ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'final_model_r0.pt')
#     baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'final_model_r0.pt')
#
#     for path, name in [(teacher_ckpt, 'Teacher'), (ours_ckpt, 'Ours'), (baseline_ckpt, 'Baseline')]:
#         if not os.path.exists(path):
#             print(f"Error: {name} ckpt not found: {path}")
#             return
#
#     # 2. 修改 create_model_args 中的参数: data='har', norm='LN'
#     args_t = create_model_args(data='har', device=device, agent='Dkfd', norm='LN')
#     teacher_model = setup_model(args_t)
#     teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
#     teacher_model.eval()
#
#     # ---------------- Ours 模型加载 ----------------
#     args_o = create_model_args(data='har', device=device, agent='Dkfd', norm='LN')
#     ours_model = setup_model(args_o)
#
#     # 1. 先读取 Checkpoint 字典
#     ours_ckpt_dict = torch.load(ours_ckpt, map_location=device)
#     # 2. 动态获取权重中 head 的输出维度 (应该是 6)
#     correct_out_features = ours_ckpt_dict['head.fc.weight'].shape[0]
#     # 3. 强制重置 Head 维度以匹配 Checkpoint
#     in_features = ours_model.head.fc.in_features
#     ours_model.head.fc = torch.nn.Linear(in_features, correct_out_features).to(device)
#
#     # 4. 现在加载就不会报错了
#     ours_model.load_state_dict(ours_ckpt_dict)
#     ours_model.eval()
#     print(f"Successfully loaded Ours model with {correct_out_features} classes.")
#
#     # ---------------- Baseline 模型加载 ----------------
#     args_b = create_model_args(data='har', device=device, agent='DT2W', norm='LN')
#     baseline_model = setup_model(args_b)
#
#     baseline_ckpt_dict = torch.load(baseline_ckpt, map_location=device)
#     # 同样强制重置 Baseline 的 Head
#     correct_out_features_b = baseline_ckpt_dict['head.fc.weight'].shape[0]
#     baseline_model.head.fc = torch.nn.Linear(in_features, correct_out_features_b).to(device)
#
#     baseline_model.load_state_dict(baseline_ckpt_dict)
#     baseline_model.eval()
#     print(f"Successfully loaded Baseline model with {correct_out_features_b} classes.")
#
#     # 3. 数据准备: 修改 data='har' 和 cls_order=[2, 1, 5, 0]
#     cls_order = [2, 1, 5, 0, 4, 3]  # HAR 数据集的类顺序
#     task_stream = IncrementalTaskStream(data='har', scenario='class', cls_order=cls_order, split='exp')
#     task_stream.setup(load_subject=False)
#
#     # 我们提取 Task 0 的数据来观察 Student 对旧知识特征的对齐情况
#     x_test, y_test = task_stream.tasks[0][2]
#     test_loader = Dataloader_from_numpy(x_test, y_test, batch_size=32, shuffle=False)
#
#     # 提取与过滤
#     fd_filter = TriBandFDFilter().to(device)
#     results = get_real_feature_data(
#         teacher_model, baseline_model, ours_model, test_loader, device, fd_filter, top_k=10
#     )
#
#     # 保存路径
#     save_dir = os.path.join(base_result_path, ours_exp_path, 'raw_feature_contrast_har')
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 绘图
#     for data in results:
#         print(f"Plotting Rank {data['rank']} (Channel {data['channel']})...")
#         plot_raw_contrast_separate(data, save_dir)
#
#     print(f"Done. Images saved to {save_dir}")
#
#
# if __name__ == '__main__':
#     main()


# -*- coding: UTF-8 -*-
"""
频域对齐可视化脚本 (2行5列版)
功能：对比 Teacher, Baseline(DT2W), Ours(Dkfd) 在 [原始, DC, 含DC低频, 不含DC低频, 高频] 的对齐效果
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
from types import SimpleNamespace
from matplotlib.ticker import MaxNLocator

# 环境初始化
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from models.base import setup_model
from utils.stream import IncrementalTaskStream
from utils.data import Dataloader_from_numpy
from utils.utils import TriBandFDFilter, seed_fixer


# def get_real_feature_data(teacher_model, baseline_model, ours_model, dataloader, device, fd_filter, top_k=10):
#     """提取特征并按频域特性筛选样本"""
#     teacher_model.eval()
#     baseline_model.eval()
#     ours_model.eval()
#     all_candidates = []
#
#     print(f"Scanning samples for frequency analysis...")
#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.to(device)
#             t_fmap = teacher_model.feature_map(x)
#             # 解耦获取组件
#             t_dc, t_low_only, t_high = fd_filter(t_fmap)
#
#             amp_low = torch.mean(torch.abs(t_low_only), dim=-1)
#             amp_high = torch.mean(torch.abs(t_high), dim=-1)
#
#             B, C = t_fmap.shape[0], t_fmap.shape[1]
#             for b in range(B):
#                 for c in range(C):
#                     all_candidates.append({
#                         'amp_low': amp_low[b, c].item(),
#                         'amp_high': amp_high[b, c].item(),
#                         'ratio': amp_low[b, c].item() / (amp_high[b, c].item() + 1e-8),
#                         'x_sample': x[b].unsqueeze(0).clone(),
#                         'c_idx': c
#                     })
#
#     # 筛选逻辑：选择高频活跃且低频趋势明显的样本
#     high_amps = [c['amp_high'] for c in all_candidates]
#     threshold_high = np.percentile(high_amps, 40)
#     active_candidates = [c for c in all_candidates if c['amp_high'] > threshold_high]
#     active_candidates.sort(key=lambda x: x['ratio'], reverse=True)
#     top_candidates = active_candidates[:top_k]
#
#     results = []
#     for i, cand in enumerate(top_candidates):
#         x_sample = cand['x_sample'].to(device)
#         c_idx = cand['c_idx']
#
#         # 获取各模型特征
#         t_f = teacher_model.feature_map(x_sample)
#         b_f = baseline_model.feature_map(x_sample)
#         o_f = ours_model.feature_map(x_sample)
#
#         # 频域分解
#         def decompose(f):
#             dc, low, high = fd_filter(f)
#             return dc, low, dc + low, high
#
#         t_dc, t_lo, t_ldc, t_hi = decompose(t_f)
#         b_dc, b_lo, b_ldc, b_hi = decompose(b_f)
#         o_dc, o_lo, o_ldc, o_hi = decompose(o_f)
#
#         def to_np(tensor): return tensor[0, c_idx, :].detach().cpu().numpy()
#
#         results.append({
#             'rank': i + 1, 'channel': c_idx,
#             't_raw': to_np(t_f), 'b_raw': to_np(b_f), 'o_raw': to_np(o_f),
#             't_dc': to_np(t_dc), 'b_dc': to_np(b_dc), 'o_dc': to_np(o_dc),
#             't_low_dc': to_np(t_ldc), 'b_low_dc': to_np(b_ldc), 'o_low_dc': to_np(o_ldc),
#             't_low_only': to_np(t_lo), 'b_low_only': to_np(b_lo), 'o_low_only': to_np(o_lo),
#             't_high': to_np(t_hi), 'b_high': to_np(b_hi), 'o_high': to_np(o_hi)
#         })
#     return results


def get_real_feature_data(teacher_model, baseline_model, ours_model, dataloader, device, fd_filter, top_k=10):
    """
    提取特征并筛选样本：
    修改点：改为挑选 amp_high (高频分量幅度) 最大的样本
    """
    teacher_model.eval()
    baseline_model.eval()
    ours_model.eval()
    all_candidates = []

    print(f"Scanning samples to find high-frequency 'Detail' active cases...")
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            t_fmap = teacher_model.feature_map(x)

            # 使用频域过滤器获取高频部分
            _, _, t_high = fd_filter(t_fmap)

            # 计算高频部分的幅度 (Mean Absolute Value)
            amp_high = torch.mean(torch.abs(t_high), dim=-1)  # [B, C]

            B, C = t_fmap.shape[0], t_fmap.shape[1]
            for b in range(B):
                for c in range(C):
                    a_high = amp_high[b, c].item()
                    all_candidates.append({
                        'amp_high': a_high,
                        'x_sample': x[b].unsqueeze(0).clone(),
                        'c_idx': c
                    })

    # ================= 核心筛选逻辑修改 =================
    # 直接按高频能量 (amp_high) 从大到小排序
    all_candidates.sort(key=lambda x: x['amp_high'], reverse=True)

    # 挑选前 top_k 个最高能量的样本
    top_candidates = all_candidates[:top_k]
    # ==================================================

    print(f"Selected top {top_k} samples with strongest High-Frequency fluctuations.")

    results = []
    for i, cand in enumerate(top_candidates):
        x_sample = cand['x_sample'].to(device)
        c_idx = cand['c_idx']

        # 获取各模型特征
        t_f = teacher_model.feature_map(x_sample)
        b_f = baseline_model.feature_map(x_sample)
        o_f = ours_model.feature_map(x_sample)

        # 频域分解逻辑
        def decompose(f):
            dc, low, high = fd_filter(f)
            return dc, low, dc + low, high

        t_dc, t_lo, t_ldc, t_hi = decompose(t_f)
        b_dc, b_lo, b_ldc, b_hi = decompose(b_f)
        o_dc, o_lo, o_ldc, o_hi = decompose(o_f)

        def to_np(tensor): return tensor[0, c_idx, :].detach().cpu().numpy()

        results.append({
            'rank': i + 1,
            'channel': c_idx,
            'amp_high': cand['amp_high'],  # 记录幅度以便确认
            't_raw': to_np(t_f), 'b_raw': to_np(b_f), 'o_raw': to_np(o_f),
            't_dc': to_np(t_dc), 'b_dc': to_np(b_dc), 'o_dc': to_np(o_dc),
            't_low_dc': to_np(t_ldc), 'b_low_dc': to_np(b_ldc), 'o_low_dc': to_np(o_ldc),
            't_low_only': to_np(t_lo), 'b_low_only': to_np(b_lo), 'o_low_only': to_np(o_lo),
            't_high': to_np(t_hi), 'b_high': to_np(b_hi), 'o_high': to_np(o_hi)
        })
    return results
def plot_visualization_2x5(data_dict, save_path):
    """绘制 2x5 对比图"""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), dpi=150)
    plt.rcParams['font.family'] = 'serif'

    t_axis = np.arange(len(data_dict['t_raw']))
    colors = {'teacher': 'black', 'baseline': '#D62728', 'ours': '#2CA02C'}
    labels = {'baseline': r'DT$^2$W', 'ours': 'HiDe (Ours)'}

    columns = [
        ('Raw Feature', ['t_raw', 'b_raw', 'o_raw']),
        ('DC (Trend)', ['t_dc', 'b_dc', 'o_dc']),
        ('Low + DC', ['t_low_dc', 'b_low_dc', 'o_low_dc']),
        ('Low (Residual)', ['t_low_only', 'b_low_only', 'o_low_only']),
        ('High (Detail)', ['t_high', 'b_high', 'o_high'])
    ]

    for col_idx, (title, keys) in enumerate(columns):
        # 统一 Y 轴
        all_vals = np.concatenate([data_dict[k] for k in keys])
        ylim = (np.min(all_vals) - 0.2, np.max(all_vals) + 0.2)

        # Row 0: Baseline vs Teacher
        ax0 = axes[0, col_idx]
        ax0.plot(t_axis, data_dict[keys[0]], color=colors['teacher'], lw=3, alpha=0.3, label='Teacher')
        ax0.plot(t_axis, data_dict[keys[1]], color=colors['baseline'], ls='--', lw=2, label=labels['baseline'])
        ax0.set_title(f"Baseline: {title}", fontsize=12, fontweight='bold')

        # Row 1: Ours vs Teacher
        ax1 = axes[1, col_idx]
        ax1.plot(t_axis, data_dict[keys[0]], color=colors['teacher'], lw=3, alpha=0.3, label='Teacher')
        ax1.plot(t_axis, data_dict[keys[2]], color=colors['ours'], ls='--', lw=2, label=labels['ours'])
        ax1.set_title(f"Ours: {title}", fontsize=12, fontweight='bold')

        for ax in [ax0, ax1]:
            ax.set_ylim(ylim)
            ax.grid(True, ls=':', alpha=0.5)
            ax.legend(loc='upper right', fontsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def create_model_args(data='dailysports', agent='Dkfd', norm='LN'):
    return SimpleNamespace(data=data, encoder='CNN', head='Linear', norm=norm,
                           feature_dim=128, n_layers=4, dropout=0.3, device='cuda',
                           agent=agent, input_norm='LN', stream_split='exp')


def main():
    base_result_path = os.path.join(project_root, 'result/tune_and_exp/CNN_dailysports')
    ours_exp_path = 'Dkfd_LN_Jan-12-14-48-33-judge'
    baseline_exp_path = 'DT2W_LN_Jan-12-14-43-12'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_fixer(0)

    # 模型加载函数 (含动态 Head 调整)
    def load_model(path, agent):
        args = create_model_args(agent=agent)
        model = setup_model(args)
        sd = torch.load(path, map_location=device)
        if 'head.fc.weight' in sd:
            out_dim = sd['head.fc.weight'].shape[0]
            model.head.fc = torch.nn.Linear(model.head.fc.in_features, out_dim)
        model.to(device).load_state_dict(sd)
        return model.eval()

    teacher_ckpt = os.path.join(base_result_path, ours_exp_path, 'teacher_after_task0_r0.pt')
    ours_ckpt = os.path.join(base_result_path, ours_exp_path, 'final_model_r0.pt')
    baseline_ckpt = os.path.join(base_result_path, baseline_exp_path, 'final_model_r0.pt')

    print("Loading models...")
    teacher = load_model(teacher_ckpt, 'Dkfd')
    ours = load_model(ours_ckpt, 'Dkfd')
    baseline = load_model(baseline_ckpt, 'DT2W')

    # 数据准备
    cls_order = [0,1,2,3]
    stream = IncrementalTaskStream(data='dailysports', scenario='class', cls_order=cls_order, split='exp')
    stream.setup(load_subject=False)
    x_test, y_test = stream.tasks[0][2]
    loader = Dataloader_from_numpy(x_test, y_test, batch_size=32, shuffle=False)

    # 提取与绘图
    fd_filter = TriBandFDFilter().to(device)
    results = get_real_feature_data(teacher, baseline, ours, loader, device, fd_filter)

    save_dir = os.path.join(base_result_path, ours_exp_path, 'freq_decouple_contrast_2')
    os.makedirs(save_dir, exist_ok=True)

    for data in results:
        print(f"Plotting Rank {data['rank']}...")
        plot_visualization_2x5(data, os.path.join(save_dir, f"rank{data['rank']:02d}_freq.png"))
    print(f"Done. Images saved to {save_dir}")


if __name__ == '__main__':
    main()