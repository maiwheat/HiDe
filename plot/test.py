import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from types import SimpleNamespace

# ==========================================
# 0. 路径修复：强制定位到项目根目录
# ==========================================
# 获取当前脚本所在目录 (TSCIL/plot) 的父目录 (TSCIL)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

# 导入项目内部模块
from models.base import setup_model
from utils.stream import IncrementalTaskStream
from utils.data import Dataloader_from_numpy


# ==========================================
# 1. 数据准备函数
# ==========================================
def prepare_data(data_name='uwave', tasks_to_use=[0, 1, 2, 3]):
    print(f"\n>>> 正在准备数据: {data_name}")
    # 设置与训练一致的类别顺序
    cls_order = [5, 0, 6, 7, 2, 4, 1, 3]

    # 初始化数据流
    task_stream = IncrementalTaskStream(data=data_name, scenario='class', cls_order=cls_order, split='exp')
    task_stream.setup(load_subject=False)

    x_list, y_list = [], []
    for t_idx in tasks_to_use:
        if t_idx < len(task_stream.tasks):
            # 获取测试集数据 (索引为 2)
            xt, yt = task_stream.tasks[t_idx][2]
            x_list.append(xt)
            y_list.append(yt)
            print(f"   - 已添加任务 {t_idx} 数据: {xt.shape}")

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    loader = Dataloader_from_numpy(x_all, y_all, batch_size=64, shuffle=False)
    return loader, y_all


# ==========================================
# 2. 模型加载函数
# ==========================================
def load_my_model(ckpt_path, device):
    print(f"\n>>> 正在加载模型权重: {ckpt_path}")
    # 构造基础参数
    args = SimpleNamespace(
        data='uwave', encoder='CNN', head='Linear', norm='BN',
        feature_dim=128, n_layers=4, dropout=0, device=device,
        agent='Dkfd', input_norm='IN', stream_split='exp'
    )
    model = setup_model(args)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    # 使用 strict=False 确保特征提取部分能正常加载
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# ==========================================
# 3. t-SNE 核心绘图逻辑
# ==========================================
def do_tsne_plot(model, loader, labels, device, save_name="tsne_single_model.png"):
    print("\n>>> 正在提取特征...")
    features = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            # 提取展平后的特征向量
            f_map = model.feature_map(x)
            features.append(f_map.view(f_map.size(0), -1).cpu().numpy())

    features = np.concatenate(features, axis=0)

    print(">>> 正在进行 t-SNE 降维 (可能需要 1-2 分钟)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca')
    embedded = tsne.fit_transform(features)

    # 绘图设置
    plt.figure(figsize=(10, 8), dpi=300)
    unique_classes = np.unique(labels)
    # 自动根据类别数分配颜色
    cmap = plt.get_cmap('tab10') if len(unique_classes) <= 10 else plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(unique_classes)))

    for i, cls in enumerate(unique_classes):
        idx = np.where(labels == cls)[0]
        plt.scatter(embedded[idx, 0], embedded[idx, 1],
                    color=colors[i], label=f'Class {int(cls)}',
                    alpha=0.7, s=40, edgecolors='white', linewidth=0.3)

    plt.title(f"t-SNE Visualization of Feature Space", fontsize=15, fontweight='bold')
    plt.legend(loc='center left', title="Classes", bbox_to_anchor=(1, 0.5), frameon=True)
    plt.axis('off')  # 隐藏坐标轴

    # 自动保存到当前脚本目录下的 result 文件夹
    save_dir = os.path.join(current_dir, "tsne_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)

    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n>>> 可视化完成！图像已保存至: {save_path}")
    plt.show()


# ==========================================
# 4. 主程序执行
# ==========================================
if __name__ == "__main__":
    # 配置运行环境
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 【请修改此处】指向你实际的 .pt 权重文件路径
    # 建议使用绝对路径或确保相对于 TSCIL 根目录的路径正确
    my_ckpt_path = "result/tune_and_exp/CNN_uwave/Dkfd_BN_Jan-21-14-54-04/ckpt_r3.pt"

    try:
        # 1. 准备数据 (UWave 通常 8 类分 4 任务，索引为 0,1,2,3)
        data_loader, y_labels = prepare_data(data_name='uwave', tasks_to_use=[0, 1, 2, 3])

        # 2. 加载模型
        model = load_my_model(my_ckpt_path, device)

        # 3. 执行可视化
        do_tsne_plot(model, data_loader, y_labels, device)

    except Exception as e:
        print(f"\n运行失败，错误原因: {e}")