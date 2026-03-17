import torch
from . import soft_dtw
from . import path_soft_dtw
from .soft_dtw_cuda import SoftDTW
from .path_soft_dtw import PathDTW

def dilate_loss(outputs, targets, alpha, gamma, device):
    use_cuda = True if device == 'cuda' else False
    dtw = SoftDTW(use_cuda=use_cuda, gamma=gamma, normalize=False)
    loss_shape = dtw(outputs, targets)
    D_matrix = dtw.D_matrix


    N, L, D = outputs.shape



    # 时间对齐损失
    # path_dtw = path_soft_dtw.PathDTWBatch.apply
    # path = path_dtw(D_matrix, gamma)
    use_cuda = (device == 'cuda')
    path_dtw_func = PathDTW(use_cuda=use_cuda)
    path = path_dtw_func(D_matrix, gamma)
    Omega = soft_dtw.pairwise_distances(torch.arange(1, L + 1).view(L, 1)).to(device)
    loss_temporal = torch.sum(path * Omega) / (L * L)
    loss_shape_mean = loss_shape.mean()
    # 1. 自动计算对齐系数 (无梯度)
    balance_weight = loss_shape_mean.detach() / (loss_temporal.detach() + 1e-8)

    # 总损失
    loss =   loss_shape +  alpha * balance_weight * loss_temporal
    return loss