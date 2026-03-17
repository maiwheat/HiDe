# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def Dataloader_from_numpy(X, Y, batch_size, shuffle=True):
    """
    - Tensors in dataloader are in cpu.
    """
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y).long())
    g = torch.Generator()
    g.manual_seed(1235)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g,pin_memory=True)


def Dataloader_from_numpy_with_idx(X, idx, Y, batch_size, shuffle=True):
    """
    - Tensors in dataloader are in cpu.
    """
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(idx).long(), torch.Tensor(Y).long())
    g = torch.Generator()
    g.manual_seed(1235)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g,pin_memory=True)


def Dataloader_from_numpy_with_sub(X, Y, Sub, batch_size, shuffle=True):
    """
    - Tensors in dataloader are in cpu.
    """
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y).long(), torch.Tensor(Sub).long())
    g = torch.Generator()
    g.manual_seed(1235)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g,pin_memory=True)

# 根据类标签提取对应的训练数据
def extract_samples_according_to_labels(x, y, target_ids, return_inds=False):
    """
    Extract corresponding samples from x and y according to the labels
    :param x: data, np array
    :param y: labels, np array
    :param target_ids: list of labels
    :return:
    """
    # get the indices
    # 使用map函数和lambda表达式，遍历y中的每个元素，检查它是否在target_ids列表中。结果是一个布尔值的列表，表示每个标签是否为目标标签。
    inds = list(map(lambda x: x in target_ids, y))
    x_extracted = x[inds]
    y_extracted = y[inds]

    if return_inds:
        return x_extracted, y_extracted, inds
    else:
        return x_extracted, y_extracted


def extract_samples_according_to_labels_with_sub(x, y, sub, target_ids, return_inds=False):
    """
    Extract corresponding samples with subject label from x and y according to the labels
    :param x: data, np array
    :param y: labels, np array
    :param sub: subject labels, np array
    :param target_ids: list of labels
    :return:
    """
    # get the indices
    inds = list(map(lambda x: x in target_ids, y))
    x_extracted = x[inds]
    y_extracted = y[inds]
    sub_extracted = sub[inds]

    if return_inds:
        return x_extracted, y_extracted, sub_extracted, inds
    else:
        return x_extracted, y_extracted, sub_extracted


def extract_samples_according_to_subjects(x, y, sub, target_ids, return_inds=False):
    # get the indices
    inds = list(map(lambda x: x in target_ids, sub))
    x_extracted = x[inds]
    y_extracted = y[inds]

    if return_inds:
        return x_extracted, y_extracted, inds
    else:
        return x_extracted, y_extracted


def extract_n_samples_randomly(x, y, n_sample):
    """
    Randomly extract n samples from x and y
    :param x: data, np array
    :param y: labels, np array
    :param n_sample: Number of samples to extract
    :return: extracted data & labels
    """
    sampled_idx = np.random.randint(0, len(y), size=n_sample)
    return x[sampled_idx], y[sampled_idx]

