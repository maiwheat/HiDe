import argparse
import sys
import os
import torch
import time

from ray.util.client import ray

from experiment.tune_and_exp import tune_and_experiment_multiple_runs
from utils.utils import Logger
from types import SimpleNamespace
from experiment.tune_config import config_default



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Hyper-parameter tuning')
    parser.add_argument('--data', dest='data', default='grabmyo', type=str)
    parser.add_argument('--encoder', dest='encoder', default='CNN', type=str)
    parser.add_argument('--agent', dest='agent', default='DT2W', type=str)
    parser.add_argument('--norm', dest='norm', default='BN', type=str)
    parser.add_argument('--lambda_kd_fmap_freq_used', dest='lambda_kd_fmap_freq_used', default=True, type=str)
    parser.add_argument('--lambda_kd_fmap_used', dest='lambda_kd_fmap_used', default=False, type=str)
    parser.add_argument('--name', type=str, default='', help='参数配置的附加名称')
    # 假设args通过parser定义，添加如下参数
    parser.add_argument('--pool_kernel', type=int, default=4, help='时序池化核大小')
    parser.add_argument('--pool_stride', type=int, default=None, help='时序池化步长，默认等于核大小')
    # 解析命令行参数，并将结果存储在args对象中。
    args = parser.parse_args()
    # Include unchanged general params
    # 使用SimpleNamespace将解析后的参数和默认参数合并，存储在args对象中。
    args = SimpleNamespace(**vars(args), **config_default)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\n ######## Run Used {}########'.format(args.device))

    print('\n ######## Run Used Data {}########'.format(args.data))
    # Set directories
    exp_start_time = time.strftime("%b-%d-%H-%M-%S", time.localtime())
    exp_path_1 = args.encoder + '_' + args.data
    exp_path_2 = args.agent + '_' + args.norm + '_' + exp_start_time + args.name
    exp_path = os.path.join(args.path_prefix, exp_path_1, exp_path_2)  # Path for running the whole experiment
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    # args.exp_path 是在代码中定义的一个属性名，用于存储解析后的命令行参数。这个属性名是自定义的，可以根据你的需求进行命名。
    args.exp_path = exp_path
    log_path = args.exp_path + '/log.txt'
    sys.stdout = Logger('{}'.format(log_path))
    tune_and_experiment_multiple_runs(args)












