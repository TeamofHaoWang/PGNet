# -*- coding: utf-8 -*-
import os
# %%
import warnings

from Model import SDAGCN

warnings.filterwarnings("ignore")
# %%
from Experiment.Experiment_merge import Exp_merge
from Experiment.Experiment_PINN import Exp_PINN
from Experiment.Experiment_PGNet import Exp_PGNet
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False
if __name__ == '__main__':

    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    # Model-related arguments
    parser.add_argument('--task', default='normal', type=str, help='task options: [normal,PINN]')
    parser.add_argument('--model_name', default='TP_Diff_Degra_V2', type=str,
                        help='Model options: [PINN,LeNet,LSTM,Transformer,Autoformer,PatchTST,AGCNN,Dual_Mixer'
    ',FCSTGNN,DegraNet,Transformer_domain,DA_LSTM,SDAGCN,PINN,CDSG,TwoP_Transformer,TP_Diff_Degra_V2]')
    parser.add_argument('--train', default=True, type=str2bool, help='Train or test')
    parser.add_argument('--resume', default=False, type=str2bool, help='Load checkpoint or not')
    parser.add_argument('--save_test', default=True, type=str2bool, help='Save predictions and targets when testing')
    parser.add_argument('--save_path', default=None, help='Path to save the results')
    parser.add_argument('--resume_path', default=None, help='Path to resume from')
    parser.add_argument('--train_epochs', default=1, type=int, help='Number of training epochs')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--loss_type', default='MSE', type=str, help='Loss type: [MSE, MAE]')
    parser.add_argument('--info', default='main test', type=str, help='Extra information')

    # Data loading parameters
    parser.add_argument('--data_root', default='D:/BaiduNetdiskDownload/RUL_framework', type=str, help='E:\RUL_Framework11.13 or /public3/sxb/RUL_framework/')
    parser.add_argument('--dataset_name', default='CMAPSS', type=str, help='Dataset options: [CMAPSS, N_CMAPSS, XJTU]')
    parser.add_argument('--Data_id_CMAPSS', default="FD003", type=str, help='For CMAPSS')
    parser.add_argument('--Data_id_N_CMAPSS', default="DS01", type=str, help='For N_CMAPSS [01 02 03 05 07]')
    parser.add_argument('--Data_id_XJTU', default="35", type=str, help='For XJTU [35 37 40]')
    parser.add_argument('--input_length', default=50, type=int, help='Input length')
    parser.add_argument('--validation', default=0.2, type=float, help='Validation split')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')

    # CMAPSS-specific data parameters
    parser.add_argument('--MAXLIFE_CMAPSS', default=125, type=int, help='Max life for CMAPSS')
    parser.add_argument('--normalization_CMAPSS', default="minmax", type=str, help='Normalization method for CMAPSS')

    # N_CMAPSS and XJTU-specific data parameters
    parser.add_argument('--s', type=int, default=10, help='Stride of window')
    parser.add_argument('--sampling', type=int, default=50, help='Subsampling of the given data')
    parser.add_argument('--change_len', type=str2bool, default=True, help='Re-generate data when input_len changes')
    parser.add_argument('--rate', type=float, default=0.8, help='Max life related rate')
    parser.add_argument('--xjtu_n_fea', type=int, default=40, help='num feature')

    # Model architecture parameters
    parser.add_argument('--d_model', default=64, type=int, help='Embedding dimension')
    parser.add_argument('--d_ff', default=128, type=int, help='Feedforward layer dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--patch_size', type=int, default=5, help='Patch size')

    # Specific model parameters (PV2, Dlinear)
    parser.add_argument('--lamda', default=0.1, type=float, help='Lambda for regularization')
    parser.add_argument('--kernel_size', default=3, type=int, help='Kernel size for Dlinear')
    parser.add_argument('--stride', default=1, type=int, help='Stride for Dlinear')

    # Diff prediction
    parser.add_argument('--is_minmax', default=True, type=str2bool, help='Enable min-max normalization XJTU needed')
    parser.add_argument('--is_diff', default=False, type=str2bool, help='Enable differential prediction')
    # Cross segmentation parameters
    parser.add_argument('--n_seg', type=int, default=8, help='Number of segments for cross attention')

    # Transformer-related parameters
    parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--factor', type=int, default=3, help='Attention factor')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')

    # PatchTST parameters
    parser.add_argument('--patch_len', type=int, default=5, help='Patch length')
    parser.add_argument('--patch_stride', type=int, default=5, help='Patch stride')

    # Autoformer parameters
    parser.add_argument('--moving_avg', default=[24], help='Window size for moving average')
    parser.add_argument('--embed', type=str, default='timeF', help='Time features encoding method')
    parser.add_argument('--freq', type=str, default='h', help='Frequency for time features encoding')

    # Domain Adaptation Transformer parameters
    parser.add_argument('--DA', default=False, type=str2bool, help='Enable domain adaptation')
    parser.add_argument('--source_domain', default="FD001", type=str, help='Source domain')
    parser.add_argument('--target_domain', default="FD001", type=str, help='Target domain')
    parser.add_argument('--type', type=int, default=2, help='Loss type for domain adaptation')

    # Classification-related parameters
    parser.add_argument('--Classify', default=False, type=str2bool, help='Enable classification')
    parser.add_argument('--D1_lr', default=0.0001, type=float, help='Learning rate for classifier')

    # Parse all arguments
    args = parser.parse_args()

    print(f"|{'=' * 101}|")
    # print all configs
    # 使用__dict__方法获取参数字典，之后遍历字典
    for key, value in args.__dict__.items():
        # turn all the config to str
        # 因为参数不一定都是str，需要将所有的数据转成str
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")

    # 设置数据路径和实验名称

    if args.data_root is not None:
        args.data_path = os.path.join(args.data_root, args.dataset_name)
    else:
        args.data_path = './' + args.dataset_name


    # 常见的训练设置
    args.use_gpu = torch.cuda.is_available()  # GPU可用时为True
    args.gpu = 0  # 默认使用GPU 0
    args.optimizer = "Adam"
    args.learning_rate_patience = 10  # 学习率耐心
    args.learning_rate_factor = 0.3  # 学习率衰减因子
    args.early_stop_patience = 3  # 提前停止耐心

    if args.task == 'normal':
        exp = Exp_merge(args)
        exp.start()
    elif args.task == 'PINN':
        exp = Exp_PINN(args)
        exp.start()

