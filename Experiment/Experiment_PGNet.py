import sys

sys.path.append("..")
import os
import numpy as np
import shutil
import torch
from torch import optim
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yaml
from N_CMAPSS_Related.N_CMAPSS_load_data import get_n_cmapss_data_, N_CMAPSSData_index
from CMAPSS_Related.load_data_CMAPSS import get_cmapss_data_index

from XJTU_Related.load_data_XJTU import get_xjtu_data_
from XJTU_Related.XJTU_Dataset import XJTUData,XJTUData_index
from torch.utils.data import DataLoader
from Model import *
from Experiment.Early_Stopping import EarlyStopping
from Experiment.learining_rate_adjust import adjust_learning_rate_class
from utils.calculate_metric import calculate_per_class_metrics
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

from tqdm import tqdm
from tool.Write_csv import *
import datetime

def get_xjtu_config(data_id, base_path='./XJTU/XJTU-SY_Bearing_Datasets'):
    folder_map = {
        "35": "35Hz12kN",
        "37": "37.5Hz11kN",
        "40": "40Hz10kN"
    }

    assert data_id in folder_map, f"Unsupported data_id: {data_id}"

    folder = folder_map[data_id]
    root_dir = os.path.join(base_path, folder)

    if folder=='35Hz12kN':
        train_bearing_data_set = ['Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5']
        test_bearing_data_set = ['Bearing1_1']

    elif folder=='37.5Hz11kN':
        train_bearing_data_set = ['Bearing2_1','Bearing2_2','Bearing2_3','Bearing2_4']
        test_bearing_data_set = ['Bearing2_5']

    elif folder=='40Hz10kN':
        train_bearing_data_set = ['Bearing3_1','Bearing3_2','Bearing3_4','Bearing3_5']
        test_bearing_data_set = ['Bearing3_3']

    return {
        "root_dir": root_dir,
        "train_bearing_data_set": train_bearing_data_set,
        "test_bearing_data_set": test_bearing_data_set
    }

class Exp_PGNet(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._get_path()

        # load dataset pretrain without test_loader
        self.train_data, self.train_loader, self.vali_data, \
        self.vali_loader, self.test_data, self.test_loader, self.input_feature = self._get_data()

        # build the Model:
        self.model = self._get_model()

        # What optimisers and loss functions can be used by the model
        self.optimizer_dict = {"Adam": optim.Adam}

    # choose device
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            print('Use GPU: cuda: ' + str(os.environ["CUDA_VISIBLE_DEVICES"]))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    # ------------------- function to build model -------------------------------------
    def _get_model(self):

        if self.args.model_name == 'TP_Diff_Degra_V2':
            model = TP_Diff_Degra_V2(self.args, save_path=self.args.save_path)

        print("Parameter :", np.sum([para.numel() for para in model.parameters()]))

        return model.double().to(self.device)

    # --------------------------- select optimizer ------------------------------
    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError

        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # ------------------------ get Dataloader -------------------------------------
    #  funnction of load CMPASS Dataset
    def _get_data(self):

        args = self.args
        # train and validation dataset
        if self.args.dataset_name == 'CMAPSS':
            X_train, y_train, index_train, X_vali, y_vali, index_vali, change_points_train, change_points_vali, X_test, y_test, index_test = get_cmapss_data_index(data_path=args.data_path,
                                                                                Data_id=args.Data_id_CMAPSS,
                                                                                sequence_length=args.input_length,
                                                                                MAXLIFE=args.MAXLIFE_CMAPSS,
                                                                                is_difference=args.is_diff,
                                                                                normalization=args.normalization_CMAPSS,
                                                                                validation=args.validation)
            self.max_life = args.MAXLIFE_CMAPSS

        elif self.args.dataset_name == 'N_CMAPSS':
            X_train, index_train, y_train, X_vali, index_vali, y_vali, X_test, index_test, y_test, self.max_life  = get_n_cmapss_data_(args=self.args,data_path=args.data_path,
                                                                                                 name=args.Data_id_N_CMAPSS)

        elif self.args.dataset_name == 'XJTU':
            data_paths_dict = get_xjtu_config(args.Data_id_XJTU,os.path.join(args.data_path,'XJTU-SY_Bearing_Datasets'))
            X_train, y_train, X_vali, y_vali, index_train, index_vali, X_test, y_test, index_test, self.max_life = get_xjtu_data_(
                root_dir=data_paths_dict['root_dir'],
                train_bearing_data_set=data_paths_dict['train_bearing_data_set'],
                test_bearing_data_set=data_paths_dict['test_bearing_data_set'],
                window_length=args.input_length,
                validation_rate=0.1,input_fea=self.args.xjtu_n_fea,sampling=self.args.sampling,stride=self.args.s,max_life_rate=self.args.rate)

        else:
            raise ValueError('without corresponding dataset')


        train_data_set = eval ( self.args.dataset_name + 'Data_index_point' ) ( X_train, index_train, y_train, change_points_train)
        vali_data_set = eval ( self.args.dataset_name + 'Data_index_point' ) ( X_vali, index_vali, y_vali, change_points_vali)
        test_data_set = eval ( self.args.dataset_name + 'Data_index' ) ( X_test, index_test, y_test )

        input_fea = X_test.shape[-1]
        args.input_feature = input_fea

        train_data_loader = DataLoader(dataset=train_data_set,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       drop_last=True)

        vali_data_loader = DataLoader(dataset=vali_data_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=True)

        test_data_loader = DataLoader(dataset=test_data_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=False)

        return train_data_set, train_data_loader, vali_data_set, vali_data_loader, test_data_set, test_data_loader, input_fea


    def save_hparam(self,per_epoch_time):
        # args: args from argparse return
        value2save = {k: v for k, v in vars(self.args).items() if not k.startswith('__') and not k.endswith('__')}
        # 删除影响使用的参数，以便加载
        del_key = ['train', 'resume', 'save_path', 'resume_path', 'batch_size', 'train_epochs', 'learning_rate']
        for key in del_key:
            del value2save[key]

        with open(os.path.join(self.save_path, 'hparam.yaml'), 'a+') as f:
            f.write(yaml.dump(value2save))
        with open(os.path.join(self.save_path, 'hparam.yaml'), 'a+') as f:
            f.write(yaml.dump(per_epoch_time))

    # 自动检测并转换字典中 NumPy 对象的函数
    def save_classify_results(self,classify_dict):

        # 将转换后的字典写入 YAML 文件
        with open(os.path.join(self.save_path, 'classify_results.yaml'), 'a+') as f:
            yaml.dump(classify_dict, f, default_flow_style=False)

    def _get_path(self):
        if not os.path.exists('./logs/'):
            os.makedirs('./logs/')

        exp_id = self.args.save_path
        # save address
        if self.args.dataset_name == 'CMAPSS':
            data = self.args.Data_id_CMAPSS
        elif self.args.dataset_name == 'N_CMAPSS':
            data = self.args.Data_id_N_CMAPSS
        elif self.args.dataset_name == 'XJTU':
            data = self.args.Data_id_XJTU
        else:
            data = self.args.dataset_name

        self.path = './logs/' + data
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.model_path = self.path + '/' + self.args.model_name
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if exp_id is not None and exp_id != 'None' and exp_id != 'none':
            self.save_path = self.model_path + '/' + exp_id
            if self.args.train:
                if os.path.exists(self.save_path):
                    shutil.rmtree(self.save_path)
                os.makedirs(self.save_path)

        else:
            # 如为None则自动加一
            path_list = os.listdir(self.model_path)
            if path_list == []:
                self.save_path = self.model_path + '/exp0'

            else:
                path_list = [int(idx[3:]) for idx in path_list]
                self.save_path = self.model_path + '/exp' + str(max(path_list) + 1)

            os.makedirs(self.save_path)
        self.args.save_path = self.save_path
        self.args.resume_path = self.save_path

    def _load_checkpoint(self):
        self.checkpoint_dir = self.args.save_path + '/best_checkpoint.pth'
        if os.path.exists(self.checkpoint_dir):
            check_point = torch.load(self.checkpoint_dir)
            self.model.load_state_dict(check_point)
        else:
            raise print('checkpoint is not exists')

    def start(self):

        # how many step of train and validation:
        train_steps = len(self.train_loader)
        vali_steps = len(self.vali_loader)
        print("train_steps: ", train_steps)
        print("validaion_steps: ", vali_steps)

        # initial early stopping
        early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)
        # initial learning rate
        learning_rate_adapter = adjust_learning_rate_class(self.args, True)
        # choose optimizer
        self.model_optim = self._select_optimizer()
        self.classify_train_criterion = nn.CrossEntropyLoss()

        if self.args.loss_type == 'MSE':
            self.pred_loss_criterion = nn.MSELoss()  # 使用均方误差损失
        elif self.args.loss_type == 'MAE':
            self.pred_loss_criterion = nn.L1Loss()  # 使用平均绝对误差损失
        elif self.args.loss_type == 'QUAN':
            self.pred_loss_criterion = QuantileLoss(quantile=0.3)  # 让模型保守预测
        else:
            raise ValueError("Unsupported loss type. Use 'MSE' or 'MAE'.")

        if self.args.resume:
            print('load checkpoint')
            self._load_checkpoint()
        else:
            print('random init')

        per_train_loss = {} #存储每一个训练epoch时间
        if self.args.train:
            # training process
            print("start pre-training")
            for epoch in range(self.args.train_epochs):
                # training process
                train_loss, epoch_time = self.training()
                per_train_loss[f"epoch_{epoch}"] = train_loss
                # validation process:
                vali_loss, classify_results = self.validation(self.vali_loader)

                self.save_classify_results(classify_results)

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}. it takse {4:.7f} seconds".format(
                        epoch + 1, train_steps, train_loss, vali_loss, epoch_time))

                # At the end of each epoch, Determine if we need to stop and adjust the learning rate
                early_stopping(vali_loss, self.model, self.save_path, pre_train=False)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                learning_rate_adapter(self.model_optim, vali_loss)

            # 读取最优的参数
            self.save_hparam(per_train_loss)

        # test:
        # 训练参数数目
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Paramaters 数量为:{0}'.format(total_params))

        average_enc_loss, average_enc_overall_loss, overall_score, test_average_time = self.test(self.test_loader)
        print(f"{self.args.dataset_name}: RMSE test performace of enc is: ", average_enc_loss, " of enc overall is: ",
              average_enc_overall_loss, 'socre of'
                                        'enc', overall_score)

        log_path = './logs/new_experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'batch_size', 'best_last_RMSE', 'best_overall_RMSE', 'score', 'windowsize',
                           'd_model', 'dropout', 'kernal_size', 'kernal_stride', 'train', 'total_params',
                           'savepath', 'resume',
                           'resumepath', 'info',
                           ]]
            write_csv(log_path, table_head, 'w+')

        time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间

        resume_dir = self.checkpoint_dir if self.args.resume else None

        if self.args.dataset_name == 'CMAPSS':
            dataset = self.args.Data_id_CMAPSS
        elif self.args.dataset_name == 'N_CMAPSS':
            dataset = self.args.Data_id_N_CMAPSS
        elif self.args.dataset_name == 'XJTU':
            dataset = self.args.Data_id_XJTU
        else:
            dataset = self.args.dataset_name

        a_log = [{'dataset': dataset, 'model': self.args.model_name, 'time': time_now,
                  'LR': self.args.learning_rate,
                  'batch_size': self.args.batch_size,
                  'best_last_RMSE': average_enc_loss, 'best_overall_RMSE': average_enc_overall_loss,
                  'score': overall_score, 'windowsize': self.args.input_length, 'd_model': self.args.d_model,
                  'dropout': self.args.dropout,
                  'kernal_size': self.args.kernel_size,
                  'kernal_stride': self.args.stride,
                  'train': self.args.train,
                  'total_params': total_params,
                  'savepath': self.save_path,
                  'resume': self.args.resume,
                  'resumepath': resume_dir,
                  'info': self.args.info,
                  }]
        write_csv_dict(log_path, a_log, 'a+')

        # # TODO score-cam
        # self.score_model = ScoreCAM(self.model)
        #
        # # test loader for decision making
        # self.model.eval()
        # enc_pred = []
        # gt = []
        # start_time = time()
        # for i, (batch_x, idx_x, batch_y) in enumerate(self.test_loader):
        #     batch_x = batch_x.double().double().to(self.device)
        #     batch_y = batch_y.double().double().to(self.device)
        #
        #     _, outputs = self.score_model(batch_x, mode='test', idx=idx_x, pre_train=False)
        #     outputs = outputs * self.max_life


    def create_labels(self, X, change_point_index):
        """
        Args:
            X: 输入张量，维度为 (batch_size, patch_num, feature_dim, patch_len)
            change_point_index: 变点索引张量，维度为 (batch_size,)，每个元素表示该batch中的变点位置（在patch_len维度上的索引）

        Returns:
            labels: 标签张量，维度为 (batch_size, patch_num, sensor_num)
                    其中，对于每个batch中的每个传感器的patch:
                        - 0: 如果该patch的所有索引都在变点之前
                        - 1: 如果该patch的索引范围包含变点
                        - 2: 如果该patch的所有索引都在变点之后
        """
        batch_size, patch_num, feature_dim, patch_len = X.size()
        labels = torch.full((batch_size, patch_num, feature_dim-1), -1, dtype=torch.long)  # 初始化为-1用于检查
        epsilon = 1e5 #这个是为了增大索引方便比较，值的大小是之前缩小索引的反过来的倍数
        # 确保change_point_index的维度正确
        assert change_point_index.size(0) == batch_size, "Batch size of X and change_point_index must match"

        for b in range(batch_size):
            current_cp = change_point_index[b].to(self.device)  # 获取当前batch的变点索引

            for p in range(patch_num):
                #索引的位置在X的第2维度的最后一个特征维，所以这里直接取
                start_index = X[b,p,-1,0]*epsilon
                end_index = X[b,p,-1,-1]*epsilon
                for f in range(feature_dim-1):
                    # 判断patch相对于变点的位置
                    if end_index.item() < current_cp[f].item():
                        # Patch的所有索引都在变点之前
                        labels[b, p, f] = 0
                    elif start_index.item() <= current_cp[f].item() <= end_index.item():
                        # Patch的索引范围包含了变点
                        labels[b, p, f] = 1
                    elif start_index.item() > current_cp[f].item():
                        # Patch的所有索引都在变点之后
                        labels[b, p, f] = 2

        return labels

    def create_labels1(self, X, change_point_index):
        """
        向量化版本的create_labels函数
        """
        batch_size, patch_num, feature_dim, patch_len = X.size()
        epsilon = 1e5

        # 确保change_point_index的维度正确
        assert change_point_index.size(0) == batch_size, "Batch size of X and change_point_index must match"

        # 提取起始和结束索引 (batch_size, patch_num)
        # 使用X的最后一个特征维度的第一个和最后一个元素
        start_indices = X[:, :, -1, 0] * epsilon  # (batch_size, patch_num)
        end_indices = X[:, :, -1, -1] * epsilon  # (batch_size, patch_num)

        # 扩展维度以便广播比较
        start_indices = start_indices.unsqueeze(-1)  # (batch_size, patch_num, 1)
        end_indices = end_indices.unsqueeze(-1)  # (batch_size, patch_num, 1)

        # 扩展change_point_index以便广播比较
        # change_point_index: (batch_size, feature_dim-1) -> (batch_size, 1, feature_dim-1)
        cp_expanded = change_point_index.unsqueeze(1)

        # 使用广播机制同时比较所有元素[8](@ref)
        condition_before = end_indices < cp_expanded  # 全部在变点之前
        condition_contains = (start_indices <= cp_expanded) & (cp_expanded <= end_indices)  # 包含变点
        condition_after = start_indices > cp_expanded  # 全部在变点之后

        # 创建标签张量
        labels = torch.full((batch_size, patch_num, feature_dim - 1), -1, dtype=torch.long, device=self.device)

        # 使用条件赋值
        labels[condition_before] = 0
        labels[condition_contains] = 1
        labels[condition_after] = 2

        return labels


    def training(self):
        start_time = time()
        iter_count = 0
        train_loss = []
        labels_counts_list = []

        self.model.train()
        for i, (batch_x, idx_x, batch_y, change_point) in enumerate(tqdm(self.train_loader)):
            iter_count += 1
            self.model_optim.zero_grad()
            batch_x = batch_x.double().to(self.device)  # [B,window_size,D]
            batch_y = batch_y.double().to(self.device)  # [B,1]

            if self.args.dataset_name != 'XJTU':
                batch_y = batch_y / self.max_life

            #制造labels
            X = batch_x.clone().to(self.device)
            CDI = idx_x.to(X.device).to(torch.float64)
            X = torch.cat((X, CDI), dim=-1)
            X = X.unfold(dimension=1, size=self.args.patch_len, step=self.args.patch_stride)  # [B,n,D,p]
            labels = self.create_labels1(X, change_point.to(self.device)).long().to(self.device) #[batch, patch_num, sensor_num]

            flat_tensor = labels.flatten()
            labels_counts_list.append(torch.bincount(flat_tensor))

            ouputs_classify, outputs_preds = self.model(batch_x,cam=False, mode='train', idx=idx_x)  # [batch, patch_num, sensor_num, num_classes]
            loss_c = self.classify_train_criterion(ouputs_classify.permute(0, 3, 1, 2), labels)
            loss_p = self.pred_loss_criterion(outputs_preds, batch_y)

            loss = loss_p + 0.1 * loss_c #这里可以设置一个随着epoch逐渐放开的机制

            train_loss.append(loss.item())
            loss.backward()
            self.model_optim.step()
            # ------------------------------------------------

        end_time = time()
        epoch_time = end_time - start_time
        train_loss = np.average(train_loss)  # avgerage loss
        return train_loss, epoch_time

    # ---------------------------------- validation function -----------------------------------------
    def validation(self, vali_loader):
        self.model.eval()
        total_loss = []
        all_preds = []  # 存储所有预测结果
        all_labels = []  # 存储所有真实标签

        for i, (batch_x, idx_x, batch_y, change_point) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)  # [B,1]

            if self.args.dataset_name != 'XJTU':
                batch_y = batch_y / self.max_life

            X = batch_x.clone().to(self.device)
            CDI = idx_x.to(X.device).to(torch.float64)
            X = torch.cat((X, CDI), dim=-1)
            X = X.unfold(dimension=1, size=self.args.patch_len, step=self.args.patch_stride)  # [B,n,D,p]
            labels = self.create_labels1(X, change_point.to(self.device)).long().to(self.device) #[batch, patch_num, sensor_num]

            ouputs_classify, outputs_preds = self.model(batch_x,cam=False, mode='train', idx=idx_x)
            loss_c = self.classify_train_criterion(ouputs_classify.permute(0, 3, 1, 2), labels)
            loss_p = self.pred_loss_criterion(outputs_preds, batch_y)

            loss = loss_p

            preds_classify = torch.argmax(ouputs_classify, dim=-1)

            all_preds.append(preds_classify)
            all_labels.append(labels)
            total_loss.append(loss.item())
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        print(calculate_per_class_metrics(all_preds, all_labels, num_classes=3))
        classify_result = calculate_per_class_metrics(all_preds, all_labels, num_classes=3)

        average_vali_loss = np.average(total_loss)

        self.model.train()

        return average_vali_loss, classify_result

    # ----------------------------------- test function ------------------------------------------
    def test(self, test_loader):
        self.model.eval()
        enc_pred = []
        gt = []
        start_time = time()
        for i, (batch_x, idx_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.double().double().to(self.device)
            batch_y = batch_y.double().double().to(self.device)

            if self.args.is_minmax:
                _, outputs = self.model(batch_x,cam=False, mode='test', idx=idx_x, pre_train=False)
                outputs = outputs * self.max_life

            # else:
            #     _, outputs = self.model(batch_x, mode='test', idx=idx_x, pre_train=False)  # outputs[B,window_size]

            batch_y = batch_y.detach().cpu().numpy()
            enc = outputs.detach().cpu().numpy()

            if self.args.dataset_name == 'XJTU' and self.args.is_minmax:
                enc = enc / self.max_life
                batch_y = batch_y / self.max_life

            gt.append(batch_y)
            enc_pred.append(enc)
        end_time = time()
        test_average_time = end_time - start_time
        gt = np.concatenate(gt).reshape(-1, 1)
        enc_pred = np.concatenate(enc_pred).reshape(-1, 1)

        x = np.arange(0, gt.shape[0])

        plt.scatter(x, enc_pred, label='Enc Pred', color='blue')
        plt.scatter(x, gt, label='GT', color='red')

        # 添加标签和标题
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Enc Pred and GT')
        plt.legend()

        fig_path = self.save_path + '/visual_result.png'
        plt.savefig(fig_path)

        # 关闭图像，释放内存
        plt.close()

        if self.args.save_test:
            result_path = self.save_path + '/result.npz'
            np.savez(result_path, test_preds=enc_pred, test_trues=gt, )

        # 算的就是RMSE
        average_enc_loss = np.sqrt(mean_squared_error(enc_pred, gt))
        average_enc_overall_loss = np.sqrt(mean_squared_error(enc_pred, gt))
        # 计算score
        overall_score = self.score_compute(enc_pred, gt)

        return average_enc_loss, average_enc_overall_loss, overall_score, test_average_time

    def score_compute(self, pred, gt):
        # pred [B] gt[B]
        B = pred.shape
        score = 0

        if self.args.dataset_name == 'XJTU':  # 按照这种算法越大越好
            nonzero_idx = np.where(gt != 0.0)[0]
            gt = gt[nonzero_idx]
            gt = gt.reshape(-1, 1)

            pred = pred[nonzero_idx]
            pred = pred.reshape(-1, 1)

            score_list = np.where(pred - gt < 0, np.exp((gt - pred) * 100 * np.log(0.5) / (gt * 20)),
                                  np.exp(-(gt - pred) * 100 * np.log(0.5) / (gt * 5)))
        else:  # 越小越好
            score_list = np.where(pred - gt < 0, np.exp(-(pred - gt) / 13) - 1, np.exp((pred - gt) / 10) - 1)
        # 这里有的paper求均值，有的求和。实验里面先全都求和计算score
        if self.args.dataset_name == 'CMAPSS':
            score = np.sum(score_list)
        else:
            score = np.mean(score_list)
        return score


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.1):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        errors = target - pred
        loss = torch.max(
            (self.quantile - 1) * errors,
            self.quantile * errors
        )
        return torch.mean(loss)



