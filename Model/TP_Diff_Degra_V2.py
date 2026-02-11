import torch
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from layers.FCSTGNN_base import *
from layers.Diff_Graph_Net_base_1 import TCN_base
from torch.nn.parameter import Parameter
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import seaborn as sns
import os
from sklearn.manifold import TSNE
import torch.nn.functional as F


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('bndh,bndw->bnwh', (x, A.to(x.device))).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = nn.Conv2d(c_in, c_out, 1, padding=0, stride=1, bias=True)
        self.dropout = dropout
        self.order = order
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=-1)
        h = self.final_conv(h.transpose(-1, 1))
        h = self.bn(h).transpose(-1, 1)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class PatchClassifier(nn.Module):
    def __init__(self, input_features=128, num_sensors=15, num_classes=3):
        super(PatchClassifier, self).__init__()
        self.num_sensors = num_sensors

        # 使用1D卷积处理传感器序列数据
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.num_classes = num_classes
        # 一个轻量的分类头，最终输出维度为num_classes
        # sensor维度用卷积将其减到14
        self.reduce_sensor_conv = nn.Conv1d(in_channels=32, out_channels=num_classes, kernel_size=2, stride=1)


    def forward(self, x):
        batch_size, patch_num, sensor_num, feature_dim = x.size()

        # 合并batch和patch维度
        x = x.contiguous().view(batch_size * patch_num, sensor_num, feature_dim).transpose(1, 2)

        # 应用1D卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 使用一个kernel_size=2的卷积，将序列长度从15减到14
        x = self.reduce_sensor_conv(x)
        x = x.transpose(1, 2)
        x = F.softmax(x,dim=-1)
        # 然后重塑回(batch_size, patch_num, sensor_num-1, num_classes)
        x = x.view(batch_size, patch_num, self.num_sensors - 1, self.num_classes)
        return x

class TP_Diff_Degra_V2(nn.Module):
    def __init__(self, args,save_path):
        super(TP_Diff_Degra_V2, self).__init__()
        # graph_construction_type = args.graph_construction_type
        self.patch_len = args.patch_size
        self.d_ff = args.d_ff
        self.d_model = args.d_model
        self.x_nonlin_map = nn.Sequential(
            nn.Linear(self.patch_len, self.d_ff),
            nn.Linear(self.d_ff, self.d_model),
            nn.BatchNorm1d(self.d_model)
        )
        self.n_sensors = args.input_feature
        self.idx_n_sensors = self.n_sensors + 1

        self.TCN_Gate = TCN_base(self.idx_n_sensors, self.d_model, 2 * self.d_model, 2 * self.d_model,
                                 2 * self.d_model)  # block=3, layers=1, kernel_size=2
        self.patch_classify = PatchClassifier(input_features=2*self.d_model)

        self.patch_len = args.patch_size
        self.patch_stride = args.patch_size

        num_patches = (args.input_length - self.patch_len) // self.patch_stride + 1

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2 * self.d_model * num_patches * self.idx_n_sensors, 2 * self.d_model)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2 * self.d_model, 2 * self.d_model)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2 * self.d_model, self.d_model)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(self.d_model, 1)),

        ]))

        self.rw_fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.idx_n_sensors * self.idx_n_sensors, 2 * self.d_model)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2 * self.d_model, self.d_model)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(self.d_model, 1)),

        ]))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # related to GCN
        init_graph = torch.randn(self.idx_n_sensors, self.idx_n_sensors)
        self.adp_graph = nn.Parameter(init_graph, requires_grad=True)

        # TODO rw
        self.bn = nn.BatchNorm1d(self.idx_n_sensors * self.idx_n_sensors)
        self.flag = True
        self.save_path = save_path

        # TODO 动态图
        self.xlstm=XLSTM_dynamic_graph(in_feature=2*self.d_model,d_model=self.d_model,save_path=save_path,num_nodes=self.idx_n_sensors)

    def len_func(self, input_len, patch_size, blocks=3, layers=1, kernel_size=2, stride=1, dilation=2, pad=0):
        n_patch = int(input_len / patch_size)
        for b in range(blocks):
            now_kernel = kernel_size
            D = 1  # dilation
            for i in range(layers):
                n_patch -= 1
                n_patch = int((n_patch + 2 * pad - now_kernel) / stride) + 1
                D *= dilation
                now_kernel = D * (now_kernel - 1) + 1
        return n_patch

    def forward(self, X, cam=False,**kwargs):
        # print(X.size())  [B,L,D]
        mode = kwargs['mode']
        idxs = kwargs['idx']

        CDI = idxs.to(X.device).to(torch.float64)
        X = torch.cat((X, CDI), dim=-1)

        X = X.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)  # [B,n,D,p]
        bs, tlen, num_node, dimension = X.size()

        ### input_X map
        input_X_ = tr.reshape(X, [bs * tlen * num_node, dimension])
        input_X_ = self.x_nonlin_map(input_X_)
        input_X_ = tr.reshape(input_X_, [bs, tlen, num_node, -1])

        TCN_input = tr.transpose(input_X_, 1, 3)  # B, C, N, L
        A_input_ = self.TCN_Gate(TCN_input, [self.adp_graph])
        b, n_p, n_s, _ = A_input_.shape

        output_classify = self.patch_classify(A_input_)

        if cam:
            return A_input_, output_classify

        # TODO DegraNet Graphs compare 串行
        sensors_feature = A_input_.clone()

        # TODO 动态图
        init_graph = torch.eye(self.idx_n_sensors).unsqueeze(0).repeat(b, 1, 1).to(sensors_feature.device)
        adj_dynamic = self.xlstm(sensors_feature,labels=output_classify,cell_past=init_graph,mode=kwargs.get('mode'))  # 为了避免衰减太快，这里每次传入的还是初始的图
        output = self.rw_fc(adj_dynamic.reshape(b,n_s*n_s))
        output = torch.clamp(output, max=1.0)  # 将 output 中大于 1 的值设为 1

        return output_classify, output


def custom_kron_batch(A, B):
    # 获取批量大小和矩阵 A、B 的形状
    batch_size, m, n = A.shape
    _, p, q = B.shape
    # 展平 A 的每个矩阵
    A_flattened = A.view(batch_size, -1)
    # 拓展 A 到 [batch_size, m*n, 1, 1] 形状
    A_expanded = A_flattened.unsqueeze(-1).unsqueeze(-1)
    # 拓展 B 到 [batch_size, 1, p, q] 形状
    B_expanded = B.unsqueeze(1)
    # A 和 B 相乘
    product = A_expanded * B_expanded
    # 调整维度顺序
    product = product.view(batch_size, m, n, p, q)
    product = product.permute(0, 1, 3, 2, 4)
    # 重塑形状
    final_result = product.reshape(batch_size, m * p, n * q)
    return final_result

def visualize_and_save(tensor_data, save_path, batch_index=0):
    # (B,D)
    B, D = tensor_data.shape
    data_for_tsne = tensor_data
    # 执行 T-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=12)
    data_tsne = tsne.fit_transform(data_for_tsne)

    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], marker='o')
    plt.title('T-SNE Visualization of Rul_graphs')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.grid()
    plt.savefig('T_SNE_Visualization.png')


'''利用XLSTM的思想生成动态图'''
class XLSTM_dynamic_graph(nn.Module):
    def __init__(self,in_feature,d_model,save_path,num_nodes,**kwargs):
        super().__init__()
        self.patch_len = 3
        self.stride = 3
        self.num_nodes = num_nodes

        self.weight_pool_k = nn.Parameter(torch.FloatTensor(num_nodes, in_feature, d_model),requires_grad=True)
        self.weight_pool_v = nn.Parameter(torch.FloatTensor(num_nodes, in_feature, d_model),requires_grad=True)

        self.bias_pool_k = nn.Parameter(torch.FloatTensor(num_nodes, d_model),requires_grad=True)
        self.bias_pool_v = nn.Parameter(torch.FloatTensor(num_nodes, d_model),requires_grad=True)

        nn.init.xavier_normal_(self.weight_pool_k)
        nn.init.xavier_normal_(self.weight_pool_v)

        nn.init.xavier_normal_(self.bias_pool_k)
        nn.init.xavier_normal_(self.bias_pool_v)

        self.input=nn.Linear(in_feature,1)
        self.forget = nn.Linear(in_feature, 1)
        # 以下是两个状态空间

        #FIXME 测试softplus约束输入门的输入
        self.q_k_activation = nn.Softplus()
        # self.key = nn.Sequential(
        #     nn.Linear(d_model,d_model),
        #     Transpose(1, 2),
        #     nn.BatchNorm1d(d_model),
        #     Transpose(1, 2),
        #     nn.Dropout(0.1)
        # )


        # self.value = nn.Sequential(
        #     Transpose(1, 2),
        #     Conv1d(in_feature, d_model, 1, padding= 0, stride=1, bias=True, groups=in_feature),
        #     nn.BatchNorm1d(d_model),
        #     Transpose(1, 2),
        #     nn.Dropout(0.1)
        # )

        # weight pool 的后续层
        # self.v_b = nn.BatchNorm1d(d_model)
        self.v_ln =nn.LayerNorm(d_model,elementwise_affine=True)
        self.v_d = nn.Dropout(0.1)

        # self.k_b = nn.BatchNorm1d(d_model)
        self.k_ln = nn.LayerNorm(d_model,elementwise_affine=True)
        self.k_d = nn.Dropout(0.1)
        self.now_ln = nn.LayerNorm(num_nodes,elementwise_affine=True)

        self.save_path=save_path
        self.flag=0
        self.visual_flag = True

        self.padding_patch = 'end'

        if self.padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

    def visual_cell(self,A, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tmp = A[0,:,:].clone()
        plt.figure()
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm',vmin=0,vmax=1)
        plt.title('cell')
        # plt.legend()
        plt.savefig(os.path.join(save_path, f'cell_{self.flag}'))
        plt.close()
        return

    def forward(self,x,labels=None,cell_past=None,normalize_past=None,m_past=None,**kwargs):
        mode=kwargs.get('mode')
        B,C,N,L=x.shape
        h_list=[]
        if cell_past!=None:
            cell_list = [cell_past]
        else:
            cell_list=[torch.zeros((B,N,N),device=x.device)]
        if normalize_past!=None:
            normalize_list=[normalize_past]
        else:
            normalize_list=[torch.zeros((B,N,1),device=x.device)]
        if m_past!=None:
            m_list=[m_past]
        else:
            m_list=[torch.zeros((B,N,1),device=x.device)]

        # 1. 对第一个维度作一阶差分
        diff_labels = labels[:,1:] - labels[:,:-1]

        # 2. 在差分后的序列开头插入全0的tensor
        zeros_tensor = torch.zeros(labels.size(0), 1, labels.size(2)).to(labels.device)  # 插入的全0 tensor

        # 3. 计算差分序列最后一个维度的L2 norm
        diff_labels_norm = torch.norm(diff_labels, p=2, dim=-1)

        # 4. 将 norm 映射到 [-1, 1] 范围内
        min_norm = diff_labels_norm.min()
        max_norm = diff_labels_norm.max()

        # 归一化到 [0, 1] 范围
        normalized_diff_labels = (diff_labels_norm - min_norm) / (max_norm - min_norm)

        # 映射到 [-1, 1] 范围
        mapped_diff_labels = normalized_diff_labels * 2 - 1

        mapped_diff_labels = torch.cat([zeros_tensor, mapped_diff_labels], dim=1)

        for i in range(x.shape[1]):
            h,cell=self.step(x[:,i],labels=mapped_diff_labels[:,i],cell_past=cell_list[-1])
            h_list.append(h);cell_list.append(cell)
            if mode == 'test' and self.visual_flag:
                self.visual_cell(h,save_path=self.save_path)
                self.flag+=1
        self.visual_flag = False
        return h_list[-1]

    def step(self,xt,labels,cell_past):
        '''xt (B,N,C);cell_past(B,N,N),normalize_past(B,N,1)'''
        a = self.input(xt)
        a[:, :-1, :] = a[:, :-1, :] + labels.unsqueeze(-1) * 0.1  # 使用显式的加法
        I_gate = torch.sigmoid(a)#(B,N,1)
        F_gate = 1 - I_gate# (B,N,1)

        # m_now=torch.max(torch.log(f_gate)+m_past,torch.log(i_gate)) # 逐元素来比较
        # i_gate=torch.exp(torch.log(i_gate)-m_now)
        # f_gate = torch.exp(torch.log(f_gate)- m_now+m_now)
        B,N,C=xt.shape


        # Linear，Conv映射方法
        # ----------------------------------------------
        # key = self.q_k_activation(self.key(xt))
        # key=key/math.sqrt(key.shape[-1])
        #
        # value = self.q_k_activation(self.value(xt))
        # value=value/math.sqrt(value.shape[-1])# (B,N,H)  # 一般不用，数值会过小
        #------------------------------------------------

        # weight pool 映射
        # ----------------------------------------------
        key = torch.einsum('bnd,ndo->bno', xt, self.weight_pool_k) + self.bias_pool_k
        key = self.q_k_activation(self.k_d(self.k_ln(key)))
        key = key / math.sqrt(key.shape[-1])

        value = torch.einsum('bnd,ndo->bno', xt, self.weight_pool_v) + self.bias_pool_v
        value = self.q_k_activation(self.v_d(self.v_ln(value)))
        # ----------------------------------------------
        now = F.relu(self.now_ln(torch.matmul(key,value.transpose(-1,-2))))
        cell=torch.multiply(F_gate,cell_past)+torch.multiply(I_gate,now)#(B,N,N)
        h=cell

        return h,cell




















