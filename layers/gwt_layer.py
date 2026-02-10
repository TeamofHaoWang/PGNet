import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import seaborn as sns

"""Based on Version-9.11"""


class GWTNet_layer(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim,d_model,d_ff,
                 dropout=0.3, kernel_size=2, blocks=3, layers=1,args=None,**kwargs):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.args=args

        residual_channels = d_model
        dilation_channels = d_model
        skip_channels = d_ff
        end_channels = d_ff

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))

        depth = list(range(blocks * layers))

        receptive_field = 1

        self.supports_len = 1

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv2d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = ModuleList([Conv2d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList([GraphConvNet(dilation_channels, residual_channels, dropout, support_len=self.supports_len)
                                              for _ in depth])

        self.diff_index_graph_norm = ModuleList([nn.BatchNorm1d(residual_channels) for _ in depth])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()

        self.output_bn = BatchNorm2d(d_model)
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1 # dilation
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field

    def forward(self, x,adj,**kargs):
        # adj = [adj_dynamic_with_explicit,adj_dynamic_with_implicit]
        mode = kargs.get('mode')
        x = self.start_conv(x)

        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # EACH BLOCK
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |   |-dil_conv -- tanh --|                |
            #         ---|                  * ----|-- 1x1 -- + -->	*x_in*
            #                |-dil_conv -- sigm --|    |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            # parametrized skip connection
            s = self.skip_convs[i](x)

            # what are we skipping??
            try:  # if i > 0 this works
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            x = self.graph_convs[i](x, adj)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = self.output_bn(skip)  # ignore last X?
        B, _, N, _ = x.shape
        x = x.transpose(1,-1)
        return x


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""

    if len(A.shape) == 2:
        return torch.einsum('ncvl,vw->ncwl', (x, A.to(x.device))).contiguous()

    if len(A.shape) == 3:
        return torch.einsum('ncvl,nvw->ncwl', (x, A.to(x.device))).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super().__init__()
        c_in =  (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list ):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = self.bn(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

