
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from torch import nn
import torch.nn.init as init

class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        
        if args.activation == "ident":
            self.activate = lambda x: x
        elif args.activation == "sigmoid":
            self.activate = nn.Sigmoid()
        elif args.activation == "LeakyReLU":
            self.activate = nn.LeakyReLU()
            
        self.lin = nn.Sequential(Linear(args.num_features, args.hidden),
                                 nn.Dropout(p=args.dropout),
                                 self.activate,
                                 Linear(args.hidden, args.feature_dim),
                                 nn.Dropout(p=args.dropout),
                                 self.activate)
    def reset_parameters(self):
        for layer in self.lin:
            if hasattr(layer, 'weight') and layer.weight is not None:
                init.xavier_uniform_(layer.weight)  # Xavier 初始化权重

            if hasattr(layer, 'bias') and layer.bias is not None:
                init.zeros_(layer.bias)  # 偏置设置为 0
        
        
    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def forward(self, x, edge_index=None):
        feat = self.lin(x)
        feat = F.normalize(feat, dim=1, p=2)
        return feat

def create_mlp(args):
    model = MLP(args)
    return model
