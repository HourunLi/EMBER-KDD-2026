import torch
import torch.nn as nn
import torch.nn.init as init

class Classifier(torch.nn.Module):
    def __init__(self, args, input_dim, num_cls):
        super(Classifier, self).__init__()
        self.body = nn.Linear(input_dim, num_cls)
        
    def forward(self, feat, edge_index = None):
        return self.body(feat)