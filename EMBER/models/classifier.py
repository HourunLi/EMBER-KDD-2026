from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_dim, num_cls):
        super().__init__()
        self.body = nn.Linear(input_dim, num_cls)

    def forward(self, feat):
        return self.body(feat)
