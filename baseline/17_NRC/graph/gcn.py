"""Pure-PyTorch sparse GCN modules for the graph NRC experiments."""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils import weight_norm


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, features, normalized_adjacency):
        support = torch.mm(features, self.weight)
        output = torch.sparse.mm(normalized_adjacency, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCNEncoder(nn.Module):
    """Two-layer sparse GCN replacing NRC's original image backbone."""

    def __init__(self, input_dim, hidden_dim=128, output_dim=128, dropout=0.5):
        super(GCNEncoder, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = float(dropout)

    def forward(self, features, normalized_adjacency):
        hidden = self.gcn1(features, normalized_adjacency)
        hidden = functional.relu(hidden)
        hidden = functional.dropout(hidden, p=self.dropout, training=self.training)
        output = self.gcn2(hidden, normalized_adjacency)
        return functional.relu(output)


class FeatureBottleneck(nn.Module):
    """Linear plus BatchNorm bottleneck matching the released NRC design."""

    def __init__(self, input_dim, bottleneck_dim=256):
        super(FeatureBottleneck, self).__init__()
        self.linear = nn.Linear(input_dim, bottleneck_dim)
        self.batch_norm = nn.BatchNorm1d(bottleneck_dim, affine=True)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.batch_norm.weight, 1.0, 0.02)
        nn.init.zeros_(self.batch_norm.bias)

    def forward(self, features):
        return self.batch_norm(self.linear(features))


class WeightNormalizedClassifier(nn.Module):
    """NRC classifier; unlike SHOT, it is adapted together with the encoder."""

    def __init__(self, input_dim, class_num=2):
        super(WeightNormalizedClassifier, self).__init__()
        self.linear = weight_norm(nn.Linear(input_dim, class_num), name="weight")
        nn.init.xavier_normal_(self.linear.weight_v)
        nn.init.ones_(self.linear.weight_g)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features):
        return self.linear(features)


def build_gcn_nrc_models(
    input_dim,
    hidden_dim=128,
    encoder_dim=128,
    bottleneck_dim=256,
    class_num=2,
    dropout=0.5,
):
    encoder = GCNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=encoder_dim,
        dropout=dropout,
    )
    bottleneck = FeatureBottleneck(
        input_dim=encoder_dim, bottleneck_dim=bottleneck_dim
    )
    classifier = WeightNormalizedClassifier(
        input_dim=bottleneck_dim, class_num=class_num
    )
    return encoder, bottleneck, classifier


def forward_model(encoder, bottleneck, classifier, features, adjacency):
    encoded = encoder(features, adjacency)
    bottleneck_features = bottleneck(encoded)
    logits = classifier(bottleneck_features)
    return bottleneck_features, logits
