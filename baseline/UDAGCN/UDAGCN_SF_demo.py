# coding=utf-8
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import itertools


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='dblp')
parser.add_argument("--name", type=str, default='UDAGCN_SF_demo')
parser.add_argument("--seed", type=int, default=200)
parser.add_argument("--UDAGCN", type=bool, default=True)
parser.add_argument("--encoder_dim", type=int, default=16)
parser.add_argument("--data_root", type=str, default='data')


args = parser.parse_args()
seed = args.seed
use_UDAGCN = args.UDAGCN
encoder_dim = args.encoder_dim


id = "source: {}, target: {}, seed: {}, name: {}, UDAGCN: {}, encoder_dim: {}".format(
    args.source, args.target, seed, args.name, use_UDAGCN, encoder_dim
)

print(id)


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = args.data_root
if not os.path.isabs(data_root):
    data_root = os.path.join(script_dir, data_root)


def load_domain_data(domain_name):
    dataset = DomainData(os.path.join(data_root, domain_name), name=domain_name)
    data = dataset[0]
    print(data)
    return dataset, data


# 阶段一只加载源域数据并进行源域监督预训练；目标域数据不参与预训练。
source_dataset, source_data = load_domain_data(args.source)
num_features = source_dataset.num_features
num_classes = source_dataset.num_classes
source_data = source_data.to(device)


class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv

        self.conv_layers = nn.ModuleList([
            model_cls(num_features, 128,
                     weight=weights[0],
                     bias=biases[0],
                     **kwargs),
            model_cls(128, encoder_dim,
                     weight=weights[1],
                     bias=biases[1],
                     **kwargs)
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


loss_func = nn.CrossEntropyLoss().to(device)

encoder = GNN(type="gcn").to(device)
if use_UDAGCN:
    ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=10).to(device)


cls_model = nn.Sequential(
    nn.Linear(encoder_dim, num_classes),
).to(device)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


if use_UDAGCN:
    att_model = Attention(encoder_dim).to(device)

models = [encoder, cls_model]
if use_UDAGCN:
    models.extend([ppmi_encoder, att_model])


def build_optimizer():
    # 两阶段均沿用原始 UDAGCN 的 Adam 学习率设置。
    params = itertools.chain(*[model.parameters() for model in models])
    return torch.optim.Adam(params, lr=3e-3)


optimizer = build_optimizer()


def gcn_encode(data, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def ppmi_encode(data, cache_name, mask=None):
    encoded_output = ppmi_encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def encode(data, cache_name, mask=None):
    gcn_output = gcn_encode(data, cache_name, mask)
    if use_UDAGCN:
        ppmi_output = ppmi_encode(data, cache_name, mask)
        outputs = att_model([gcn_output, ppmi_output])
        return outputs
    else:
        return gcn_output


def predict(data, cache_name, mask=None):
    encoded_output = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    with torch.no_grad():
        logits = predict(data, cache_name, mask)
        preds = logits.argmax(dim=1)
        labels = data.y if mask is None else data.y[mask]
        accuracy = evaluate(preds, labels)
    return accuracy


def clear_all_encoder_caches():
    # 进入 SFDA 目标迁移阶段前清理源图传播缓存，阶段二只保留源域预训练得到的模型参数。
    for model in models:
        for module in model.modules():
            if hasattr(module, "cache_dict"):
                module.cache_dict.clear()


epochs = 200


def add_original_weight_regularization(loss):
    # 沿用原始 UDAGCN_demo.py 中对所有 weight 参数加入 param.mean() * 3e-3 的实现。
    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                loss = loss + param.mean() * 3e-3
    return loss


def pretrain_source(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    encoded_source = encode(source_data, "source")
    source_logits = cls_model(encoded_source)

    # 阶段一：源域预训练。保留原始 demo 的源域分类监督损失，不使用目标域数据。
    cls_loss = loss_func(source_logits, source_data.y)
    loss = add_original_weight_regularization(cls_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def adapt_target(epoch, target_data):
    for model in models:
        model.train()
    optimizer.zero_grad()

    encoded_target = encode(target_data, "target")

    # 阶段二：SFDA 目标域迁移。移除硬依赖源/目标同时输入的 L_DA，仅保留原目标熵损失 L_T。
    target_logits = cls_model(encoded_target)
    target_probs = F.softmax(target_logits, dim=-1)
    target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
    loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))
    loss = loss_entropy * (epoch / epochs * 0.01)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print("=============================================================")
print("Stage 1: source pretraining")

best_source_acc = 0.0
best_source_epoch = 0
for epoch in range(1, epochs):
    pretrain_source(epoch)
    source_correct = test(source_data, "source", source_data.test_mask)
    print("[Stage 1] Epoch: {}, source_acc: {}".format(epoch, source_correct))
    if source_correct > best_source_acc:
        best_source_acc = source_correct
        best_source_epoch = epoch

clear_all_encoder_caches()

# 源域预训练结束后释放源域数据，确保目标迁移阶段不再访问源域样本或源图缓存。
del source_data
del source_dataset
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


print("=============================================================")
print("Stage 2: target adaptation without source data")

target_dataset, target_data = load_domain_data(args.target)
if target_dataset.num_features != num_features:
    raise ValueError("source and target num_features mismatch: {} vs {}".format(
        num_features, target_dataset.num_features
    ))
if target_dataset.num_classes != num_classes:
    raise ValueError("source and target num_classes mismatch: {} vs {}".format(
        num_classes, target_dataset.num_classes
    ))
target_data = target_data.to(device)

# 目标迁移阶段重新构造优化器，沿用原始学习率，同时避免携带源域优化器动量状态。
optimizer = build_optimizer()

best_target_acc = 0.0
best_target_epoch = 0
for epoch in range(1, epochs):
    adapt_target(epoch, target_data)
    target_correct = test(target_data, "target")
    print("[Stage 2] Epoch: {}, target_acc: {}".format(epoch, target_correct))
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_target_epoch = epoch

print("=============================================================")
line = "{} - best_source_epoch: {}, best_source_acc: {}, best_target_epoch: {}, best_target_acc: {}".format(
    id, best_source_epoch, best_source_acc, best_target_epoch, best_target_acc
)

print(line)
