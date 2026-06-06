# coding=utf-8
"""
GRADE on Pokec: pokec_z (source) -> pokec_n (target)

数据加载完全遵照 code/dataset.py 中的 load_pokec / feature_norm / index_to_mask。
算法采用 GRADE（GCN 编码 + JS/MMD/C 域判别器 + GRL 对抗训练），
与原始 GRADE.py / GRADE_train.py 保持一致。
适配工作：将 PyG edge_index 转换为 DGL 图，供 GRADE 的 dgl.GraphConv 使用；
         以 source train_mask 作为有监督训练节点，target test_mask 作为评估节点。
"""

from __future__ import division
from __future__ import print_function

import os
import random
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import dgl
from dgl.nn.pytorch import GraphConv
# 下面所有dgl.xx都有改
# import dgl-cu113
# from dgl-cu113.nn.pytorch import GraphConv

from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, "/home/disk2/lhr/sfda/code")
from utils import fair_metric
# Optional: remove it right after so it doesn't mess with other local imports later
sys.path.pop(0)

# =============================================================================
# 0.  命令行参数（与原始 main.py 风格保持一致）
# =============================================================================
parser = argparse.ArgumentParser(description="GRADE on Pokec")
parser.add_argument('--n_hidden',      type=int,   default=64)
parser.add_argument('--n_layers',      type=int,   default=2)
parser.add_argument('--epochs',        type=int,   default=200)
parser.add_argument('--lr',            type=float, default=0.05)
parser.add_argument('--weight_decay',  type=float, default=1e-4)
parser.add_argument('--dropout',       type=float, default=0.0)
parser.add_argument('--disc',          type=str,   default='JS',
                    choices=['JS', 'MMD', 'C'])
parser.add_argument('--seed',          type=int,   default=42)
parser.add_argument('--cuda',          type=int,   default=0)
args = parser.parse_args()

# =============================================================================
# 1.  随机种子
# =============================================================================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu"
)

# =============================================================================
# 2.  数据加载（完全照搬 code/dataset.py 中的实现，路径指向 Benchmark 数据集）
# =============================================================================
DATASET_ROOT = "/home/disk2/lhr/sfda/fairDomainAdaption/Benchmark-GraphFairness/dataset"


def feature_norm(features):
    """Min-Max 归一化到 [-1, 1]，与 code/dataset.py 完全一致。"""
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def index_to_mask(node_num, index):
    """与 code/dataset.py 完全一致。"""
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1
    return mask


def load_pokec(
    dataset,
    id,
    sens_attr="region",
    predict_attr="I_am_working_in_field",
    path=DATASET_ROOT,
):
    """
    与 code/dataset.py:load_pokec 逻辑完全一致。

    Benchmark 目录结构：
        dataset/pokec_z/pokec_z.csv
        dataset/pokec_z/pokec_z_edges.txt
        dataset/pokec_n/pokec_n.csv
        dataset/pokec_n/pokec_n_edges.txt

    调用：
        load_pokec(dataset="pokec", id="_z")   # source: pokec_z
        load_pokec(dataset="pokec", id="_n")   # target: pokec_n
    """
    full_name = dataset + id                        # e.g. "pokec_z"
    csv_path  = os.path.join(path, full_name, full_name + ".csv")
    idx_features_labels = pd.read_csv(csv_path)

    # 取 pokec_z 与 pokec_n 列的交集作为特征列（保证 source/target 特征对齐）
    header  = list(pd.read_csv(os.path.join(path, "pokec_z", "pokec_z.csv")).columns)
    header2 = list(pd.read_csv(os.path.join(path, "pokec_n", "pokec_n.csv")).columns)
    header  = [i for i in header if i in header2]
    header.remove("user_id")
    header.remove(sens_attr)
    header.remove(predict_attr)

    features = idx_features_labels[header]
    features = torch.FloatTensor(np.array(features, dtype=np.float32))

    labels = idx_features_labels[predict_attr].values
    labels = torch.LongTensor(labels)
    labels[labels > 1] = 1          # 与 code/dataset.py 相同的二值化处理

    sens_labels = idx_features_labels[sens_attr].values.astype(int)
    sens_labels = torch.FloatTensor(sens_labels)

    # ── 建图（与 code/dataset.py 完全一致）──────────────────────────────────
    idx     = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        os.path.join(path, full_name, full_name + "_edges.txt"), dtype=int
    )
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    edge_index, _ = from_scipy_sparse_matrix(adj)

    # ── train / val / test 划分（与 code/dataset.py 完全一致）───────────────
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[: int(0.8 * len(label_idx_0))],
        label_idx_1[: int(0.8 * len(label_idx_1))],
    )
    idx_val = np.append(
        label_idx_0[int(0.8 * len(label_idx_0)) : int(0.9 * len(label_idx_0))],
        label_idx_1[int(0.8 * len(label_idx_1)) : int(0.9 * len(label_idx_1))],
    )
    idx_test = np.append(
        label_idx_0[int(0.9 * len(label_idx_0)) :],
        label_idx_1[int(0.9 * len(label_idx_1)) :],
    )
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask   = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask  = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask


# =============================================================================
# 3.  适配层：PyG edge_index -> DGL 图
# =============================================================================

def edge_index_to_dgl(edge_index, num_nodes):
    """
    将 PyG 格式的 edge_index (2 x E LongTensor) 转换为 DGL 同构图。
    GRADE 的 dgl.nn.GraphConv 需要 DGLGraph 作为输入。
    edge_index 已包含自环（load_pokec 中 adj + sp.eye），无需再次添加。
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    # g = dgl-cu113.graph((src, dst), num_nodes=num_nodes)
    return g


# =============================================================================
# 4.  加载数据集
# =============================================================================

print("Loading source dataset (pokec_z) ...")
src_ei, src_x, src_y, src_sens, src_train, src_val, src_test = load_pokec(
    dataset="pokec", id="_z"
)
src_x = feature_norm(src_x)

print("Loading target dataset (pokec_n) ...")
tgt_ei, tgt_x, tgt_y, tgt_sens, tgt_train, tgt_val, tgt_test = load_pokec(
    dataset="pokec", id="_n"
)
tgt_x = feature_norm(tgt_x)

# 构建 DGL 图（供 GRADE GraphConv 使用）
src_g = edge_index_to_dgl(src_ei, src_x.shape[0])
tgt_g = edge_index_to_dgl(tgt_ei, tgt_x.shape[0])

num_features = src_x.shape[1]
num_classes  = 2   # pokec 二分类（label ∈ {0, 1}）

print(f"Source : {src_x.shape[0]:>6d} nodes | {num_features} features | "
      f"{src_ei.shape[1]} edges")
print(f"Target : {tgt_x.shape[0]:>6d} nodes | {num_features} features | "
      f"{tgt_ei.shape[1]} edges")
print(f"Classes: {num_classes}  Disc: {args.disc}")

# 将图和特征迁移到 device
src_g = src_g.to(device)
tgt_g = tgt_g.to(device)
src_x = src_x.to(device)
src_y = src_y.to(device)
tgt_x = tgt_x.to(device)
tgt_y = tgt_y.to(device)
src_train = src_train.to(device)
src_test  = src_test.to(device)
tgt_test  = tgt_test.to(device)

# =============================================================================
# 5.  GRADE 模型（与原始 GRADE.py 完全一致）
# =============================================================================

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp)
        for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma
    )
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

"""
class GRADE(nn.Module):
    与 GRADE/cross-network node classification/GRADE.py 完全一致。

    def __init__(self, g_s, g_t, in_feats, n_hidden, n_classes,
                 n_layers, dropout, activation=F.relu, disc="JS"):
        super(GRADE, self).__init__()
        self.disc = disc
        self.g_s = g_s
        self.g_t = g_t
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)

        if disc == "JS":
            self.discriminator = nn.Sequential(
                nn.Linear(n_hidden * n_layers + n_classes, 2)
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(n_hidden * n_layers + n_classes * 2, 2)
            )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features_s, labels_s, features_t, alpha=1.0):
        
        features_s: 全图 source 特征（所有节点），labels_s 只用 train_mask 节点监督。
        适配说明：原始 GRADE 以全图节点做分类损失；
                  这里改为只对 source 的 train_mask 节点计算分类损失，
                  域对齐仍使用全部节点的表征，与域自适应惯例一致。
        
        s_f = []
        t_f = []
        for i, layer in enumerate(self.layers):
            features_s = self.dropout(features_s)
            features_t = self.dropout(features_t)
            features_s = layer(self.g_s, features_s)
            features_t = layer(self.g_t, features_t)
            s_f.append(features_s)
            t_f.append(features_t)

        logits_s = self.fc(features_s)
        logits_t = self.fc(features_t)
        s_f.append(logits_s)
        t_f.append(logits_t)

        # ── 分类损失：仅对 source train_mask 节点 ────────────────────────────
        preds_s   = torch.log_softmax(logits_s, dim=-1)
        class_loss = F.nll_loss(preds_s[labels_s["mask"]], labels_s["labels"])

        s_f_cat = torch.cat(s_f, dim=1)   # (N_s, n_hidden*n_layers + n_classes)
        t_f_cat = torch.cat(t_f, dim=1)   # (N_t, n_hidden*n_layers + n_classes)

        domain_loss = 0.0
        if self.disc == "JS":
            domain_preds = self.discriminator(
                ReverseLayerF.apply(
                    torch.cat([s_f_cat, t_f_cat], dim=0), alpha
                )
            )
            domain_labels = np.array(
                [0] * s_f_cat.shape[0] + [1] * 
"""

class GRADE(nn.Module):
    def __init__(self, g_s, g_t, in_feats, n_hidden, n_classes, n_layers, dropout, activation=F.relu, disc="JS"):
        super(GRADE, self).__init__()
        self.disc = disc
        self.g_s = g_s
        self.g_t = g_t
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # self.layers.append(GraphConv(n_hidden, n_classes, activation=None))
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)

        if disc == "JS":
            self.discriminator = nn.Sequential(
                nn.Linear(n_hidden*n_layers+n_classes, 2)
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(n_hidden * n_layers + n_classes * 2, 2)
            )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features_s, labels_s, features_t, alpha=1.0):
        s_f = []
        t_f = []
        for i, layer in enumerate(self.layers):
            features_s = self.dropout(features_s)
            features_t = self.dropout(features_t)
            features_s = layer(self.g_s, features_s)
            features_t = layer(self.g_t, features_t)
            s_f.append(features_s)
            t_f.append(features_t)
        features_s = self.fc(features_s)
        features_t = self.fc(features_t)
        s_f.append(features_s)
        t_f.append(features_t)
        preds_s = torch.log_softmax(features_s, dim=-1)
        class_loss = F.nll_loss(preds_s, labels_s)

        s_f = torch.cat(s_f, dim=1)
        t_f = torch.cat(t_f, dim=1)
        domain_loss = 0.
        if self.disc == "JS":
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_f, t_f], dim=0), alpha))
            domain_labels = np.array([0] * features_s.shape[0] + [1] * features_t.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=features_s.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
        elif self.disc == "MMD":
            mind = min(s_f.shape[0], t_f.shape[0])
            domain_loss = mmd_rbf_noaccelerate(s_f[:mind], t_f[:mind])
        elif self.disc == "C":
            ratio = 8
            s_l_f = torch.cat([s_f, ratio * self.one_hot_embedding(labels_s)], dim=1)
            t_l_f = torch.cat([t_f, ratio * F.softmax(features_t, dim=1)], dim=1)
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_l_f, t_l_f], dim=0), alpha))
            domain_labels = np.array([0] * features_s.shape[0] + [1] * features_t.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=features_s.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
        loss = class_loss + domain_loss * 0.01
        return loss

    def inference(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.g_t, x)
        x = self.fc(x)
        return torch.log_softmax(x, dim=-1)

    def one_hot_embedding(self, labels):
        y = torch.eye(self.n_classes, device=labels.device)
        return y[labels]

# =============================================================================
# 6.  训练和评估
# =============================================================================
def test(model, features_t, labels_t, sens_t, mask_t):
    model.eval()
    with torch.no_grad():
        logits = model.inference(features_t)
        predictions = logits.argmax(dim=1)

        y_true = labels_t[mask_t].cpu().numpy()
        y_pred = predictions[mask_t].cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)

        probs = torch.softmax(logits[mask_t], dim=1)[:, 1].cpu().numpy()
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = float('nan')

        sens = sens_t[mask_t].cpu().numpy().astype(int)
        delta_dp, delta_eo = fair_metric(y_pred, y_true, sens)

    return accuracy, auc, delta_dp, delta_eo

def train():
    # 实例化原版的 GRADE
    model = GRADE(src_g, tgt_g, num_features, args.n_hidden, num_classes, args.n_layers, args.dropout, disc=args.disc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 克隆一份源域标签，把不在 src_train (训练集) 里的节点标签设为 -100
    # PyTorch 的 F.nll_loss 默认 ignore_index=-100，会自动忽略这些节点，防止数据泄露
    masked_src_y = src_y.clone()
    masked_src_y[~src_train] = -100

    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        alpha = 2 / (1 + np.exp(- 10 * epoch / args.epochs)) - 1
        
        # 按原版 GRADE 的要求：只传 4 个参数，并且只接收 1 个返回的 loss
        loss = model(src_x, masked_src_y, tgt_x, alpha)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            acc, auc, _, _ = test(model, tgt_x, tgt_y, tgt_sens, tgt_test)
            # 因为原版没返回具体的 cls_loss 和 dom_loss，打印时省去它们
            print(f"Epoch [{epoch+1}/{args.epochs}]  "
                  f"Loss: {loss.item():.4f}  "
                  f"TgtTestAcc: {acc:.4f}")

    return model

# 执行训练与结果输出
trained_model = train()
acc, auc, dp, eo = test(trained_model, tgt_x, tgt_y, tgt_sens, tgt_test)

print(f"\n=== Final Results (pokec_z -> pokec_n) ===")
print(f"Target Domain ACC         : {acc:.4f}")
print(f"Target Domain ROC-AUC     : {auc:.4f}")
print(f"Target Domain delta-DP    : {dp:.4f}")
print(f"Target Domain delta-EO    : {eo:.4f}")