# coding=utf-8
"""
UDAGCN on Pokec: pokec_z (source) -> pokec_n (target)

数据加载完全遵照 code/dataset.py 中的 load_pokec / feature_norm / index_to_mask。
UDAGCN 算法原样保留（GCN + PPMI 双编码器、注意力融合、GRL 域分类器、熵正则）。
"""

import os
import sys
import random
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# ── 把 dual_gnn 包加入搜索路径 ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.ppmi_conv import PPMIConv

sys.path.append("/home/disk2/lhr/sfda/code")
from utils import fair_metric

# =============================================================================
# 0.  全局设置
# =============================================================================
SEED = 200
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 1.  数据加载（完全照搬 code/dataset.py 中的实现）
# =============================================================================

def feature_norm(features):
    """Min-Max 归一化到 [-1, 1]，与 code/dataset.py 完全一致。"""
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1
    return mask


def load_pokec(
    dataset,
    id,
    sens_attr="region",
    predict_attr="I_am_working_in_field",
    path="/home/disk2/lhr/sfda/fairDomainAdaption/Benchmark-GraphFairness/dataset/",
):
    """
    与 code/dataset.py:load_pokec 逻辑完全一致。

    Benchmark 目录结构：
        dataset/pokec_z/pokec_z.csv
        dataset/pokec_z/pokec_z_edges.txt
        dataset/pokec_n/pokec_n.csv
        dataset/pokec_n/pokec_n_edges.txt

    调用示例：
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
# 2.  加载数据集
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

num_features = src_x.shape[1]
num_classes  = int(max(src_y.max().item(), tgt_y.max().item())) + 1

print(f"Source : {src_x.shape[0]:>6d} nodes | {num_features} features | "
      f"{src_ei.shape[1]} edges")
print(f"Target : {tgt_x.shape[0]:>6d} nodes | {num_features} features | "
      f"{tgt_ei.shape[1]} edges")
print(f"Classes: {num_classes}")

# 包装成 PyG Data 对象，方便统一访问
source_data = Data(
    x=src_x, edge_index=src_ei, y=src_y,
    train_mask=src_train, val_mask=src_val, test_mask=src_test,
    sens_labels=src_sens,
)
target_data = Data(
    x=tgt_x, edge_index=tgt_ei, y=tgt_y,
    train_mask=tgt_train, val_mask=tgt_val, test_mask=tgt_test,
    sens_labels=tgt_sens,
)

source_data = source_data.to(device)
target_data = target_data.to(device)

# =============================================================================
# 3.  UDAGCN 模型（与 UDAGCN_demo.py 完全一致）
# =============================================================================

encoder_dim = 16   # 与原始 demo 默认值一致
use_UDAGCN  = True
rate        = 0.0  # GRL 调制系数，训练时动态更新


class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()
        if base_model is None:
            weights = [None, None]
            biases  = [None, None]
        else:
            weights = [conv.weight for conv in base_model.conv_layers]
            biases  = [conv.bias   for conv in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type
        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv
        self.conv_layers = nn.ModuleList([
            model_cls(num_features, 128,
                      weight=weights[0], bias=biases[0], **kwargs),
            model_cls(128, encoder_dim,
                      weight=weights[1], bias=biases[1], **kwargs),
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * rate, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        return torch.sum(stacked * weights, dim=1)


loss_func    = nn.CrossEntropyLoss().to(device)
encoder      = GNN(type="gcn").to(device)
ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=10).to(device)

cls_model = nn.Sequential(
    nn.Linear(encoder_dim, num_classes),
).to(device)

domain_model = nn.Sequential(
    GRL(),
    nn.Linear(encoder_dim, 40),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(40, 2),
).to(device)

att_model = Attention(encoder_dim).to(device)

models = [encoder, cls_model, domain_model, ppmi_encoder, att_model]
params = itertools.chain(*[m.parameters() for m in models])
optimizer = torch.optim.Adam(params, lr=3e-3)

# =============================================================================
# 4.  UDAGCN 前向工具函数（与 UDAGCN_demo.py 完全一致）
# =============================================================================

def gcn_encode(data, cache_name, mask=None):
    out = encoder(data.x, data.edge_index, cache_name)
    return out if mask is None else out[mask]


def ppmi_encode(data, cache_name, mask=None):
    out = ppmi_encoder(data.x, data.edge_index, cache_name)
    return out if mask is None else out[mask]


def encode(data, cache_name, mask=None):
    gcn_out  = gcn_encode(data, cache_name, mask)
    ppmi_out = ppmi_encode(data, cache_name, mask)
    return att_model([gcn_out, ppmi_out])


def predict(data, cache_name, mask=None):
    return cls_model(encode(data, cache_name, mask))


def test_acc(data, cache_name, mask=None):
    for m in models:
        m.eval()
    with torch.no_grad():
        logits = predict(data, cache_name, mask)
        preds  = logits.argmax(dim=1)
        labels = data.y if mask is None else data.y[mask]
    return preds.eq(labels).float().mean().item()


# =============================================================================
# 5.  训练循环（与 UDAGCN_demo.py 完全一致）
# =============================================================================

epochs = 200


def train(epoch):
    global rate
    for m in models:
        m.train()
    optimizer.zero_grad()

    rate = min((epoch + 1) / epochs, 0.05)

    enc_src = encode(source_data, "source")
    enc_tgt = encode(target_data, "target")
    src_logits = cls_model(enc_src)

    # 分类损失：只用 source train_mask（对应 code/dataset.py 的 80% 训练集）
    cls_loss = loss_func(
        src_logits[source_data.train_mask],
        source_data.y[source_data.train_mask],
    )

    # L2 正则（与原始代码一致）
    for m in models:
        for name, param in m.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3

    # 域分类损失（GRL 已内置在 domain_model）
    src_dom = domain_model(enc_src)
    tgt_dom = domain_model(enc_tgt)
    dom_loss = (
        loss_func(src_dom, torch.zeros(src_dom.size(0), dtype=torch.long, device=device))
        + loss_func(tgt_dom, torch.ones(tgt_dom.size(0),  dtype=torch.long, device=device))
    )

    # 目标域熵正则
    tgt_logits = cls_model(enc_tgt)
    tgt_probs  = torch.clamp(F.softmax(tgt_logits, dim=-1), min=1e-9, max=1.0)
    loss_entropy = torch.mean(torch.sum(-tgt_probs * torch.log(tgt_probs), dim=-1))

    loss = cls_loss + dom_loss + loss_entropy * (epoch / epochs * 0.01)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), cls_loss.item(), dom_loss.item()


# =============================================================================
# 6.  完整评估（Accuracy + AUC-ROC + F1）
# =============================================================================

def full_evaluate(data, cache_name, mask):
    for m in models:
        m.eval()
    with torch.no_grad():
        logits = predict(data, cache_name, mask)
        preds  = logits.argmax(dim=1).cpu().numpy()
        probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    acc      = accuracy_score(y_true, preds)
    f1_micro = f1_score(y_true, preds, average="micro")
    f1_macro = f1_score(y_true, preds, average="macro")

    sens = data.sens_labels[mask].cpu().numpy().astype(int)
    delta_dp, delta_eo = fair_metric(preds, y_true, sens)

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    return acc, auc, delta_dp, delta_eo


# =============================================================================
# 7.  运行
# =============================================================================

print(f"\nDevice : {device}")
print(f"num_features={num_features}  encoder_dim={encoder_dim}  "
      f"num_classes={num_classes}  epochs={epochs}")
print("\nStarting UDAGCN training (pokec_z -> pokec_n) ...\n")

best_tgt_acc = 0.0
best_epoch   = 0

for epoch in range(1, epochs + 1):
    loss, cls_l, dom_l = train(epoch)
    if epoch % 20 == 0:
        src_acc = test_acc(source_data, "source", source_data.test_mask)
        tgt_acc = test_acc(target_data, "target")
        print(f"Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  "
              f"cls={cls_l:.4f}  dom={dom_l:.4f}  "
              f"src_acc={src_acc:.4f}  tgt_acc={tgt_acc:.4f}")
        if tgt_acc > best_tgt_acc:
            best_tgt_acc = tgt_acc
            best_epoch   = epoch

print("\n" + "=" * 60)
print(f"Best target acc={best_tgt_acc:.4f}  @ epoch {best_epoch}")
print("=" * 60)

# Final detailed evaluation on target test set
acc, auc, dp, eo = full_evaluate(
    target_data, "target", target_data.test_mask
)
print(f"\n=== Final Results (pokec_z -> pokec_n) ===")
print(f"Target Accuracy  : {acc:.4f}")
print(f"Target AUC-ROC   : {auc:.4f}")
print(f"Target delta-DP  : {dp:.4f}")
print(f"Target delta-EO  : {eo:.4f}")

# Source performance for reference
src_acc_f, src_auc, src_f1mi, src_f1ma = full_evaluate(
    source_data, "source", source_data.test_mask
)
print(f"\n=== Source Reference (pokec_z test set) ===")
print(f"Source Accuracy  : {src_acc_f:.4f}")
print(f"Source AUC-ROC   : {src_auc:.4f}")
print(f"Source F1 (micro): {src_f1mi:.4f}")
print(f"Source F1 (macro): {src_f1ma:.4f}")
