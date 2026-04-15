"""
HGDA on Pokec: pokec_z (source) -> pokec_n (target)

数据加载直接复用 code/dataset.py 中的 load_pokec / feature_norm / index_to_mask。
HGDA 模型代码原样保留，不做任何修改。
"""

import sys
import os
import random
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops, degree
from torch_geometric.data import Data
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

sys.path.append("/home/disk2/lhr/sfda/code")
from utils import fair_metric

# ──────────────────────────────────────────────────────────
# 0.  Seed
# ──────────────────────────────────────────────────────────
SEED = 20
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1
    return mask


def load_pokec(dataset, id,
               sens_attr="region",
               predict_attr="I_am_working_in_field",
               path="/home/disk2/lhr/sfda/fairDomainAdaption/Benchmark-GraphFairness/dataset/"):
    """直接照搬 code/dataset.py:load_pokec，一字不改。"""
    idx_features_labels = pd.read_csv(
        os.path.join(path, dataset + id, dataset + id + ".csv")
    )
    # 取两个数据集列的交集作为特征列（保证 source/target 特征对齐）
    header = list(
        pd.read_csv(os.path.join(path, "pokec_z", "pokec_z.csv")).columns
    )
    header2 = list(
        pd.read_csv(os.path.join(path, "pokec_n", "pokec_n.csv")).columns
    )
    header = [i for i in header if i in header2]
    header.remove("user_id")
    header.remove(sens_attr)
    header.remove(predict_attr)

    features = idx_features_labels[header]
    features = torch.FloatTensor(np.array(features, dtype=np.float32))

    labels = idx_features_labels[predict_attr].values
    labels = torch.LongTensor(labels)
    labels[labels > 1] = 1

    sens_labels = idx_features_labels[sens_attr].values.astype(int)
    sens_labels = torch.FloatTensor(sens_labels)

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        os.path.join(path, dataset + id, dataset + id + "_edges.txt"), dtype=int
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


# ──────────────────────────────────────────────────────────
# 2.  Load datasets
# ──────────────────────────────────────────────────────────
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

print(f"Source: {src_x.shape[0]} nodes | {src_x.shape[1]} features | "
      f"{src_ei.shape[1]} edges")
print(f"Target: {tgt_x.shape[0]} nodes | {tgt_x.shape[1]} features | "
      f"{tgt_ei.shape[1]} edges")
print(f"Output classes: {int(max(src_y.max(), tgt_y.max())) + 1}")

# ──────────────────────────────────────────────────────────
# 3.  HGDA Model  (原始代码原样保留，不做任何改动)
# ──────────────────────────────────────────────────────────

def compute_normalized_adjacency(edge_index, num_nodes):
    """Compute normalized adjacency matrix Ã = D^(-1/2) A D^(-1/2)"""
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse.FloatTensor(edge_index, edge_weight, (num_nodes, num_nodes))
    return adj

def compute_normalized_laplacian(edge_index, num_nodes):
    """Compute normalized Laplacian L̃ = I - D^(-1/2) A D^(-1/2)"""
    adj = compute_normalized_adjacency(edge_index, num_nodes)
    device = adj.device

    idx = torch.arange(num_nodes, device=device)
    identity = torch.sparse.FloatTensor(
        torch.stack([idx, idx]),
        torch.ones(num_nodes, device=device),
        (num_nodes, num_nodes)
    )

    laplacian = identity - adj
    return laplacian

class SpectralFilter(nn.Module):
    """Base class for spectral filters on graphs"""
    def __init__(self, input_dim, hidden_dim):
        super(SpectralFilter, self).__init__()
        self.weight = nn.Linear(input_dim, hidden_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, filter_matrix):
        filtered = torch.sparse.mm(filter_matrix, x) if filter_matrix.is_sparse else filter_matrix @ x
        transformed = self.weight(filtered)
        output = F.relu(self.alpha * transformed)
        return output


class HomophilicFilter(SpectralFilter):
    """Homophilic (Low-pass) Filter: Uses normalized adjacency Ã"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, adj_normalized):
        return super().forward(x, adj_normalized)


class FullPassFilter(SpectralFilter):
    """Full-pass Filter: H_F = I (identity matrix)"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, num_nodes):
        transformed = self.weight(x)
        output = F.relu(self.alpha * transformed)
        return output

class HeterophilicFilter(SpectralFilter):
    """Heterophilic (High-pass) Filter: Uses normalized Laplacian L̃"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, laplacian_normalized):
        return super().forward(x, laplacian_normalized)


class DomainAlignmentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(DomainAlignmentModel, self).__init__()
        self.num_layers = num_layers
        self.homophilic_filters  = nn.ModuleList()
        self.fullpass_filters    = nn.ModuleList()
        self.heterophilic_filters = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.homophilic_filters.append(HomophilicFilter(in_dim, hidden_dim))
            self.fullpass_filters.append(FullPassFilter(in_dim, hidden_dim))
            self.heterophilic_filters.append(HeterophilicFilter(in_dim, hidden_dim))
        self.weight_homo   = nn.Parameter(torch.tensor(1.0))
        self.weight_full   = nn.Parameter(torch.tensor(1.0))
        self.weight_hetero = nn.Parameter(torch.tensor(1.0))
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, num_nodes):
        adj_normalized       = compute_normalized_adjacency(edge_index, num_nodes)
        laplacian_normalized = compute_normalized_laplacian(edge_index, num_nodes)
        h_homo   = x
        h_full   = x
        h_hetero = x
        for i in range(self.num_layers):
            h_homo   = self.homophilic_filters[i](h_homo, adj_normalized)
            h_full   = self.fullpass_filters[i](h_full, num_nodes)
            h_hetero = self.heterophilic_filters[i](h_hetero, laplacian_normalized)
        combined = (
            self.weight_homo   * h_homo +
            self.weight_full   * h_full +
            self.weight_hetero * h_hetero
        )
        return combined, h_homo, h_full, h_hetero


# ──────────────────────────────────────────────────────────
# 4.  Hyperparameters & setup
# ──────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_EPOCHS = 100
LR         = 0.01

input_dim  = src_x.size(1)
output_dim = int(max(src_y.max(), tgt_y.max()).item()) + 1

model = DomainAlignmentModel(
    input_dim, HIDDEN_DIM, output_dim, num_layers=NUM_LAYERS
).to(DEVICE)
optimizer         = torch.optim.Adam(model.parameters(), lr=LR)
classification_loss_fn = nn.CrossEntropyLoss()
kl_loss_fn        = nn.KLDivLoss(reduction='batchmean')

# move to device
src_x   = src_x.to(DEVICE);   src_y   = src_y.to(DEVICE)
src_ei  = src_ei.to(DEVICE)
tgt_x   = tgt_x.to(DEVICE);   tgt_y   = tgt_y.to(DEVICE)
tgt_ei  = tgt_ei.to(DEVICE)
src_train_mask = src_train.to(DEVICE)
tgt_test_mask  = tgt_test.to(DEVICE)

SRC_N = src_x.size(0)
TGT_N = tgt_x.size(0)

# ──────────────────────────────────────────────────────────
# 5.  Train / Evaluate  (逻辑与原始 HGDA_MAG.py 保持一致)
# ──────────────────────────────────────────────────────────

def train(model, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Source domain forward pass
        src_combined, src_homo, src_full, src_hetero = model(
            src_x, src_ei, SRC_N
        )
        src_logits = model.classifier(src_combined)
        source_class_loss = classification_loss_fn(
            src_logits[src_train_mask], src_y[src_train_mask]
        )

        # Target domain forward pass
        tgt_combined, tgt_homo, tgt_full, tgt_hetero = model(
            tgt_x, tgt_ei, TGT_N
        )
        tgt_logits = model.classifier(tgt_combined)

        # Pseudo-labels for target domain
        with torch.no_grad():
            pseudo_target_labels = tgt_logits.argmax(dim=1)
        target_class_loss = classification_loss_fn(tgt_logits, pseudo_target_labels)

        # Domain alignment loss (KL divergence between filter outputs)
        min_nodes = min(SRC_N, TGT_N)
        src_indices = torch.randperm(SRC_N, device=DEVICE)[:min_nodes]
        tgt_indices = torch.randperm(TGT_N, device=DEVICE)[:min_nodes]

        kl_homo = kl_loss_fn(
            F.log_softmax(tgt_homo[tgt_indices],   dim=1),
            F.softmax(src_homo[src_indices],        dim=1),
        )
        kl_full = kl_loss_fn(
            F.log_softmax(tgt_full[tgt_indices],   dim=1),
            F.softmax(src_full[src_indices],        dim=1),
        )
        kl_hetero = kl_loss_fn(
            F.log_softmax(tgt_hetero[tgt_indices], dim=1),
            F.softmax(src_hetero[src_indices],      dim=1),
        )
        kl_loss = kl_homo + kl_full + kl_hetero

        loss = source_class_loss + 0.1 * kl_loss + 0.1 * target_class_loss
        loss.backward()
        optimizer.step()

        # Evaluate on target test set every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                tgt_combined_eval, _, _, _ = model(tgt_x, tgt_ei, TGT_N)
                tgt_logits_eval = model.classifier(tgt_combined_eval)
                preds = tgt_logits_eval.argmax(dim=1)
                y_true = tgt_y[tgt_test_mask].cpu().numpy()
                y_pred = preds[tgt_test_mask].cpu().numpy()
                test_acc = accuracy_score(y_true, y_pred)
            print(
                f"Epoch [{epoch+1}/{num_epochs}]  "
                f"Loss: {loss:.4f}  "
                f"SrcCE: {source_class_loss:.4f}  "
                f"TgtPseudo: {target_class_loss:.4f}  "
                f"KL: {kl_loss:.4f}  "
                f"TgtTestAcc: {test_acc:.4f}"
            )
            model.train()


def test(model):
    model.eval()
    with torch.no_grad():
        tgt_combined, _, _, _ = model(tgt_x, tgt_ei, TGT_N)
        tgt_logits = model.classifier(tgt_combined)
        predictions = tgt_logits.argmax(dim=1)

        mask   = tgt_test_mask
        y_true = tgt_y[mask].cpu().numpy()
        y_pred = predictions[mask].cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)

        probs = torch.softmax(tgt_logits[mask], dim=1)[:, 1].cpu().numpy()
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = float('nan')

        sens = tgt_sens[mask].cpu().numpy().astype(int)
        delta_dp, delta_eo = fair_metric(y_pred, y_true, sens)

        print(f"\n=== Final Results (pokec_z -> pokec_n) ===")
        print(f"Target Domain ACC         : {accuracy:.4f}")
        print(f"Target Domain ROC-AUC     : {auc:.4f}")
        print(f"Target Domain delta-DP    : {delta_dp:.4f}")
        print(f"Target Domain delta-EO    : {delta_eo:.4f}")
        print(f"\nLearned Filter Weights:")
        print(f"  Homophilic  (low-pass)  : {model.weight_homo.item():.4f}")
        print(f"  Full-pass               : {model.weight_full.item():.4f}")
        print(f"  Heterophilic (high-pass): {model.weight_hetero.item():.4f}")


# ──────────────────────────────────────────────────────────
# 6.  Run
# ──────────────────────────────────────────────────────────
print(f"\nDevice : {DEVICE}")
print(f"Input  : {input_dim}  Hidden: {HIDDEN_DIM}  Output: {output_dim}  Layers: {NUM_LAYERS}")
print(f"Epochs : {NUM_EPOCHS}  LR: {LR}\n")
print("Starting training...")
train(model, optimizer, num_epochs=NUM_EPOCHS)
test(model)
