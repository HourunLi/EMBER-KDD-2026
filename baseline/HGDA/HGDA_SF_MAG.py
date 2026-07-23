# -*- coding: utf-8 -*-
"""
HGDA_SF_MAG: 将 HGDA 从 UGDA 改造成 Source-Free Domain Adaptation 对比基线。

改造原则：
1. 阶段一只访问源域图，使用源域分类损失训练完整 HGDA 特征提取器和分类头。
2. 阶段一结束后保存三个滤波器、组合权重、分类头，以及三个滤波分支的类条件原型。
3. 阶段二只访问目标域图，加载源域保存的模型参数和原型，用伪标签做原型 KL 对齐，
   并保留目标域熵最小化损失。

阶段一和阶段二的主要超参数沿用原始 HGDA_MAG.py：
hidden_dim=64, num_layers=2, lr=0.01, num_epochs=50, alpha=0.1。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops, degree
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import scipy.sparse as sp
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops, degree
except ModuleNotFoundError as e:
    raise ImportError("Please ensure all necessary libraries such as torch, pygda, and torch_geometric are installed.") from e

REPO_ROOT = Path(__file__).resolve().parents[2]
SFFGNN_DIR = REPO_ROOT / "SFFGNN"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SFFGNN_DIR) not in sys.path:
    sys.path.insert(0, str(SFFGNN_DIR))

from EMBER.dataset import get_dataset


def compute_normalized_adjacency(edge_index, num_nodes):
    """计算归一化邻接矩阵 Â = D^(-1/2) A D^(-1/2)。"""
    # 添加自环
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # 计算节点度
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # 构造归一化边权
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # 转换为稀疏张量
    adj = torch.sparse.FloatTensor(edge_index, edge_weight, (num_nodes, num_nodes))

    return adj


def compute_normalized_laplacian(edge_index, num_nodes):
    """计算归一化拉普拉斯矩阵 L̃ = I - D^(-1/2) A D^(-1/2)。"""
    adj = compute_normalized_adjacency(edge_index, num_nodes)
    identity = torch.sparse.FloatTensor(
        torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]),
        torch.ones(num_nodes),
        (num_nodes, num_nodes)
    )
    # L̃ = I - Â
    laplacian = identity - adj
    return laplacian


class SpectralFilter(nn.Module):
    """图谱滤波器基类。"""
    def __init__(self, input_dim, hidden_dim):
        super(SpectralFilter, self).__init__()
        self.weight = nn.Linear(input_dim, hidden_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, filter_matrix):
        # 应用滤波：H^l = ReLU(alpha * filter_matrix @ H^(l-1) @ W^(l-1))
        filtered = torch.sparse.mm(filter_matrix, x) if filter_matrix.is_sparse else filter_matrix @ x
        transformed = self.weight(filtered)
        output = F.relu(self.alpha * transformed)
        return output


class HomophilicFilter(SpectralFilter):
    """同质性低通滤波器：使用归一化邻接矩阵 Â。"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, adj_normalized):
        return super().forward(x, adj_normalized)


class FullPassFilter(SpectralFilter):
    """全通滤波器：H_F = I。"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, num_nodes):
        # 全通分支只使用节点属性，不做图结构滤波。
        transformed = self.weight(x)
        output = F.relu(self.alpha * transformed)
        return output


class HeterophilicFilter(SpectralFilter):
    """异质性高通滤波器：使用归一化拉普拉斯矩阵 L̃。"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, laplacian_normalized):
        return super().forward(x, laplacian_normalized)


class DomainAlignmentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(DomainAlignmentModel, self).__init__()

        self.num_layers = num_layers

        # 为每一层初始化三类滤波器。
        self.homophilic_filters = nn.ModuleList()
        self.fullpass_filters = nn.ModuleList()
        self.heterophilic_filters = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.homophilic_filters.append(HomophilicFilter(in_dim, hidden_dim))
            self.fullpass_filters.append(FullPassFilter(in_dim, hidden_dim))
            self.heterophilic_filters.append(HeterophilicFilter(in_dim, hidden_dim))

        # 三类滤波输出的可学习组合权重。
        self.weight_homo = nn.Parameter(torch.tensor(1.0))
        self.weight_full = nn.Parameter(torch.tensor(1.0))
        self.weight_hetero = nn.Parameter(torch.tensor(1.0))

        # 分类头。
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, num_nodes):
        # 预先计算图滤波矩阵。
        adj_normalized = compute_normalized_adjacency(edge_index, num_nodes)
        laplacian_normalized = compute_normalized_laplacian(edge_index, num_nodes)

        # 三个分支分别逐层传播。
        h_homo = x
        h_full = x
        h_hetero = x

        for i in range(self.num_layers):
            h_homo = self.homophilic_filters[i](h_homo, adj_normalized)
            h_full = self.fullpass_filters[i](h_full, num_nodes)
            h_hetero = self.heterophilic_filters[i](h_hetero, laplacian_normalized)

        # 使用可学习权重组合三个滤波分支输出。
        combined = (self.weight_homo * h_homo +
                   self.weight_full * h_full +
                   self.weight_hetero * h_hetero)

        return combined, h_homo, h_full, h_hetero


# 原始 HGDA 超参数
input_dim = source_dataset.x.size(1)
hidden_dim = 64
output_dim = int(source_dataset.y.max().item()) + 1
num_layers = 2
learning_rate = 0.01
source_epochs = 50
target_epochs = 50
alpha = 0.1

checkpoint_dir = os.path.join(os.path.dirname(__file__), "results", "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "HGDA_SF_MAG_source.pt")

# 准备数据张量
source_features, source_labels = source_dataset.x, source_dataset.y
source_edge_index = source_dataset.edge_index
source_num_nodes = source_features.size(0)

target_features, target_labels = target_dataset.x, target_dataset.y
target_edge_index = target_dataset.edge_index
target_num_nodes = target_features.size(0)

classification_loss_fn = nn.CrossEntropyLoss()
kl_loss_fn = nn.KLDivLoss(reduction='batchmean')


def entropy_minimization_loss(logits):
    """目标域熵最小化损失 L_T。"""
    probs = F.softmax(logits, dim=1).clamp(min=1e-8)
    return -(probs * probs.log()).sum(dim=1).mean()


def compute_class_conditional_prototypes(h_homo, h_full, h_hetero, labels, num_classes):
    """阶段一：按源域真实标签分别计算低通、全通和高通滤波分支的类条件原型。"""
    prototypes = {
        "homo": [],
        "full": [],
        "hetero": [],
    }

    for class_id in range(num_classes):
        class_mask = labels == class_id
        if class_mask.sum().item() == 0:
            # 极端情况下某类缺失，使用零向量占位，保证阶段二维度稳定。
            prototypes["homo"].append(torch.zeros(h_homo.size(1), device=h_homo.device))
            prototypes["full"].append(torch.zeros(h_full.size(1), device=h_full.device))
            prototypes["hetero"].append(torch.zeros(h_hetero.size(1), device=h_hetero.device))
            continue

        prototypes["homo"].append(h_homo[class_mask].mean(dim=0))
        prototypes["full"].append(h_full[class_mask].mean(dim=0))
        prototypes["hetero"].append(h_hetero[class_mask].mean(dim=0))

    return {name: torch.stack(values, dim=0).detach() for name, values in prototypes.items()}


def prototype_kl_loss(target_mean, source_prototype):
    """沿用原 HGDA 的 KL 口径，在隐藏维上比较 target 均值嵌入与 source 原型分布。"""
    return kl_loss_fn(
        F.log_softmax(target_mean.unsqueeze(0), dim=1),
        F.softmax(source_prototype.unsqueeze(0), dim=1),
    )


def source_free_alignment_loss(tgt_homo, tgt_full, tgt_hetero, tgt_logits, prototypes):
    """阶段二：基于伪标签和存储源域原型计算 L_H^SF。"""
    with torch.no_grad():
        pseudo_target_labels = tgt_logits.argmax(dim=1)

    total_loss = tgt_logits.new_tensor(0.0)
    num_nodes = tgt_logits.size(0)
    num_classes = prototypes["homo"].size(0)

    for class_id in range(num_classes):
        class_mask = pseudo_target_labels == class_id
        class_count = class_mask.sum()
        if class_count.item() == 0:
            continue

        # p_hat(c)：目标域伪标签估计的类别先验。
        class_prior = class_count.float() / float(num_nodes)
        tgt_homo_mean = tgt_homo[class_mask].mean(dim=0)
        tgt_full_mean = tgt_full[class_mask].mean(dim=0)
        tgt_hetero_mean = tgt_hetero[class_mask].mean(dim=0)

        class_loss = (
            prototype_kl_loss(tgt_homo_mean, prototypes["homo"][class_id])
            + prototype_kl_loss(tgt_hetero_mean, prototypes["hetero"][class_id])
            + prototype_kl_loss(tgt_full_mean, prototypes["full"][class_id])
        )
        total_loss = total_loss + class_prior * class_loss

    return total_loss


def save_source_knowledge(model, prototypes, path):
    """保存阶段二需要的源域模型参数和类条件原型，不保存源域原始数据。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "prototypes": {k: v.detach().cpu() for k, v in prototypes.items()},
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "num_layers": num_layers,
        },
        path,
    )


def load_source_knowledge(path):
    """加载源域预训练权重和原型，供目标域 source-free 适配使用。"""
    artifact = torch.load(path, map_location="cpu")
    model = DomainAlignmentModel(
        artifact["input_dim"],
        artifact["hidden_dim"],
        artifact["output_dim"],
        artifact["num_layers"],
    )
    model.load_state_dict(artifact["model_state"])
    prototypes = artifact["prototypes"]
    return model, prototypes


def stage1_pretrain_source(model, optimizer, num_epochs=50):
    """阶段一：仅在源图 G^S 上使用 L_S 预训练 HGDA。"""
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 源域前向传播。
        src_combined, src_homo, src_full, src_hetero = model(
            source_features, source_edge_index, source_num_nodes
        )
        src_logits = model.classifier(src_combined)
        source_class_loss = classification_loss_fn(src_logits, source_labels)

        source_class_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            src_combined_eval, _, _, _ = model(
                source_features, source_edge_index, source_num_nodes
            )
            src_logits_eval = model.classifier(src_combined_eval)
            predictions = src_logits_eval.argmax(dim=1)
            source_accuracy = (predictions == source_labels).float().mean().item()

        if (epoch + 1) % 5 == 0:
            print(f"[Stage 1][Epoch {epoch+1}/{num_epochs}], "
                  f"Source Loss: {source_class_loss:.4f}, "
                  f"Source Accuracy: {source_accuracy:.4f}")

    model.eval()
    with torch.no_grad():
        _, src_homo, src_full, src_hetero = model(
            source_features, source_edge_index, source_num_nodes
        )
        prototypes = compute_class_conditional_prototypes(
            src_homo, src_full, src_hetero, source_labels, output_dim
        )

    save_source_knowledge(model, prototypes, checkpoint_path)
    print(f"\n[Stage 1] Source knowledge saved to: {checkpoint_path}")


def stage2_adapt_target(model, prototypes, optimizer, num_epochs=50):
    """阶段二：只使用目标图 G^T、源域预训练权重和存储原型进行无源自适应。"""
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 目标域前向传播。
        tgt_combined, tgt_homo, tgt_full, tgt_hetero = model(
            target_features, target_edge_index, target_num_nodes
        )
        tgt_logits = model.classifier(tgt_combined)

        sf_alignment_loss = source_free_alignment_loss(
            tgt_homo, tgt_full, tgt_hetero, tgt_logits, prototypes
        )
        target_entropy_loss = entropy_minimization_loss(tgt_logits)

        # L_hgdasf = L_H^SF + alpha * L_T
        loss = sf_alignment_loss + alpha * target_entropy_loss
        loss.backward()
        optimizer.step()

        # 目标域标签只用于最终/日志评估，不参与训练损失。
        with torch.no_grad():
            model.eval()
            tgt_combined_eval, _, _, _ = model(
                target_features, target_edge_index, target_num_nodes
            )
            tgt_logits_eval = model.classifier(tgt_combined_eval)
            predictions = tgt_logits_eval.argmax(dim=1)
            test_accuracy = (predictions == target_labels).float().mean().item()

        if (epoch + 1) % 5 == 0:
            print(f"[Stage 2][Epoch {epoch+1}/{num_epochs}], Loss: {loss:.4f}, "
                  f"SF Alignment Loss: {sf_alignment_loss:.4f}, "
                  f"Target Entropy Loss: {target_entropy_loss:.4f}, "
                  f"Target Test Accuracy: {test_accuracy:.4f}")


def test(model):
    model.eval()
    with torch.no_grad():
        tgt_combined, _, _, _ = model(
            target_features, target_edge_index, target_num_nodes
        )
        tgt_logits = model.classifier(tgt_combined)
        predictions = tgt_logits.argmax(dim=1)
        accuracy = (predictions == target_labels).float().mean().item()

        true_labels = target_labels.cpu().numpy()
        pred_labels = predictions.cpu().numpy()
        micro_f1 = f1_score(true_labels, pred_labels, average='micro')
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')

        print(f"\n=== Final Results (HGDA_SF_MAG) ===")
        print(f"Target Domain Test Accuracy: {accuracy:.4f}")
        print(f"Target Domain Micro-F1: {micro_f1:.4f}")
        print(f"Target Domain Macro-F1: {macro_f1:.4f}")

        # 打印滤波器组合权重。
        print(f"\nLearned Filter Weights:")
        print(f"  Homophilic (Low-pass): {model.weight_homo.item():.4f}")
        print(f"  Full-pass: {model.weight_full.item():.4f}")
        print(f"  Heterophilic (High-pass): {model.weight_hetero.item():.4f}")


# 训练并测试 source-free HGDA 基线。
print("Starting HGDA_SF_MAG stage 1: source pretraining...")
source_model = DomainAlignmentModel(input_dim, hidden_dim, output_dim, num_layers=num_layers)
source_optimizer = torch.optim.Adam(source_model.parameters(), lr=learning_rate)
stage1_pretrain_source(source_model, source_optimizer, num_epochs=source_epochs)

# 阶段一结束后释放源域原始数据和源模型；阶段二只依赖 checkpoint 中的权重和原型。
del source_model
del source_optimizer
del source_features
del source_labels
del source_edge_index
del source_num_nodes
del source_dataset
del MAG_CN_dataset
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nStarting HGDA_SF_MAG stage 2: source-free target adaptation...")
target_model, source_prototypes = load_source_knowledge(checkpoint_path)
target_optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate)
stage2_adapt_target(target_model, source_prototypes, target_optimizer, num_epochs=target_epochs)
test(target_model)
