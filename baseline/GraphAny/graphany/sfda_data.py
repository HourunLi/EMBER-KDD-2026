# -*- coding: utf-8 -*-
"""GraphAny 在四个公平迁移数据集上的轻量包装。

数据读取直接复用 SFFGNN/dataset.py，保证原始路径、特征列、标签、
敏感属性、边构造和归一化逻辑与主方法一致。这里额外提供 GraphAny
需要的传播特征和 source-derived LinearGNN 权重工具；target 阶段不会
使用 target 标签来构造输入 logits。
"""

import os
import os.path as osp
import sys
from pathlib import Path
from types import SimpleNamespace

import dgl
import dgl.function as fn
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
SFFGNN_DIR = REPO_ROOT / "SFFGNN"


DATASET_ALIASES = {
    "pokec_z": ("pokec", "_z"),
    "pokec_n": ("pokec", "_n"),
    "bailA_2": ("bailA", "_2"),
    "bailA_1": ("bailA", "_1"),
    "germanA_2": ("germanA", "_2"),
    "germanA_1": ("germanA", "_1"),
    "syn-2": ("syn", "-2"),
    "syn-1": ("syn", "-1"),
}


def load_sffgnn_get_dataset():
    """按 SFFGNN 的原始导入环境加载 get_dataset。"""
    for path in (str(REPO_ROOT), str(SFFGNN_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from EMBER.dataset import get_dataset

    return get_dataset


def sample_k_nodes_per_label(label, visible_nodes, k, num_class):
    """从每个类别采样少量可见节点，保持 GraphAny 原训练协议。"""
    ref_node_idx = [
        (label[visible_nodes] == lbl).nonzero().view(-1)
        for lbl in range(num_class)
    ]
    sampled_indices = []
    for label_indices in ref_node_idx:
        if len(label_indices) == 0:
            continue
        sampled_indices.append(label_indices[torch.randperm(len(label_indices))[:k]])
    if not sampled_indices:
        return visible_nodes[:0]
    return visible_nodes[torch.cat(sampled_indices)]


class FairGraphDataset:
    """GraphAny 可直接使用的公平数据集包装器。"""

    def __init__(
        self,
        ds_name,
        cfg,
        cache_dir,
        train_batch_size=128,
        val_test_batch_size=100000,
        preprocess_device=torch.device("cpu"),
    ):
        if ds_name not in DATASET_ALIASES:
            raise ValueError(f"Unknown fair dataset: {ds_name}")

        self.name = ds_name
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.train_batch_size = train_batch_size
        self.val_test_batch_size = val_test_batch_size
        self.preprocess_device = preprocess_device

        os.makedirs(cache_dir, exist_ok=True)
        data = self._load_raw_data(ds_name)

        self.label = data.y.long()
        self.sens_labels = data.sens_labels
        self.train_mask = data.train_mask.bool()
        self.val_mask = data.val_mask.bool()
        self.test_mask = data.test_mask.bool()
        self.num_class = int(self.label.max().item()) + 1
        self.n_nodes = data.x.shape[0]

        g = dgl.graph(
            (data.edge_index[0], data.edge_index[1]),
            num_nodes=self.n_nodes,
        )
        if cfg.add_self_loop:
            g = dgl.add_self_loop(g)
        else:
            g = dgl.remove_self_loop(g)
        if cfg.to_bidirected:
            g = dgl.to_bidirected(g)
        self.g = dgl.to_simple(g)

        self.train_indices = self.train_mask.nonzero().view(-1)
        self.cache_f_name = osp.join(
            cache_dir,
            f"{self.name}_{cfg.n_hops}hop_selfloop={cfg.add_self_loop}_"
            f"bidirected={cfg.to_bidirected}_features.pt",
        )
        self.features = self.prepare_prop_features(self.g, data.x, cfg.n_hops)

        del self.g
        torch.cuda.empty_cache()

    def _load_raw_data(self, ds_name):
        sffgnn_dataset, domain_id = DATASET_ALIASES[ds_name]
        get_dataset = load_sffgnn_get_dataset()
        return get_dataset(SimpleNamespace(dataset=sffgnn_dataset), domain_id)

    def to(self, device):
        """将训练和评估会用到的张量移动到目标设备。"""

        def move(x):
            if isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            if hasattr(x, "to"):
                return x.to(device)
            return x

        for attr in [
            "label",
            "sens_labels",
            "train_mask",
            "val_mask",
            "test_mask",
            "train_indices",
            "features",
        ]:
            if hasattr(self, attr):
                setattr(self, attr, move(getattr(self, attr)))

    def prepare_prop_features(self, g, input_feats, n_hops):
        """计算 GraphAny 原始的 X/L/H 多通道传播特征。"""
        if osp.exists(self.cache_f_name):
            return torch.load(self.cache_f_name, map_location="cpu")

        g = g.to(self.preprocess_device)
        input_feats = input_feats.to(self.preprocess_device)
        dim = input_feats.size(1)

        lp_feats = torch.zeros(n_hops, g.number_of_nodes(), dim).to(
            self.preprocess_device
        )
        hp_feats = torch.zeros(n_hops, g.number_of_nodes(), dim).to(
            self.preprocess_device
        )

        g.ndata["LP"] = input_feats
        g.ndata["HP"] = input_feats

        for hop_idx in range(n_hops):
            # 低通道：邻居均值聚合。
            g.update_all(fn.copy_u("LP", "temp"), fn.mean("temp", "LP"))

            # 高通道：当前表示减去邻居均值。
            g.update_all(fn.copy_u("HP", "temp"), fn.mean("temp", "HP_out"))
            g.ndata["HP"] = g.ndata["HP"] - g.ndata["HP_out"]

            lp_feats[hop_idx] = g.ndata["LP"].clone()
            hp_feats[hop_idx] = g.ndata["HP"].clone()

        lp_dict = {f"L{idx + 1}": x for idx, x in enumerate(lp_feats)}
        hp_dict = {f"H{idx + 1}": x for idx, x in enumerate(hp_feats)}
        features = {"X": input_feats, **lp_dict, **hp_dict}
        torch.save({k: v.cpu() for k, v in features.items()}, self.cache_f_name)
        return features

    def fit_channel_weights(self, features, visible_nodes, bootstrap=False):
        """只在 source 上拟合每个 LinearGNN channel 的线性权重。"""
        weights = {}
        label = self.label.detach().cpu()
        visible_nodes = visible_nodes.detach().cpu()
        channels = set(self.cfg.feat_channels + self.cfg.pred_channels)

        for channel in channels:
            feat = features[channel].detach().cpu()
            ref_nodes = visible_nodes
            if bootstrap:
                ref_nodes = sample_k_nodes_per_label(
                    label,
                    visible_nodes,
                    self.cfg.n_per_label_examples,
                    self.num_class,
                )
            if len(ref_nodes) == 0:
                raise RuntimeError(f"No visible nodes for channel {channel}")

            y_ref = torch.nn.functional.one_hot(
                label[ref_nodes], self.num_class
            ).float()
            weights[channel] = torch.linalg.lstsq(
                feat[ref_nodes], y_ref, driver="gelss"
            )[0]

        return weights

    def compute_channel_logits(self, features, visible_nodes, sample, device):
        """GraphAny 训练阶段的动态 source logits。"""
        weights = self.fit_channel_weights(
            features,
            visible_nodes,
            bootstrap=sample,
        )
        return self.apply_channel_weights(features, weights, device)

    def apply_channel_weights(self, features, weights, device):
        """用 source-derived 权重生成 logits；target 阶段不需要标签。"""
        logits = {}
        channels = set(self.cfg.feat_channels + self.cfg.pred_channels)
        for channel in channels:
            feat = features[channel].to(device)
            weight = weights[channel].to(device)
            logits[channel] = feat @ weight
        return logits

    def train_dataloader(self):
        return DataLoader(
            self.train_indices,
            batch_size=self.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_mask.nonzero().view(-1),
            batch_size=self.val_test_batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_mask.nonzero().view(-1),
            batch_size=self.val_test_batch_size,
        )
