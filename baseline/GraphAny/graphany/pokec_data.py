"""
pokec_data.py  --  Pokec dataset loader + GraphAny-compatible wrapper

Source dataset : pokec_z
Target dataset : pokec_n

Preprocessing exactly follows /home/disk2/lhr/sfda/code/dataset.py::load_pokec.
The PokecGraphDataset class exposes the same interface as GraphAny's GraphDataset
so the existing InductiveNodeClassification training loop works unchanged.
"""

import logging
import os
import os.path as osp
import random

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold._utils import (
    _binary_search_perplexity as sklearn_binary_search_perplexity,
)
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

POKEC_DATA_ROOT = (
    "/home/disk2/lhr/sfda/fairDomainAdaption/Benchmark-GraphFairness/dataset"
)
SENS_ATTR    = "region"
PREDICT_ATTR = "I_am_working_in_field"


# ---------------------------------------------------------------------------
# Helper utilities  (mirror dataset.py)
# ---------------------------------------------------------------------------

def feature_norm(features: torch.Tensor) -> torch.Tensor:
    """Min-max normalise to [-1, 1]  (matches dataset.py::feature_norm)."""
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    denom = (max_values - min_values).clamp(min=1e-8)
    return 2 * (features - min_values).div(denom) - 1


def index_to_mask(node_num: int, index: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = True
    return mask


def _get_cond_gaussian_prob(X, entropy, metric="euclidean"):
    """Entropy-normalised conditional Gaussian probability (same as graphany/data.py)."""
    perplexity = np.exp2(entropy)
    distances  = pdist(X, metric=metric)
    distances  = squareform(distances)
    distances **= 2
    distances  = distances.astype(np.float32)
    return sklearn_binary_search_perplexity(distances, perplexity, verbose=0)


# ---------------------------------------------------------------------------
# Pokec loader  (mirrors dataset.py::load_pokec exactly)
# ---------------------------------------------------------------------------

def load_pokec(
    dataset_name: str,
    sens_attr: str = SENS_ATTR,
    predict_attr: str = PREDICT_ATTR,
    data_root: str = POKEC_DATA_ROOT,
    seed: int = 42,
):
    """
    Load one pokec split ("pokec_z" or "pokec_n").

    Feature columns are the *intersection* of columns present in BOTH pokec_z
    and pokec_n (excluding user_id, sens_attr, predict_attr) -- exactly as in
    dataset.py::load_pokec which intersects region_job_z and region_job_n headers.

    Returns
    -------
    edge_index  : LongTensor  [2, E]
    features    : FloatTensor [N, F]   (NOT yet normalised)
    labels      : LongTensor  [N]      (binarised: values > 1 -> 1)
    sens_labels : FloatTensor [N]
    train_mask  : BoolTensor  [N]
    val_mask    : BoolTensor  [N]
    test_mask   : BoolTensor  [N]
    adj         : scipy.sparse.coo_matrix  (symmetric, with self-loops)
    """
    path_z = osp.join(data_root, "pokec_z")
    path_n = osp.join(data_root, "pokec_n")

    # Intersection of columns in both splits (same logic as dataset.py)
    header_z = list(pd.read_csv(osp.join(path_z, "pokec_z.csv"), nrows=0).columns)
    header_n = list(pd.read_csv(osp.join(path_n, "pokec_n.csv"), nrows=0).columns)
    common_cols = [c for c in header_z if c in header_n]

    feat_cols = [
        c for c in common_cols
        if c not in ("user_id", sens_attr, predict_attr)
    ]

    csv_path = osp.join(data_root, dataset_name, f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)

    features    = torch.FloatTensor(np.array(df[feat_cols], dtype=np.float32))
    labels      = torch.LongTensor(df[predict_attr].values)
    labels[labels > 1] = 1          # binarise as in dataset.py
    sens_labels = torch.FloatTensor(df[sens_attr].values.astype(int))

    # ---- Build adjacency from edge list ---------------------------------
    edge_file = osp.join(data_root, dataset_name, f"{dataset_name}_edges.txt")
    node_ids  = np.array(df["user_id"], dtype=int)
    idx_map   = {j: i for i, j in enumerate(node_ids)}

    edges_unordered = np.genfromtxt(edge_file, dtype=int)
    mapped = [idx_map.get(v) for v in edges_unordered.flatten()]
    edges  = np.array(mapped, dtype=object).reshape(edges_unordered.shape)
    valid  = np.all(edges != None, axis=1)  # noqa: E711
    edges  = edges[valid].astype(np.int64)

    n = labels.shape[0]
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(n, n), dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(n)

    adj_coo    = adj.tocoo()
    edge_index = torch.from_numpy(
        np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64)
    )

    # ---- 80 / 10 / 10 stratified split (matches dataset.py) ------------
    random.seed(seed)
    label_idx_0 = list(np.where(labels == 0)[0])
    label_idx_1 = list(np.where(labels == 1)[0])
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    label_idx_0 = np.array(label_idx_0)
    label_idx_1 = np.array(label_idx_1)

    idx_train = np.append(
        label_idx_0[: int(0.8 * len(label_idx_0))],
        label_idx_1[: int(0.8 * len(label_idx_1))],
    )
    idx_val = np.append(
        label_idx_0[int(0.8 * len(label_idx_0)): int(0.9 * len(label_idx_0))],
        label_idx_1[int(0.8 * len(label_idx_1)): int(0.9 * len(label_idx_1))],
    )
    idx_test = np.append(
        label_idx_0[int(0.9 * len(label_idx_0)):],
        label_idx_1[int(0.9 * len(label_idx_1)):],
    )

    train_mask = index_to_mask(n, torch.LongTensor(idx_train))
    val_mask   = index_to_mask(n, torch.LongTensor(idx_val))
    test_mask  = index_to_mask(n, torch.LongTensor(idx_test))

    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask, adj


# ---------------------------------------------------------------------------
# GraphAny-Compatible Dataset Wrapper
# ---------------------------------------------------------------------------

class PokecGraphDataset:
    """
    Wraps the Pokec dataset into an interface identical to `GraphDataset`
    from `data.py` so it works seamlessly with GraphAny's pipelines.
    """
    def __init__(
        self,
        ds_name: str,
        cfg,
        cache_dir: str,
        train_batch_size: int = 128,
        val_test_batch_size: int = 100000,
        preprocess_device=torch.device("cpu"),
        seed: int = 42,
    ):
        self.name = ds_name
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.train_batch_size = train_batch_size
        self.val_test_batch_size = val_test_batch_size
        self.preprocess_device = preprocess_device
        self.seed = seed

        # 1. Load exact Pokec data
        log.info(f"Loading {self.name} raw dataset...")
        edge_index, self.feat, self.label, self.sens_labels, \
            self.train_mask, self.val_mask, self.test_mask, _ = load_pokec(
            dataset_name=self.name, seed=self.seed
        )
        
        self.num_class = len(self.label.unique())
        self.n_nodes = self.feat.shape[0]

        # 2. Build DGL Graph directly from edges
        self.g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=self.n_nodes)
        
        if self.cfg.add_self_loop:
            self.g = dgl.add_self_loop(self.g)
        else:
            self.g = dgl.remove_self_loop(self.g)
            
        if self.cfg.to_bidirected:
            self.g = dgl.to_bidirected(self.g)
            
        self.g = dgl.to_simple(self.g)
        self.train_indices = self.train_mask.nonzero().view(-1)

        # 3. Formulate Cache Paths
        self.cache_f_name = osp.join(
            cache_dir,
            f"{self.name}_{cfg.n_hops}hop_selfloop={cfg.add_self_loop}_"
            f"bidirected={cfg.to_bidirected}_seed={self.seed}.pt",
        )
        self.dist_f_name = osp.join(
            cache_dir,
            f"{self.name}_{cfg.n_hops}hop_selfloop={cfg.add_self_loop}_"
            f"bidirected={cfg.to_bidirected}_seed={self.seed}_"
            f"{cfg.feat_chn}_entropy={cfg.entropy}_dist.pt",
        )

        # 4. Process/Load Propagated Features, Logits, and Distances
        self.features, self.unmasked_pred, self.dist = \
            self.prepare_prop_features_logits_and_dist_features(self.g, self.feat, n_hops=cfg.n_hops)

        # Free graph from memory as training primarily relies on features/logits
        del self.g
        del self.feat
        torch.cuda.empty_cache()

    def to(self, device):
        """Recursively transfers nested tensors & dicts to specific target device."""
        def to_device(input_data):
            if input_data is None:
                return None
            elif isinstance(input_data, dict):
                return {k: to_device(v) for k, v in input_data.items()}
            elif isinstance(input_data, list):
                return [to_device(i) for i in input_data]
            elif hasattr(input_data, "to"):
                return input_data.to(device)
            return input_data

        attrs = [
            "label", "train_mask", "val_mask", "test_mask",
            "train_indices", "unmasked_pred", "dist"
        ]
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, to_device(getattr(self, attr)))

    def sample_k_nodes_per_label(self, label, visible_nodes, k, num_class):
        ref_node_idx = [
            (label[visible_nodes] == lbl).nonzero().view(-1) for lbl in range(num_class)
        ]
        sampled_indices = [
            label_indices[torch.randperm(len(label_indices))[:k]]
            for label_indices in ref_node_idx
        ]
        return visible_nodes[torch.cat(sampled_indices)]

    def compute_linear_gnn_logits(self, features, n_per_label_examples, visible_nodes, bootstrap=False):
        """Standard CPU-backed gelss solver for LinearGNN."""
        preds = {}
        label = self.label.cpu()
        visible_nodes = visible_nodes.cpu()
        num_class = self.num_class

        for channel, F in features.items():
            F = F.cpu()
            if bootstrap:
                ref_nodes = self.sample_k_nodes_per_label(
                    label, visible_nodes, n_per_label_examples, num_class
                )
            else:
                ref_nodes = visible_nodes

            Y_L = torch.nn.functional.one_hot(label[ref_nodes], num_class).float()
            
            # Solve exact formulation W = (F^T F)^(-1) F^T Y
            W = torch.linalg.lstsq(F[ref_nodes], Y_L, driver="gelss")[0]
            preds[channel] = F @ W

        return preds

    def compute_channel_logits(self, features, visible_nodes, sample, device):
        target_channels = set(self.cfg.feat_channels + self.cfg.pred_channels)
        pred_logits = self.compute_linear_gnn_logits(
            {c: features[c] for c in target_channels},
            self.cfg.n_per_label_examples,
            visible_nodes,
            bootstrap=sample,
        )
        return {c: logits.to(device) for c, logits in pred_logits.items()}

    def prepare_prop_features_logits_and_dist_features(self, g, input_feats, n_hops):
        """Calculate Low-pass/High-pass features, compute channel logits and distance maps."""
        # 1. Message Passing (LP and HP Generation)
        if not os.path.exists(self.cache_f_name):
            log.info(f"Computing {self.name} message passing to {self.cache_f_name}")
            g = g.to(self.preprocess_device)
            dim = input_feats.size(1)

            LP = torch.zeros(n_hops, g.number_of_nodes(), dim).to(self.preprocess_device)
            HP = torch.zeros(n_hops, g.number_of_nodes(), dim).to(self.preprocess_device)

            g.ndata["LP"] = input_feats.to(self.preprocess_device)
            g.ndata["HP"] = input_feats.to(self.preprocess_device)

            for hop_idx in range(n_hops):
                # Low-pass filter (D^-1 A)
                g.update_all(fn.copy_u("LP", "temp"), fn.mean("temp", "LP"))
                
                # High-pass filter (I - D^-1 A)
                g.update_all(fn.copy_u("HP", "temp"), fn.mean("temp", "HP_out"))
                g.ndata["HP"] = g.ndata["HP"] - g.ndata["HP_out"]

                LP[hop_idx] = g.ndata["LP"].clone()
                HP[hop_idx] = g.ndata["HP"].clone()

            lp_feat_dict = {f"L{l + 1}": x for l, x in enumerate(LP)}
            hp_feat_dict = {f"H{l + 1}": x for l, x in enumerate(HP)}

            features = {"X": input_feats, **lp_feat_dict, **hp_feat_dict}

            unmasked_pred = self.compute_channel_logits(
                features,
                self.train_indices,
                sample=False,
                device=self.preprocess_device,
            )
            torch.save((features, unmasked_pred), self.cache_f_name)
        else:
            log.info(f"Loading {self.name} message passing features from cache.")
            features, unmasked_pred = torch.load(self.cache_f_name, map_location="cpu")

        # 2. Distance Computation (Conditional Gaussian Probability mapping)
        if not os.path.exists(self.dist_f_name):
            log.info(f"Computing {self.name} dist maps to {self.dist_f_name}")
            y_feat = np.stack(
                [unmasked_pred[c].cpu().numpy() for c in self.cfg.feat_channels],
                axis=1,
            )
            bsz, n_channel, n_class = y_feat.shape
            dist_feat_dim = n_channel * (n_channel - 1)

            cond_gaussian_prob = np.zeros((bsz, n_channel, n_channel))
            for i in range(bsz):
                # Uses _get_cond_gaussian_prob previously defined in pokec_data.py
                cond_gaussian_prob[i, :, :] = _get_cond_gaussian_prob(
                    y_feat[i, :, :], self.cfg.entropy
                )

            dist = np.zeros((bsz, dist_feat_dim), dtype=np.float32)
            pair_index = 0
            for c in range(n_channel):
                for c_prime in range(n_channel):
                    if c != c_prime:
                        dist[:, pair_index] = cond_gaussian_prob[:, c, c_prime]
                        pair_index += 1

            dist = torch.from_numpy(dist)
            torch.save(dist, self.dist_f_name)
        else:
            log.info(f"Loading {self.name} conditional gaussian dist from cache.")
            dist = torch.load(self.dist_f_name, map_location="cpu")

        return features, unmasked_pred, dist