"""
sfda_data.py  --  Generic fairness-dataset loader + GraphAny-compatible wrapper

Supports the four SFDA benchmark dataset pairs:
  pokec   : pokec_z   (source)  ->  pokec_n   (target)
  bailA   : bailA_2   (source)  ->  bailA_1   (target)
  german  : german_2  (source)  ->  german_1  (target)
  syn     : syn-2     (source)  ->  syn-1     (target)

Data loading logic is identical to SFFGNN/dataset.py.
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
from scipy.spatial import distance_matrix as scipy_distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import load_npz
from sklearn.manifold._utils import (
    _binary_search_perplexity as sklearn_binary_search_perplexity,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data root paths  (mirror SFFGNN/dataset.py hard-coded paths)
# ---------------------------------------------------------------------------

BAIL_ROOT    = "/home/disk2/lhr/fairDomainAdaption/mine/dataset/bailA"
GERMAN_ROOT  = "/home/disk2/lhr/fairDomainAdaption/mine/dataset/german"
SYN_ROOT     = "/home/disk2/lhr/fairDomainAdaption/mine/dataset/syn"
POKEC_ROOT   = "/home/disk2/lhr/fairDomainAdaption/mine/dataset_bak/pokec"


# ---------------------------------------------------------------------------
# Shared helpers (mirror SFFGNN/dataset.py)
# ---------------------------------------------------------------------------

def feature_norm(features: torch.Tensor) -> torch.Tensor:
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    denom = (max_values - min_values).clamp(min=1e-8)
    return 2 * (features - min_values).div(denom) - 1


def index_to_mask(node_num: int, index: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = True
    return mask


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(
        1 / (1 + scipy_distance_matrix(x.T.T, x.T.T)),
        columns=x.T.columns, index=x.T.columns,
    )
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    return np.array(idx_map)


def _stratified_split(labels, seed):
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
    return idx_train, idx_val, idx_test


def _cond_gaussian_prob(X, entropy):
    perplexity = np.exp2(entropy)
    distances  = pdist(X, metric="euclidean")
    distances  = squareform(distances)
    distances **= 2
    distances  = distances.astype(np.float32)
    return sklearn_binary_search_perplexity(distances, perplexity, verbose=0)


# ---------------------------------------------------------------------------
# Per-dataset loaders (mirror SFFGNN/dataset.py exactly)
# ---------------------------------------------------------------------------

def load_bail(dataset_id: str, seed: int = 42):
    """Load bailA_1 or bailA_2."""
    path  = BAIL_ROOT
    name  = "bailA"
    sens_attr    = "WHITE"
    predict_attr = "RECID"

    df = pd.read_csv(osp.join(path, f"{name}{dataset_id}.csv"))
    header = list(df.columns)
    header.remove(predict_attr)
    header.remove("user_id")
    header.remove(sens_attr)

    labels      = torch.LongTensor(df[predict_attr].values)
    sens_labels = torch.LongTensor(df[sens_attr].values.astype(int))
    features    = torch.FloatTensor(np.array(df[header], dtype=np.float32))

    npz_path = osp.join(path, f"{name}{dataset_id}_edges.npz")
    txt_path = osp.join(path, f"{name}{dataset_id}_edges.txt")
    if osp.exists(npz_path):
        adj = load_npz(npz_path)
    else:
        if osp.exists(txt_path):
            edges_unordered = np.genfromtxt(txt_path).astype("int")
        else:
            edges_unordered = build_relationship(df[header], thresh=0.6)
            np.savetxt(txt_path, edges_unordered)

        idx_map = {j: i for i, j in enumerate(np.arange(features.shape[0]))}
        edges   = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32,
        )
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

    adj_coo    = adj.tocoo()
    edge_index = torch.from_numpy(
        np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64)
    )

    idx_train, idx_val, idx_test = _stratified_split(labels.numpy(), seed)
    n = features.shape[0]
    train_mask = index_to_mask(n, torch.LongTensor(idx_train))
    val_mask   = index_to_mask(n, torch.LongTensor(idx_val))
    test_mask  = index_to_mask(n, torch.LongTensor(idx_test))

    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask, adj


def load_german(dataset_id: str, seed: int = 42):
    """Load german_1 or german_2."""
    path  = GERMAN_ROOT
    name  = "german"
    sens_attr    = "Gender"
    predict_attr = "GoodCustomer"

    df = pd.read_csv(osp.join(path, f"{name}{dataset_id}.csv"))
    header = list(df.columns)
    header.remove(predict_attr)
    header.remove("OtherLoansAtStore")
    header.remove("PurposeOfLoan")
    header.remove(sens_attr)

    df[sens_attr] = df[sens_attr].replace({"Female": 1, "Male": 0})

    txt_path = osp.join(path, f"{name}{dataset_id}_edges.txt")
    if osp.exists(txt_path):
        edges_unordered = np.genfromtxt(txt_path).astype("int")
    else:
        edges_unordered = build_relationship(df[header], thresh=0.8)
        np.savetxt(txt_path, edges_unordered)

    features    = torch.FloatTensor(np.array(sp.csr_matrix(df[header], dtype=np.float32).todense()))
    labels_np   = df[predict_attr].values.copy()
    labels_np[labels_np == -1] = 0
    labels      = torch.LongTensor(labels_np)
    sens_labels = torch.FloatTensor(df[sens_attr].values.astype(int))

    idx_map = {j: i for i, j in enumerate(np.arange(features.shape[0]))}
    edges   = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    adj_coo    = adj.tocoo()
    edge_index = torch.from_numpy(
        np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64)
    )

    # german uses random.seed(20) in SFFGNN/dataset.py
    random.seed(20)
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

    n = features.shape[0]
    train_mask = index_to_mask(n, torch.LongTensor(idx_train))
    val_mask   = index_to_mask(n, torch.LongTensor(idx_val))
    test_mask  = index_to_mask(n, torch.LongTensor(idx_test))

    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask, adj


def load_syn(dataset_id: str, seed: int = 42):
    """Load syn-1 or syn-2."""
    path = SYN_ROOT
    name = "syn"

    features    = pd.read_csv(osp.join(path, f"{name}{dataset_id}_feat.csv"), header=None)
    features    = torch.FloatTensor(features.values.astype(np.float32))
    labels      = pd.read_csv(osp.join(path, f"{name}{dataset_id}_label.txt"), header=None)
    labels      = torch.LongTensor(labels.values.astype(int).squeeze())
    sens_labels = pd.read_csv(osp.join(path, f"{name}{dataset_id}_sens.txt"), header=None)
    sens_labels = torch.LongTensor(sens_labels.values.astype(int).squeeze())

    edges_path = osp.join(path, f"{name}{dataset_id}_edges.txt")
    if osp.exists(edges_path):
        edges_unordered = np.genfromtxt(edges_path, delimiter=",").astype("int")
    else:
        raise FileNotFoundError(f"Edge file not found: {edges_path}")

    idx_map = {j: i for i, j in enumerate(np.arange(features.shape[0]))}
    edges   = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    adj_coo    = adj.tocoo()
    edge_index = torch.from_numpy(
        np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64)
    )

    idx_train, idx_val, idx_test = _stratified_split(labels.numpy(), seed)
    n = features.shape[0]
    train_mask = index_to_mask(n, torch.LongTensor(idx_train))
    val_mask   = index_to_mask(n, torch.LongTensor(idx_val))
    test_mask  = index_to_mask(n, torch.LongTensor(idx_test))

    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask, adj


def load_pokec(dataset_id: str, seed: int = 42):
    """Load region_job_z or region_job_n (dataset_id='_z' or '_n').
    Mirrors SFFGNN/dataset.py::load_pokec exactly.
    """
    path         = POKEC_ROOT
    dataset      = "region_job"
    sens_attr    = "region"
    predict_attr = "I_am_working_in_field"

    # Feature columns = intersection of region_job_z and region_job_n headers
    header   = list(pd.read_csv(osp.join(path, "region_job_z.csv")).columns)
    header2  = list(pd.read_csv(osp.join(path, "region_job_n.csv")).columns)
    header   = [c for c in header if c in header2]
    header.remove("user_id")
    header.remove(sens_attr)
    header.remove(predict_attr)

    df = pd.read_csv(osp.join(path, f"{dataset}{dataset_id}.csv"))

    features    = torch.FloatTensor(np.array(df[header], dtype=np.float32))
    labels      = torch.LongTensor(df[predict_attr].values)
    labels[labels > 1] = 1
    sens_labels = torch.FloatTensor(df[sens_attr].values.astype(int))

    # Build graph from relationship file (mirrors SFFGNN/dataset.py)
    idx     = np.array(df["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        osp.join(path, f"{dataset}{dataset_id}_relationship.txt"), dtype=int
    )
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=int
    ).reshape(edges_unordered.shape)

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
# Dispatch table
# ---------------------------------------------------------------------------

_LOADERS = {
    "pokec_z": (load_pokec, "_z"),
    "pokec_n": (load_pokec, "_n"),
    "bailA_2": (load_bail,  "_2"),
    "bailA_1": (load_bail,  "_1"),
    "german_2": (load_german, "_2"),
    "german_1": (load_german, "_1"),
    "syn-2":   (load_syn,   "-2"),
    "syn-1":   (load_syn,   "-1"),
}


# ---------------------------------------------------------------------------
# GraphAny-compatible dataset wrapper
# ---------------------------------------------------------------------------

class SFDAGraphDataset:
    """
    Wraps any of the four fairness benchmark datasets into the same interface
    as PokecGraphDataset so run_sfda.py can use them identically.
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
        if ds_name not in _LOADERS:
            raise ValueError(f"Unknown dataset: {ds_name}. Choose from {list(_LOADERS)}")

        self.name              = ds_name
        self.cfg               = cfg
        self.cache_dir         = cache_dir
        self.train_batch_size  = train_batch_size
        self.val_test_batch_size = val_test_batch_size
        self.preprocess_device = preprocess_device
        self.seed              = seed

        loader_fn, loader_arg = _LOADERS[ds_name]
        log.info(f"Loading {ds_name} raw dataset...")
        edge_index, feat_raw, self.label, self.sens_labels, \
            self.train_mask, self.val_mask, self.test_mask, _ = loader_fn(loader_arg, seed=seed)

        # Normalise features exactly as SFFGNN/dataset.py::get_dataset
        feat_raw = feature_norm(feat_raw)

        self.num_class = len(self.label.unique())
        self.n_nodes   = feat_raw.shape[0]

        # Build DGL graph from edge_index
        self.g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=self.n_nodes)
        if cfg.add_self_loop:
            self.g = dgl.add_self_loop(self.g)
        else:
            self.g = dgl.remove_self_loop(self.g)
        if cfg.to_bidirected:
            self.g = dgl.to_bidirected(self.g)
        self.g = dgl.to_simple(self.g)

        self.train_indices = self.train_mask.nonzero().view(-1)

        safe_name = ds_name.replace("/", "_")
        self.cache_f_name = osp.join(
            cache_dir,
            f"{safe_name}_{cfg.n_hops}hop_selfloop={cfg.add_self_loop}_"
            f"bidirected={cfg.to_bidirected}_seed={seed}.pt",
        )
        self.dist_f_name = osp.join(
            cache_dir,
            f"{safe_name}_{cfg.n_hops}hop_selfloop={cfg.add_self_loop}_"
            f"bidirected={cfg.to_bidirected}_seed={seed}_"
            f"{cfg.feat_chn}_entropy={cfg.entropy}_dist.pt",
        )

        self.features, self.unmasked_pred, self.dist = \
            self._prepare(self.g, feat_raw, n_hops=cfg.n_hops)

        del self.g
        del feat_raw
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def to(self, device):
        def _to(x):
            if x is None:
                return None
            if isinstance(x, dict):
                return {k: _to(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_to(i) for i in x]
            if hasattr(x, "to"):
                return x.to(device)
            return x

        for attr in ["label", "train_mask", "val_mask", "test_mask",
                     "train_indices", "features", "unmasked_pred", "dist"]:
            if hasattr(self, attr):
                setattr(self, attr, _to(getattr(self, attr)))

    # ------------------------------------------------------------------
    def sample_k_nodes_per_label(self, label, visible_nodes, k, num_class):
        ref_node_idx = [
            (label[visible_nodes] == lbl).nonzero().view(-1) for lbl in range(num_class)
        ]
        sampled = [
            idx[torch.randperm(len(idx))[:k]] for idx in ref_node_idx
        ]
        return visible_nodes[torch.cat(sampled)]

    def compute_linear_gnn_logits(self, features, n_per_label, visible_nodes, bootstrap=False):
        # lstsq uses CPU gelss (small ref set); full-graph F @ W runs on F's device (GPU).
        preds = {}
        label = self.label
        num_class = self.num_class
        visible_nodes = visible_nodes.to(label.device)

        for channel, F in features.items():
            if bootstrap:
                ref_nodes = self.sample_k_nodes_per_label(
                    label, visible_nodes, n_per_label, num_class
                )
            else:
                ref_nodes = visible_nodes

            Y_L = torch.nn.functional.one_hot(label[ref_nodes], num_class).float()
            W = torch.linalg.lstsq(
                F[ref_nodes].cpu(), Y_L.cpu(), driver="gelss"
            )[0].to(F.device)
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

    def _prepare(self, g, input_feats, n_hops):
        if not osp.exists(self.cache_f_name):
            log.info(f"Computing {self.name} message passing -> {self.cache_f_name}")
            g = g.to(self.preprocess_device)
            dim = input_feats.size(1)

            LP = torch.zeros(n_hops, g.number_of_nodes(), dim).to(self.preprocess_device)
            HP = torch.zeros(n_hops, g.number_of_nodes(), dim).to(self.preprocess_device)

            g.ndata["LP"] = input_feats.to(self.preprocess_device)
            g.ndata["HP"] = input_feats.to(self.preprocess_device)

            for hop in range(n_hops):
                g.update_all(fn.copy_u("LP", "temp"), fn.mean("temp", "LP"))
                g.update_all(fn.copy_u("HP", "temp"), fn.mean("temp", "HP_out"))
                g.ndata["HP"] = g.ndata["HP"] - g.ndata["HP_out"]
                LP[hop] = g.ndata["LP"].clone()
                HP[hop] = g.ndata["HP"].clone()

            lp_feat = {f"L{l+1}": x for l, x in enumerate(LP)}
            hp_feat = {f"H{l+1}": x for l, x in enumerate(HP)}
            features = {"X": input_feats, **lp_feat, **hp_feat}

            unmasked_pred = self.compute_channel_logits(
                features, self.train_indices, sample=False,
                device=self.preprocess_device,
            )
            torch.save((features, unmasked_pred), self.cache_f_name)
        else:
            log.info(f"Loading {self.name} features from cache.")
            features, unmasked_pred = torch.load(self.cache_f_name, map_location="cpu")

        if not osp.exists(self.dist_f_name):
            log.info(f"Computing {self.name} dist maps -> {self.dist_f_name}")
            y_feat = np.stack(
                [unmasked_pred[c].cpu().numpy() for c in self.cfg.feat_channels],
                axis=1,
            )
            bsz, n_channel, n_class = y_feat.shape
            dist_feat_dim = n_channel * (n_channel - 1)

            cond_gaussian_prob = np.zeros((bsz, n_channel, n_channel))
            for i in range(bsz):
                cond_gaussian_prob[i, :, :] = _cond_gaussian_prob(
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
            log.info(f"Loading {self.name} dist from cache.")
            dist = torch.load(self.dist_f_name, map_location="cpu")

        return features, unmasked_pred, dist
