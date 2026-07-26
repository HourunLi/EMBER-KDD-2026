import pandas as pd
import os
import numpy as np
import random
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
import torch
from utils import sens_correlation
import scipy.sparse as sp
from scipy.sparse import load_npz


def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1

    return mask


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map


def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="dataset/credit/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    # sensitive feature removal
    # header.remove('Age')

    #    # Normalize MaxBillAmountOverLast6Months
    #    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
    #
    #    # Normalize MaxPaymentAmountOverLast6Months
    #    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
    #
    #    # Normalize MostRecentBillAmount
    #    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
    #
    #    # Normalize MostRecentPaymentAmount
    #    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
    #
    #    # Normalize TotalMonthsOverdue
    #    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # print(features)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    #
    #

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(
        label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(
        0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="dataset/bail/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    # print(features)

    # features = normalize(features)
    # adj = adj + sp.eye(adj.shape[0])

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(
        label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(
        0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens

# 自己改的
def load_bailA(dataset, sens_attr="WHITE", predict_attr="RECID", path="dataset/bailA/", label_number=1000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    labels = idx_features_labels[predict_attr].values
    labels = torch.LongTensor(labels)
    sens_labels = idx_features_labels[sens_attr].values.astype(int)
    sens_labels = torch.LongTensor(sens_labels)
    features = idx_features_labels[header]
    features = torch.FloatTensor(np.array(features, dtype=np.float32))
    adj = load_npz(f'{path}/{dataset}_edges.npz')

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:int(0.5 * len(label_idx_0))],
                          label_idx_1[: int(0.5 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], 
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):],
                         label_idx_1[int(0.75 * len(label_idx_1)):])
    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens


def _stratified_source_masks(labels, sens, val_ratio=0.2, seed=20,
                             eligible_mask=None):
    """Split a source domain without using any target-domain information.

    The split is stratified by the joint (label, sensitive attribute) value so
    both source train and source validation normally contain every group needed
    by the utility and fairness metrics.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"source validation ratio must be in (0, 1), got {val_ratio}")

    labels_np = labels.cpu().numpy().astype(np.int64)
    sens_np = sens.cpu().numpy().astype(np.int64)
    if eligible_mask is None:
        eligible_np = np.ones(len(labels_np), dtype=bool)
    else:
        eligible_np = eligible_mask.cpu().numpy().astype(bool)
    rng = np.random.RandomState(seed)
    train_indices, val_indices = [], []

    for label_value in np.unique(labels_np):
        for sens_value in np.unique(sens_np):
            indices = np.where(
                eligible_np
                & (labels_np == label_value)
                & (sens_np == sens_value))[0]
            if len(indices) == 0:
                continue
            rng.shuffle(indices)
            val_count = max(1, int(round(len(indices) * val_ratio)))
            if val_count == len(indices) and len(indices) > 1:
                val_count -= 1
            val_indices.extend(indices[:val_count].tolist())
            train_indices.extend(indices[val_count:].tolist())

    train_mask = index_to_mask(
        len(labels), torch.LongTensor(sorted(train_indices)))
    val_mask = index_to_mask(
        len(labels), torch.LongTensor(sorted(val_indices)))
    return train_mask, val_mask


def _read_domain_edge_list(edge_path):
    """Read comma- or whitespace-delimited integer edge endpoints."""
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"domain edge list not found: {edge_path}")

    first_data_line = None
    with open(edge_path, "r", encoding="utf-8-sig") as edge_file:
        for line in edge_file:
            stripped_line = line.strip()
            if stripped_line:
                first_data_line = stripped_line
                break
    if first_data_line is None:
        raise ValueError(f"edge list is empty: {edge_path}")

    # Synthetic edge files are comma-separated (for example ``0,177``), while
    # GermanA/BailA/Pokec files use spaces or tabs. ``delimiter=None`` lets
    # NumPy handle arbitrary whitespace for the latter group.
    delimiter = "," if "," in first_data_line else None

    # np.savetxt() writes arrays as values such as 1.000000e+00 by default.
    # Some NumPy versions turn those strings into the integer fill value -1
    # when genfromtxt is called directly with dtype=int64. Read as float first,
    # verify integral endpoints, and only then cast to int64.
    edges_float = np.genfromtxt(
        edge_path, dtype=np.float64, delimiter=delimiter,
        encoding="utf-8-sig")
    if edges_float.size == 0:
        raise ValueError(f"edge list is empty: {edge_path}")
    if edges_float.ndim == 1:
        edges_float = edges_float.reshape(1, -1)
    if edges_float.shape[1] != 2:
        raise ValueError(f"edge list must have two columns: {edge_path}")
    if not np.isfinite(edges_float).all():
        raise ValueError(f"edge list contains NaN or infinite values: {edge_path}")

    rounded_edges = np.rint(edges_float)
    if not np.allclose(edges_float, rounded_edges, rtol=0, atol=1e-8):
        raise ValueError(f"edge list contains non-integer endpoints: {edge_path}")
    return rounded_edges.astype(np.int64)


def _load_domain_adjacency(edge_path, node_count):
    """Load a re-indexed domain edge list and build both graph formats."""
    edges = _read_domain_edge_list(edge_path)
    if edges.ndim == 1:
        edges = edges.reshape(1, -1)
    if edges.size and (edges.min() < 0 or edges.max() >= node_count):
        raise ValueError(
            f"edge index is outside [0, {node_count - 1}] in {edge_path}")

    return _build_domain_adjacency(edges, node_count)


def _build_domain_adjacency(edges, node_count):
    """Build normalized sparse adjacency and edge_index from row indices."""
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0], dtype=np.float32),
         (edges[:, 0], edges[:, 1])),
        shape=(node_count, node_count), dtype=np.float32)
    # The supplied domain files are already symmetric. maximum() also keeps
    # this loader correct if a future file contains only one edge direction.
    adj = adj.maximum(adj.T).tocsr()
    # Preserve the self-loop behavior of the original loaders while avoiding
    # duplicate accumulation for domain files that already contain loops.
    adj = adj.maximum(sp.eye(node_count, dtype=np.float32, format="csr"))
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(
        sys_normalized_adjacency(adj))
    edge_index, _ = from_scipy_sparse_matrix(adj)
    return adj_norm_sp, edge_index


def _load_id_mapped_domain_adjacency(edge_path, user_ids):
    """Map original user IDs in an edge list to current-domain row indices."""
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"domain edge list not found: {edge_path}")

    user_ids = np.asarray(user_ids, dtype=np.int64)
    if len(np.unique(user_ids)) != len(user_ids):
        raise ValueError("domain CSV contains duplicate user_id values")

    unordered_edges = _read_domain_edge_list(edge_path)

    # Some ID-based edge files may use a negative sentinel when an original
    # endpoint is outside the current domain. Such rows are not valid nodes in
    # the induced subgraph.
    negative_endpoint_mask = (unordered_edges < 0).any(axis=1)
    removed_edge_count = int(negative_endpoint_mask.sum())
    if removed_edge_count > 0:
        unordered_edges = unordered_edges[~negative_endpoint_mask]
        print(
            f"Filtered {removed_edge_count} edge(s) with negative sentinel "
            f"endpoints from {edge_path}")
    if unordered_edges.shape[0] == 0:
        raise ValueError(
            f"no valid within-domain edges remain after filtering {edge_path}")

    id_to_row = pd.Series(np.arange(len(user_ids)), index=user_ids)
    mapped_edges = id_to_row.reindex(unordered_edges.reshape(-1))
    if mapped_edges.isna().any():
        missing_ids = np.unique(
            unordered_edges.reshape(-1)[mapped_edges.isna().to_numpy()])
        preview = missing_ids[:10].tolist()
        raise ValueError(
            f"{edge_path} contains user IDs absent from its domain CSV: "
            f"{preview}{' ...' if len(missing_ids) > 10 else ''}")

    edges = mapped_edges.to_numpy(dtype=np.int64).reshape(-1, 2)
    return _build_domain_adjacency(edges, len(user_ids))


def load_bailA_domain(dataset, role, preprocessing_stats=None,
                      sens_attr="WHITE", predict_attr="RECID",
                      path="dataset/bailA/", source_val_ratio=0.2,
                      split_seed=20):
    """Load one BailA domain for source-only training or target-only testing.

    ``bailA_1.csv`` and ``bailA_2.csv`` contain an extra ``user_id`` column
    that is absent from the original combined ``bailA.csv``.  It identifies
    rows in the original graph and must not be used as a predictive feature.

    Source normalization statistics are returned as part of the learned
    preprocessing state.  A target domain must reuse those statistics instead
    of fitting preprocessing on target data.
    """
    if role not in {"source", "target"}:
        raise ValueError(f"role must be 'source' or 'target', got {role!r}")

    csv_path = os.path.join(path, f"{dataset}.csv")
    edge_path = os.path.join(path, f"{dataset}_edges.txt")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"BailA domain CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    feature_names = [
        name for name in frame.columns
        if name not in {predict_attr, "user_id"}
    ]
    if sens_attr not in feature_names:
        raise ValueError(f"sensitive attribute {sens_attr!r} is not a feature")

    raw_features = torch.FloatTensor(
        frame[feature_names].to_numpy(dtype=np.float32))
    labels = torch.LongTensor(frame[predict_attr].to_numpy(dtype=np.int64))
    sens = torch.LongTensor(frame[sens_attr].to_numpy(dtype=np.int64))
    sens_idx = feature_names.index(sens_attr)

    if role == "source":
        feature_min = raw_features.min(dim=0)[0]
        feature_max = raw_features.max(dim=0)[0]
        preprocessing_stats = {
            "feature_names": tuple(feature_names),
            "feature_min": feature_min.clone(),
            "feature_max": feature_max.clone(),
            "sens_idx": sens_idx,
        }
    else:
        if preprocessing_stats is None:
            raise ValueError(
                "target loading requires normalization statistics fitted on the source domain")
        expected_names = tuple(preprocessing_stats["feature_names"])
        if tuple(feature_names) != expected_names:
            raise ValueError(
                "source and target feature columns differ: "
                f"source={expected_names}, target={tuple(feature_names)}")
        if sens_idx != preprocessing_stats["sens_idx"]:
            raise ValueError("source and target sensitive-attribute indices differ")
        feature_min = preprocessing_stats["feature_min"].cpu()
        feature_max = preprocessing_stats["feature_max"].cpu()

    feature_range = feature_max - feature_min
    safe_range = torch.where(feature_range == 0,
                             torch.ones_like(feature_range), feature_range)
    features = 2 * (raw_features - feature_min) / safe_range - 1
    # Preserve the original binary sensitive attribute, as in get_dataset().
    features[:, sens_idx] = raw_features[:, sens_idx]

    adj_norm_sp, edge_index = _load_domain_adjacency(
        edge_path, len(frame))

    if role == "source":
        train_mask, val_mask = _stratified_source_masks(
            labels, sens, val_ratio=source_val_ratio, seed=split_seed)
        test_mask = torch.zeros(len(frame), dtype=torch.bool)
    else:
        train_mask = torch.zeros(len(frame), dtype=torch.bool)
        val_mask = torch.zeros(len(frame), dtype=torch.bool)
        test_mask = torch.ones(len(frame), dtype=torch.bool)

    data = Data(
        x=features,
        edge_index=edge_index,
        adj_norm_sp=adj_norm_sp,
        y=labels.float(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        sens=sens,
    )
    return data, sens_idx, preprocessing_stats


def load_germanA_domain(dataset, role, preprocessing_stats=None,
                        sens_attr="Gender", predict_attr="GoodCustomer",
                        path="dataset/germanA/", source_val_ratio=0.2,
                        split_seed=20):
    """Load GermanA_2 for source training or GermanA_1 for target testing.

    This follows the original German/GermanA feature processing: the two
    categorical columns ``OtherLoansAtStore`` and ``PurposeOfLoan`` are
    removed, Gender is converted to a binary feature, and the remaining German
    features are not min/max-normalized.  The split-file-only ``user_id`` is
    also excluded because it is an identifier rather than a model feature.
    """
    if role not in {"source", "target"}:
        raise ValueError(f"role must be 'source' or 'target', got {role!r}")

    csv_path = os.path.join(path, f"{dataset}.csv")
    edge_path = os.path.join(path, f"{dataset}_edges.txt")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GermanA domain CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "user_id" not in frame.columns:
        raise ValueError(
            f"GermanA domain CSV must contain user_id for edge mapping: {csv_path}")
    if sens_attr not in frame.columns:
        raise ValueError(f"sensitive attribute {sens_attr!r} is missing")
    if predict_attr not in frame.columns:
        raise ValueError(f"prediction attribute {predict_attr!r} is missing")

    frame = frame.copy()
    gender_mapping = {
        "Male": 0, "Female": 1,
        "0": 0, "1": 1,
        0: 0, 1: 1,
    }
    mapped_gender = frame[sens_attr].map(gender_mapping)
    if mapped_gender.isna().any():
        unknown_values = sorted(
            frame.loc[mapped_gender.isna(), sens_attr].astype(str).unique())
        raise ValueError(
            f"unsupported {sens_attr} values in {csv_path}: {unknown_values}")
    frame[sens_attr] = mapped_gender.astype(np.int64)

    excluded_columns = {
        predict_attr,
        "user_id",
        "OtherLoansAtStore",
        "PurposeOfLoan",
    }
    feature_names = [
        name for name in frame.columns if name not in excluded_columns]
    if sens_attr not in feature_names:
        raise ValueError(f"sensitive attribute {sens_attr!r} is not a feature")

    features = torch.FloatTensor(
        frame[feature_names].to_numpy(dtype=np.float32))
    labels_array = frame[predict_attr].to_numpy(dtype=np.int64).copy()
    label_mask = torch.BoolTensor(labels_array >= 0)
    labels_array[labels_array == -1] = 0
    labels = torch.LongTensor(labels_array)
    sens = torch.LongTensor(frame[sens_attr].to_numpy(dtype=np.int64))
    sens_mask = (sens == 0) | (sens == 1)
    sens_idx = feature_names.index(sens_attr)

    if role == "source":
        preprocessing_stats = {
            "dataset_family": "germanA",
            "feature_names": tuple(feature_names),
            "sens_idx": sens_idx,
        }
    else:
        if preprocessing_stats is None:
            raise ValueError(
                "target loading requires preprocessing metadata fitted on the source domain")
        if preprocessing_stats.get("dataset_family") != "germanA":
            raise ValueError("source preprocessing metadata is not for GermanA")
        expected_names = tuple(preprocessing_stats["feature_names"])
        if tuple(feature_names) != expected_names:
            raise ValueError(
                "source and target feature columns differ: "
                f"source={expected_names}, target={tuple(feature_names)}")
        if sens_idx != preprocessing_stats["sens_idx"]:
            raise ValueError("source and target sensitive-attribute indices differ")

    # GermanA split edge files are already re-indexed to local CSV row numbers
    # (0 ... number_of_domain_nodes - 1). The CSV user_id column refers to the
    # original full German graph and must not be used to remap these edges.
    adj_norm_sp, edge_index = _load_domain_adjacency(
        edge_path, len(frame))

    if role == "source":
        train_mask, val_mask = _stratified_source_masks(
            labels, sens, val_ratio=source_val_ratio, seed=split_seed,
            eligible_mask=label_mask & sens_mask)
        test_mask = torch.zeros(len(frame), dtype=torch.bool)
    else:
        train_mask = torch.zeros(len(frame), dtype=torch.bool)
        val_mask = torch.zeros(len(frame), dtype=torch.bool)
        test_mask = torch.ones(len(frame), dtype=torch.bool)

    data = Data(
        x=features,
        edge_index=edge_index,
        adj_norm_sp=adj_norm_sp,
        y=labels.float(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        sens=sens,
        label_mask=label_mask,
        sens_mask=sens_mask,
    )
    return data, sens_idx, preprocessing_stats


def load_pokec_domain(dataset, role, preprocessing_stats=None,
                      sens_attr="region",
                      predict_attr="I_am_working_in_field",
                      path="dataset", source_val_ratio=0.2,
                      split_seed=20):
    """Load Pokec_z as a source graph or Pokec_n as a target graph.

    Pokec_z and Pokec_n do not have identical CSV columns.  The source feature
    names therefore define the frozen model schema.  At target inference,
    columns are reordered to that schema, source-only columns are filled with
    zero, and target-only columns are ignored.  Min/max statistics are fitted
    on the source only and retained as preprocessing state.

    Pokec edge files contain original user IDs rather than zero-based row
    indices, so each domain is independently re-indexed through its user_id
    column before its adjacency matrix is built.
    """
    if role not in {"source", "target"}:
        raise ValueError(f"role must be 'source' or 'target', got {role!r}")

    domain_path = os.path.join(path, dataset)
    csv_path = os.path.join(domain_path, f"{dataset}.csv")
    edge_path = os.path.join(domain_path, f"{dataset}_edges.txt")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Pokec domain CSV not found: {csv_path}")
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Pokec domain edge list not found: {edge_path}")

    frame = pd.read_csv(csv_path)
    required_columns = {"user_id", sens_attr, predict_attr}
    missing_required = sorted(required_columns - set(frame.columns))
    if missing_required:
        raise ValueError(
            f"required Pokec columns are missing from {csv_path}: "
            f"{missing_required}")

    if role == "source":
        feature_names = [
            name for name in frame.columns
            if name not in {"user_id", predict_attr}
        ]
        sens_idx = feature_names.index(sens_attr)
    else:
        if preprocessing_stats is None:
            raise ValueError(
                "target loading requires preprocessing metadata fitted on the source domain")
        if preprocessing_stats.get("dataset_family") != "pokec":
            raise ValueError("source preprocessing metadata is not for Pokec")
        feature_names = list(preprocessing_stats["feature_names"])
        sens_idx = preprocessing_stats["sens_idx"]
        if sens_attr not in feature_names:
            raise ValueError(
                f"source feature schema does not contain {sens_attr!r}")

    labels_array = pd.to_numeric(
        frame[predict_attr], errors="raise").to_numpy(dtype=np.int64)
    labels_array = labels_array.copy()
    labels_array[labels_array > 1] = 1

    sens_array = pd.to_numeric(
        frame[sens_attr], errors="raise").to_numpy(dtype=np.float32)
    sens_array = sens_array.copy()
    sens_array[sens_array > 0] = 1

    # reindex() fixes the column order and supplies zero for source-only
    # features that do not exist in Pokec_n. Existing NaNs are also filled.
    feature_frame = frame.reindex(columns=feature_names, fill_value=0).fillna(0)
    feature_frame = feature_frame.copy()
    feature_frame[sens_attr] = sens_array
    raw_features = torch.FloatTensor(
        feature_frame.to_numpy(dtype=np.float32))
    labels = torch.LongTensor(labels_array)
    sens = torch.FloatTensor(sens_array)

    if role == "source":
        feature_min = raw_features.min(dim=0)[0]
        feature_max = raw_features.max(dim=0)[0]
        preprocessing_stats = {
            "dataset_family": "pokec",
            "feature_names": tuple(feature_names),
            "feature_min": feature_min.clone(),
            "feature_max": feature_max.clone(),
            "sens_idx": sens_idx,
        }
    else:
        feature_min = preprocessing_stats["feature_min"].cpu()
        feature_max = preprocessing_stats["feature_max"].cpu()

    feature_range = feature_max - feature_min
    safe_range = torch.where(feature_range == 0,
                             torch.ones_like(feature_range), feature_range)
    features = 2 * (raw_features - feature_min) / safe_range - 1
    # As in get_dataset(), retain the sensitive channel in its original form.
    features[:, sens_idx] = raw_features[:, sens_idx]

    user_ids = pd.to_numeric(
        frame["user_id"], errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(user_ids)) != len(user_ids):
        raise ValueError(f"duplicate user_id values in {csv_path}")

    unordered_edges = _read_domain_edge_list(edge_path)

    id_to_row = pd.Series(np.arange(len(user_ids)), index=user_ids)
    mapped_edges = id_to_row.reindex(unordered_edges.reshape(-1))
    if mapped_edges.isna().any():
        missing_ids = np.unique(
            unordered_edges.reshape(-1)[mapped_edges.isna().to_numpy()])
        preview = missing_ids[:10].tolist()
        raise ValueError(
            f"{edge_path} contains user IDs absent from {csv_path}: "
            f"{preview}{' ...' if len(missing_ids) > 10 else ''}")
    edges = mapped_edges.to_numpy(dtype=np.int64).reshape(-1, 2)
    adj_norm_sp, edge_index = _build_domain_adjacency(
        edges, len(frame))

    label_mask = labels >= 0
    sens_mask = sens >= 0
    eligible_mask = label_mask & sens_mask
    if role == "source":
        train_mask, val_mask = _stratified_source_masks(
            labels, sens, val_ratio=source_val_ratio, seed=split_seed,
            eligible_mask=eligible_mask)
        test_mask = torch.zeros(len(frame), dtype=torch.bool)
    else:
        train_mask = torch.zeros(len(frame), dtype=torch.bool)
        val_mask = torch.zeros(len(frame), dtype=torch.bool)
        # All target nodes participate in message passing. Metrics are defined
        # only where both the task label and sensitive attribute are observed.
        test_mask = eligible_mask

    data = Data(
        x=features,
        edge_index=edge_index,
        adj_norm_sp=adj_norm_sp,
        y=labels.float(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        sens=sens,
        label_mask=label_mask,
        sens_mask=sens_mask,
    )
    return data, sens_idx, preprocessing_stats


def load_syn_domain(dataset, role, preprocessing_stats=None,
                    path="dataset/syn", source_val_ratio=0.2,
                    split_seed=20):
    """Load syn-2 for source training or syn-1 for frozen target testing.

    The sensitive attribute is appended as the last input channel, matching the
    existing synthetic-data path. Min/max statistics are fitted on the source
    only and reused unchanged for the target domain.
    """
    if role not in {"source", "target"}:
        raise ValueError(f"role must be 'source' or 'target', got {role!r}")

    feature_path = os.path.join(path, f"{dataset}_feat.csv")
    label_path = os.path.join(path, f"{dataset}_label.txt")
    sens_path = os.path.join(path, f"{dataset}_sens.txt")
    edge_path = os.path.join(path, f"{dataset}_edges.txt")
    required_paths = (feature_path, label_path, sens_path, edge_path)
    missing_paths = [file_path for file_path in required_paths
                     if not os.path.exists(file_path)]
    if missing_paths:
        raise FileNotFoundError(
            f"synthetic-domain files not found: {missing_paths}")

    feature_frame = pd.read_csv(feature_path, header=None).fillna(0)
    labels_array = pd.read_csv(
        label_path, header=None).to_numpy().reshape(-1).astype(np.int64)
    sens_array = pd.read_csv(
        sens_path, header=None).to_numpy().reshape(-1).astype(np.int64)
    if not (len(feature_frame) == len(labels_array) == len(sens_array)):
        raise ValueError(
            f"feature/label/sensitive row counts differ for {dataset}: "
            f"features={len(feature_frame)}, labels={len(labels_array)}, "
            f"sensitive={len(sens_array)}")

    labels_array = labels_array.copy()
    labels_array[labels_array > 0] = 1
    sens_array = sens_array.copy()
    sens_array[sens_array > 0] = 1

    base_features = torch.FloatTensor(
        feature_frame.to_numpy(dtype=np.float32))
    sens = torch.LongTensor(sens_array)
    labels = torch.LongTensor(labels_array)
    raw_features = torch.cat(
        [base_features, sens.float().unsqueeze(1)], dim=1)
    sens_idx = raw_features.shape[1] - 1

    if role == "source":
        feature_min = raw_features.min(dim=0)[0]
        feature_max = raw_features.max(dim=0)[0]
        preprocessing_stats = {
            "dataset_family": "syn",
            "base_feature_count": base_features.shape[1],
            "feature_min": feature_min.clone(),
            "feature_max": feature_max.clone(),
            "sens_idx": sens_idx,
        }
    else:
        if preprocessing_stats is None:
            raise ValueError(
                "target loading requires preprocessing metadata fitted on the source domain")
        if preprocessing_stats.get("dataset_family") != "syn":
            raise ValueError("source preprocessing metadata is not for synthetic data")
        expected_count = preprocessing_stats["base_feature_count"]
        if base_features.shape[1] != expected_count:
            raise ValueError(
                "source and target synthetic feature dimensions differ: "
                f"source={expected_count}, target={base_features.shape[1]}")
        if sens_idx != preprocessing_stats["sens_idx"]:
            raise ValueError("source and target sensitive-attribute indices differ")
        feature_min = preprocessing_stats["feature_min"].cpu()
        feature_max = preprocessing_stats["feature_max"].cpu()

    feature_range = feature_max - feature_min
    safe_range = torch.where(feature_range == 0,
                             torch.ones_like(feature_range), feature_range)
    features = 2 * (raw_features - feature_min) / safe_range - 1
    features[:, sens_idx] = raw_features[:, sens_idx]

    adj_norm_sp, edge_index = _load_domain_adjacency(
        edge_path, len(feature_frame))

    label_mask = labels >= 0
    sens_mask = sens >= 0
    eligible_mask = label_mask & sens_mask
    if role == "source":
        train_mask, val_mask = _stratified_source_masks(
            labels, sens, val_ratio=source_val_ratio, seed=split_seed,
            eligible_mask=eligible_mask)
        test_mask = torch.zeros(len(feature_frame), dtype=torch.bool)
    else:
        train_mask = torch.zeros(len(feature_frame), dtype=torch.bool)
        val_mask = torch.zeros(len(feature_frame), dtype=torch.bool)
        test_mask = eligible_mask

    data = Data(
        x=features,
        edge_index=edge_index,
        adj_norm_sp=adj_norm_sp,
        y=labels.float(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        sens=sens,
        label_mask=label_mask,
        sens_mask=sens_mask,
    )
    return data, sens_idx, preprocessing_stats

def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="dataset/german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender']
                                  == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    #    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
    #        val = idx_features_labels['PurposeOfLoan'].unique()[i]
    #        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

    #    # Normalize LoanAmount
    #    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
    #
    #    # Normalize Age
    #    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
    #
    #    # Normalize LoanDuration
    #    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
    #
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(
        label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(
        0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens

# 自己改的
def load_germanA(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="dataset/germanA/", label_number=1000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    # build relationship
    adj = load_npz(f'{path}/{dataset}_edges.npz')

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:int(0.5 * len(label_idx_0))],
                          label_idx_1[: int(0.5 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], 
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):],
                         label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens


# 自己加的（参考FairSIN）
def load_pokec_n(dataset, sens_attr="region", predict_attr="I_am_working_in_field", path="dataset/pokec/", label_number=3000):
    sens_number=500
    seed=20
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove(predict_attr)
    
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    # num_classes = len(idx_features_labels[predict_attr].unique()) - 1
    # labels = torch.eye(num_classes)[labels]
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels > 0)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:int(0.5 * len(label_idx_0))],
                          label_idx_1[: int(0.5 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], 
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):],
                         label_idx_1[int(0.75 * len(label_idx_1)):])


    sens = idx_features_labels[sens_attr].values
    sens_idx = set(np.where(sens >= 0)[0])
    sens = torch.FloatTensor(sens)
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    # pokec data division
    labels[labels>1]=1
    if sens_attr:
        sens[sens>0]=1
        
    from collections import Counter
    print('predict_attr:',Counter(idx_features_labels[predict_attr]))
    print('sens_attr:',Counter(idx_features_labels[sens_attr]))
    print('total dimension:', features.shape)
    # random.shuffle(sens_idx)

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens

# 自己加的
def load_syn_1(dataset, sens_attr="", predict_attr="", path="dataset/syn-1/", label_number=-1):
    features = pd.read_csv(os.path.join(path, "{}_feat.csv".format(dataset)), header=None)
    # features = torch.FloatTensor(features.values.astype(np.float32))

    labels = pd.read_csv(os.path.join(path, "{}_label.txt".format(dataset)), header=None)
    labels = torch.LongTensor(labels.values.astype(int).squeeze())

    sens_labels = pd.read_csv(os.path.join(path, "{}_sens.txt".format(dataset)), header=None)
    # sens_labels = torch.LongTensor(sens_labels.values.astype(int).squeeze())

    # 自己加的，适应外面的接口
    features = np.concatenate([features.values, sens_labels.values], axis=1)
    # features = torch.FloatTensor(features.values.astype(np.float32))  # 会报错
    features = torch.FloatTensor(features.astype(np.float32))
    sens_labels = torch.LongTensor(sens_labels.values.astype(int).squeeze())

    if os.path.exists(os.path.join(path, "{}_edges.txt".format(dataset))):
        edges_unordered = np.genfromtxt(os.path.join(path, "{}_edges.txt".format(dataset)), delimiter=',').astype('int')
    else:
        raise NotImplementedError

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.seed(20)  # 固定随机种子，保证每次的train, val, test一致
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:int(0.5 * len(label_idx_0))],
                          label_idx_1[: int(0.5 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], 
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens_labels

def get_dataset(dataname, top_k):
    if(dataname == 'credit'):
        load, label_num = load_credit, 6000
    elif(dataname == 'bail'):
        load, label_num = load_bail, 100
    elif(dataname == 'german'):
        load, label_num = load_german, 100
    # 自己加的
    elif(dataname == 'pokec_n'):
        load, label_num = load_pokec_n, 3000
    elif(dataname == 'bailA'):
        load, label_num = load_bailA, 100
    elif(dataname == 'germanA'):
        load, label_num = load_germanA, 100
    elif(dataname == 'syn-1'):
        load, label_num = load_syn_1, -1

    adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens = load(
        dataset=dataname, label_number=label_num)

    if(dataname == 'credit'):
        sens_idx = 1
    elif(dataname == 'bail' or dataname == 'german' or dataname == 'bailA' or dataname == 'germanA'):
        sens_idx = 0
    # 自己加的
    elif(dataname == 'pokec_n'):
        sens_idx = 3
    elif(dataname == 'syn-1'):
        sens_idx = features.shape[1] - 1

    x_max, x_min = torch.max(features, dim=0)[0], torch.min(features, dim=0)[0]

    if(dataname != 'german' and dataname != 'germanA'):
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features


    corr_matrix = sens_correlation(features, sens_idx)
    corr_idx = np.argsort(-np.abs(corr_matrix))
    if(top_k > 0):
        # corr_idx = np.concatenate((corr_idx[:top_k], corr_idx[-top_k:]))
        corr_idx = corr_idx[:top_k]

    return Data(x=features, edge_index=edge_index, adj_norm_sp=adj_norm_sp, y=labels.float(), train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, sens=sens), sens_idx, corr_matrix, corr_idx, x_min, x_max
