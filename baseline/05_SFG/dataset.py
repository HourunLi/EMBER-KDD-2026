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


def feature_norm(features, min_values=None, max_values=None):
    """Min-max normalize features with optional externally supplied statistics.

    Cross-domain evaluation must fit preprocessing on the source domain only.
    Passing ``min_values`` and ``max_values`` therefore applies source-domain
    statistics to another domain without inspecting its feature distribution.
    Constant source features are mapped to zero to avoid division by zero.
    """
    if min_values is None:
        min_values = features.min(axis=0)[0]
    if max_values is None:
        max_values = features.max(axis=0)[0]

    feature_range = max_values - min_values
    constant_features = feature_range == 0
    safe_range = feature_range.clone()
    safe_range[constant_features] = 1

    normalized = 2 * (features - min_values).div(safe_range) - 1
    normalized[:, constant_features] = 0
    return normalized


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
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
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
def load_bailA(dataset, sens_attr="WHITE", predict_attr="RECID", path="dataset/bailA/",
               label_number=1000, split_mode="random", split_seed=20):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    if 'user_id' in header:
        # Domain CSVs retain the original row identifier for bookkeeping. It
        # is not a predictive attribute and must not be exposed to the model.
        header.remove('user_id')
    labels = idx_features_labels[predict_attr].values
    labels = torch.LongTensor(labels)
    sens_labels = idx_features_labels[sens_attr].values.astype(int)
    sens_labels = torch.LongTensor(sens_labels)
    features = idx_features_labels[header]
    features = torch.FloatTensor(np.array(features, dtype=np.float32))
    edge_npz_path = os.path.join(path, "{}_edges.npz".format(dataset))
    edge_txt_path = os.path.join(path, "{}_edges.txt".format(dataset))
    if os.path.exists(edge_npz_path):
        adj = load_npz(edge_npz_path)
    elif os.path.exists(edge_txt_path):
        edges = np.genfromtxt(edge_txt_path, dtype=np.int64)
        if edges.ndim == 1:
            edges = edges.reshape(1, -1)
        if edges.shape[1] != 2:
            raise ValueError(
                "Expected two integer columns in '{}', got shape {}."
                .format(edge_txt_path, edges.shape))
        if edges.min() < 0 or edges.max() >= features.shape[0]:
            raise ValueError(
                "Edge indices in '{}' are outside the node range [0, {})."
                .format(edge_txt_path, features.shape[0]))

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0], dtype=np.float32),
             (edges[:, 0], edges[:, 1])),
            shape=(features.shape[0], features.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    else:
        raise FileNotFoundError(
            "Neither '{}' nor '{}' exists.".format(edge_npz_path, edge_txt_path))

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    if split_mode == "random":
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        split_rng = np.random.RandomState(split_seed)
        split_rng.shuffle(label_idx_0)
        split_rng.shuffle(label_idx_1)
        idx_train = np.append(label_idx_0[:int(0.5 * len(label_idx_0))],
                              label_idx_1[:int(0.5 * len(label_idx_1))])
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):],
                             label_idx_1[int(0.75 * len(label_idx_1)):])
    elif split_mode == "full_test":
        # Target-domain labels must not participate in training or model
        # selection. The whole target graph is exposed only to final testing.
        idx_train = np.array([], dtype=np.int64)
        idx_val = np.array([], dtype=np.int64)
        idx_test = np.arange(features.shape[0], dtype=np.int64)
    elif split_mode == "full_train":
        # Source-only domain generalization: every labeled source node is used
        # by the supervised classifier objective. No source hold-out is kept.
        idx_train = np.arange(features.shape[0], dtype=np.int64)
        idx_val = np.array([], dtype=np.int64)
        idx_test = np.array([], dtype=np.int64)
    else:
        raise ValueError(
            "Unsupported bailA split_mode '{}'. Use 'random', 'full_train', "
            "or 'full_test'."
            .format(split_mode))

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))
    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens

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

def load_pokec(dataset, sens_attr="region", predict_attr="I_am_working_in_field", path="./dataset/pokec/", label_number=1000, sens_number=500,
               seed=20, split_ratio=None, val_idx=True):
    """Load data"""
    # print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    # labels[labels == -1] = 0

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    # delete edge connecting nodes with negative label
    # label_del_idx = np.where(labels < 0)[0]
    # del_target = np.isin(edges[:, 0], label_del_idx)
    # del_source = np.isin(edges[:, 1], label_del_idx)
    # del_edge = del_target + del_source
    # edges_new = edges[~del_edge]
    # adj = sp.coo_matrix((np.ones(edges_new.shape[0]), (edges_new[:, 0], edges_new[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)

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
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    # seed = 20
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    # print("label_num", label_idx.shape)
    # print("label=0", np.where(labels == 0)[0].shape)
    random.shuffle(label_idx)

    if split_ratio is None:
        split_ratio = [0.5, 0.25, 0.25]

    idx_train = label_idx[:min(int(split_ratio[0] * len(label_idx)), label_number)]
    if val_idx:
        idx_val = label_idx[int(split_ratio[0] * len(label_idx)):int((split_ratio[0] + split_ratio[1]) * len(label_idx))]
        idx_test = label_idx[int((split_ratio[0] + split_ratio[1]) * len(label_idx)):]
    else:
        idx_test = label_idx[label_number:]
        idx_val = idx_test

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    # random.shuffle(sens_idx)
    # a = torch.unique(labels)
    # return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train
    return adj, edge_index, features, labels, train_mask, val_mask, test_mask, sens

# 自己改的，对齐train val test
def load_pokec_n(dataset,sens_attr="region",predict_attr="I_am_working_in_field", path="dataset/pokec/", label_number=3000,sens_number=500, seed=20, split_ratio=None, val_idx=True):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
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

    # import random
    # random.seed(seed)
    # label_idx = np.where(labels>=0)[0]
    # random.shuffle(label_idx)

    # idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    # idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    # if test_idx:
    #     idx_test = label_idx[label_number:]
    #     idx_val = idx_test
    # else:
    #     idx_test = label_idx[int(0.75 * len(label_idx)):]

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

    return adj, edge_index, features, labels, train_mask, val_mask, test_mask, sens

# 自己改的
def load_syn_1(dataset,sens_attr="",predict_attr="", path="dataset/syn-1/", label_number=-1,sens_number=-1, seed=-1, split_ratio=None, val_idx=True):
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

    return adj, edge_index, features, labels, train_mask, val_mask, test_mask, sens_labels

POKEC_DOMAIN_EXCLUSIVE_FEATURES = {
    'zberatelstvo',
    'hackovanie',
    'vtacik',
    'plave',
    'niekto',
    'slobodny',
    'alternativne',
    'alternativa',
    'horolezectvo',
    'bezkovanie',
    'surfing',
    'literaturu o umeni a architekture',
    'madarsky',
}


def build_local_edge_graph(edge_path, num_nodes, delimiter=None,
                           add_self_loops=True):
    raw_edges = np.genfromtxt(edge_path, delimiter=delimiter)
    if raw_edges.ndim == 1:
        raw_edges = raw_edges.reshape(1, -1)
    if raw_edges.shape[1] != 2:
        raise ValueError(
            "Expected two edge columns in '{}', got shape {}."
            .format(edge_path, raw_edges.shape))

    edges = raw_edges.astype(np.int64)
    if not np.allclose(raw_edges, edges):
        raise ValueError("Non-integer node identifiers found in '{}'.".format(
            edge_path))
    if edges.min() < 0 or edges.max() >= num_nodes:
        raise ValueError(
            "Edge indices in '{}' are outside the node range [0, {})."
            .format(edge_path, num_nodes))
    return build_sparse_graph(edges, num_nodes, add_self_loops)


def build_id_mapped_graph(edge_path, node_ids, add_self_loops=True):
    raw_edges = np.genfromtxt(edge_path, dtype=np.int64)
    if raw_edges.ndim == 1:
        raw_edges = raw_edges.reshape(1, -1)
    if raw_edges.shape[1] != 2:
        raise ValueError(
            "Expected two edge columns in '{}', got shape {}."
            .format(edge_path, raw_edges.shape))

    node_id_map = {int(node_id): index
                   for index, node_id in enumerate(node_ids)}
    flattened_edges = raw_edges.reshape(-1)
    mapped_edges = np.fromiter(
        (node_id_map.get(int(node_id), -1) for node_id in flattened_edges),
        dtype=np.int64, count=flattened_edges.size).reshape(raw_edges.shape)
    if np.any(mapped_edges < 0):
        missing_ids = np.unique(raw_edges[mapped_edges < 0])
        raise ValueError(
            "Edges in '{}' reference {} node IDs absent from the CSV, "
            "for example {}.".format(
                edge_path, missing_ids.size, missing_ids[:5].tolist()))
    return build_sparse_graph(mapped_edges, len(node_ids), add_self_loops)


def build_sparse_graph(edges, num_nodes, add_self_loops):
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0], dtype=np.float32),
         (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if add_self_loops:
        adj = adj + sp.eye(num_nodes, dtype=np.float32)

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)
    return adj_norm_sp, edge_index


def load_germanA_domain(dataset, path="dataset/germanA/"):
    csv_path = os.path.join(path, "{}.csv".format(dataset))
    edge_path = os.path.join(path, "{}_edges.txt".format(dataset))
    frame = pd.read_csv(csv_path)

    feature_names = list(frame.columns)
    for excluded_column in (
            'user_id', 'GoodCustomer', 'OtherLoansAtStore', 'PurposeOfLoan'):
        if excluded_column in feature_names:
            feature_names.remove(excluded_column)

    frame.loc[frame['Gender'] == 'Female', 'Gender'] = 1
    frame.loc[frame['Gender'] == 'Male', 'Gender'] = 0
    features = torch.FloatTensor(
        frame[feature_names].to_numpy(dtype=np.float32))

    labels = frame['GoodCustomer'].to_numpy(dtype=np.int64)
    labels[labels == -1] = 0
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(frame['Gender'].to_numpy(dtype=np.int64))

    adj_norm_sp, edge_index = build_local_edge_graph(
        edge_path, features.shape[0], delimiter=None, add_self_loops=True)
    return adj_norm_sp, edge_index, features, labels, sens, feature_names, 'Gender'


def load_pokec_domain(dataset):
    path = os.path.join("dataset", dataset)
    csv_path = os.path.join(path, "{}.csv".format(dataset))
    edge_path = os.path.join(path, "{}_edges.txt".format(dataset))
    frame = pd.read_csv(csv_path)

    excluded_columns = {
        'user_id',
        'I_am_working_in_field',
    } | POKEC_DOMAIN_EXCLUSIVE_FEATURES
    feature_names = sorted([
        column for column in frame.columns
        if column not in excluded_columns
    ])
    if len(feature_names) != 265:
        raise ValueError(
            "Expected 265 aligned Pokec features for '{}', found {}."
            .format(dataset, len(feature_names)))

    features = torch.FloatTensor(
        frame[feature_names].to_numpy(dtype=np.float32))
    labels = torch.LongTensor(
        frame['I_am_working_in_field'].to_numpy(dtype=np.int64))
    sens_values = frame['region'].to_numpy(dtype=np.int64)
    sens_values[sens_values > 0] = 1
    sens = torch.LongTensor(sens_values)

    node_ids = frame['user_id'].to_numpy(dtype=np.int64)
    adj_norm_sp, edge_index = build_id_mapped_graph(
        edge_path, node_ids, add_self_loops=True)
    return adj_norm_sp, edge_index, features, labels, sens, feature_names, 'region'


def load_syn_domain(dataset, path="dataset/syn/"):
    feature_path = os.path.join(path, "{}_feat.csv".format(dataset))
    label_path = os.path.join(path, "{}_label.txt".format(dataset))
    sens_path = os.path.join(path, "{}_sens.txt".format(dataset))
    edge_path = os.path.join(path, "{}_edges.txt".format(dataset))

    raw_features = pd.read_csv(feature_path, header=None).to_numpy(
        dtype=np.float32)
    labels = torch.LongTensor(
        pd.read_csv(label_path, header=None).to_numpy(
            dtype=np.int64).reshape(-1))
    sens = torch.LongTensor(
        pd.read_csv(sens_path, header=None).to_numpy(
            dtype=np.int64).reshape(-1))
    features = torch.FloatTensor(np.concatenate([
        raw_features,
        sens.numpy().reshape(-1, 1).astype(np.float32),
    ], axis=1))
    feature_names = [
        'feature_{}'.format(index) for index in range(raw_features.shape[1])
    ] + ['sensitive_attribute']

    adj_norm_sp, edge_index = build_local_edge_graph(
        edge_path, features.shape[0], delimiter=',', add_self_loops=True)
    return (
        adj_norm_sp, edge_index, features, labels, sens, feature_names,
        'sensitive_attribute')


def get_domain_dataset(dataset_family, domain_name, split_mode, top_k):
    if dataset_family == 'bailA':
        loaded = load_bailA(
            domain_name, split_mode=split_mode)
        adj_norm_sp, edge_index, features, labels, _, _, _, sens = loaded
        feature_names = list(pd.read_csv(
            os.path.join("dataset/bailA", "{}.csv".format(domain_name)),
            nrows=0).columns)
        feature_names.remove('RECID')
        if 'user_id' in feature_names:
            feature_names.remove('user_id')
        sensitive_name = 'WHITE'
        normalize_features = True
    elif dataset_family == 'germanA':
        (adj_norm_sp, edge_index, features, labels, sens, feature_names,
         sensitive_name) = load_germanA_domain(domain_name)
        normalize_features = False
    elif dataset_family == 'pokec':
        (adj_norm_sp, edge_index, features, labels, sens, feature_names,
         sensitive_name) = load_pokec_domain(domain_name)
        normalize_features = True
    elif dataset_family == 'syn':
        (adj_norm_sp, edge_index, features, labels, sens, feature_names,
         sensitive_name) = load_syn_domain(domain_name)
        normalize_features = True
    else:
        raise ValueError(
            "Unsupported cross-domain dataset family '{}'."
            .format(dataset_family))

    if features.shape[1] != len(feature_names):
        raise ValueError(
            "Feature tensor/column mismatch for '{}': {} versus {}."
            .format(domain_name, features.shape[1], len(feature_names)))
    sens_idx = feature_names.index(sensitive_name)

    labels = labels.clone()
    labels[labels > 1] = 1
    sens = sens.clone()
    sens[sens > 1] = 1
    valid_label_mask = torch.logical_and(labels >= 0, labels <= 1)
    valid_sensitive_mask = torch.logical_and(sens >= 0, sens <= 1)

    empty_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    if split_mode == 'full_train':
        train_mask = valid_label_mask.clone()
        val_mask = empty_mask.clone()
        test_mask = empty_mask.clone()
    elif split_mode == 'full_test':
        train_mask = empty_mask.clone()
        val_mask = empty_mask.clone()
        test_mask = torch.logical_and(
            valid_label_mask, valid_sensitive_mask)
    else:
        raise ValueError(
            "Cross-domain split_mode must be 'full_train' or 'full_test'.")

    x_max = torch.max(features, dim=0)[0]
    x_min = torch.min(features, dim=0)[0]
    if normalize_features:
        features = feature_norm(features, x_min, x_max)
        features[:, sens_idx] = sens.to(features.dtype)

    corr_matrix = sens_correlation(features, sens_idx)
    corr_idx = np.argsort(-np.abs(corr_matrix))
    if top_k > 0:
        corr_idx = corr_idx[:top_k]

    data = Data(
        x=features,
        edge_index=edge_index,
        adj_norm_sp=adj_norm_sp,
        y=labels.float(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        sens=sens,
        valid_label_mask=valid_label_mask,
        valid_sensitive_mask=valid_sensitive_mask,
    )
    data.domain_name = domain_name
    data.dataset_family = dataset_family
    data.feature_names = feature_names
    return data, sens_idx, corr_matrix, corr_idx, x_min, x_max


def get_dataset(dataname, top_k, dataset_name=None, split_mode="random",
                split_seed=20, normalization_stats=None):
    if(dataname == 'credit'):
        load, label_num = load_credit, 6000
    elif(dataname == 'bail'):
        load, label_num = load_bail, 100
    elif(dataname == 'german'):
        load, label_num = load_german, 100
    elif(dataname == 'pokec_z'):
        load, label_num = load_pokec, 4000
    elif(dataname == 'pokec_n'):
        load, label_num = load_pokec_n, 3500
    # 自己改的
    elif(dataname == 'bailA'):
        load, label_num = load_bailA, 100
    elif(dataname == 'germanA'):
        load, label_num = load_germanA, 100
    elif(dataname == 'syn-1'):
        load, label_num = load_syn_1, -1
    

    dataset_name = dataname if dataset_name is None else dataset_name
    load_kwargs = dict(dataset=dataset_name, label_number=label_num)
    if load is load_bailA:
        load_kwargs.update(split_mode=split_mode, split_seed=split_seed)
    elif split_mode != "random":
        raise ValueError(
            "split_mode='{}' is currently supported only for bailA domains."
            .format(split_mode))

    adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens = load(
        **load_kwargs)

    feature_names = None
    if dataname == 'bailA':
        feature_names = list(pd.read_csv(
            os.path.join("dataset/bailA/", "{}.csv".format(dataset_name)),
            nrows=0).columns)
        feature_names.remove('RECID')
        if 'user_id' in feature_names:
            feature_names.remove('user_id')
        sens_idx = feature_names.index('WHITE')
    elif(dataname == 'credit'):
        sens_idx = 1
    elif(dataname == 'bail' or dataname == 'german' or dataname == 'germanA'):
        sens_idx = 0
    elif(dataname == 'pokec_z' or dataname == 'pokec_n'):
        sens_idx = 3
    elif(dataname == 'syn-1'):
        sens_idx = features.shape[1] - 1

    x_max, x_min = torch.max(features, dim=0)[0], torch.min(features, dim=0)[0]

    if(dataname != 'german' and dataname != 'germanA'):
        if normalization_stats is None:
            norm_min, norm_max = x_min, x_max
        else:
            norm_min, norm_max = normalization_stats
            if norm_min.shape[0] != features.shape[1] or norm_max.shape[0] != features.shape[1]:
                raise ValueError(
                    "Source and target domains have different feature dimensions: "
                    "normalization statistics contain {} features, but '{}' contains {}."
                    .format(norm_min.shape[0], dataset_name, features.shape[1]))

        norm_features = feature_norm(features, norm_min, norm_max)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    corr_matrix = sens_correlation(features, sens_idx)
    corr_idx = np.argsort(-np.abs(corr_matrix))
    if(top_k > 0):
        # corr_idx = np.concatenate((corr_idx[:top_k], corr_idx[-top_k:]))
        corr_idx = corr_idx[:top_k]

    labels[labels > 1] = 1
    data = Data(x=features, edge_index=edge_index, adj_norm_sp=adj_norm_sp,
                y=labels.float(), train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask, sens=sens)
    data.domain_name = dataset_name

    if dataname == 'bailA':
        data.feature_names = feature_names

    return data, sens_idx, corr_matrix, corr_idx, x_min, x_max
