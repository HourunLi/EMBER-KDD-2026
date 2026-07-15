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

def get_dataset(dataname, top_k):
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
    

    adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens = load(
        dataset=dataname, label_number=label_num)

    if(dataname == 'credit'):
        sens_idx = 1
    elif(dataname == 'bail' or dataname == 'german' or dataname == 'bailA' or dataname == 'germanA'):
        sens_idx = 0
    elif(dataname == 'pokec_z' or dataname == 'pokec_n'):
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

    labels[labels > 1] = 1
    return Data(x=features, edge_index=edge_index, adj_norm_sp=adj_norm_sp, y=labels.float(), train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, sens=sens), sens_idx, corr_matrix, corr_idx, x_min, x_max
