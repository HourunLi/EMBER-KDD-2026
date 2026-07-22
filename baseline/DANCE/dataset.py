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
from torch_scatter import scatter_add
from neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist

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
    # print("Z-Score Standardized Data:")
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    standardized_data = (features - mean) / std
    return standardized_data

    # print("Min-Max Normalized Data:")
    # min_values = features.min(axis=0, keepdim=True).values
    # max_values = features.max(axis=0, keepdim=True).values
    # return 2 * (features - min_values).div(max_values - min_values) - 1
    # return (features - min_values) / (max_values - min_values)
    

# x: A data matrix where each column represents a feature or point.
# thresh: A threshold (default is 0.25) used to determine which pairs of points should be connected based on their similarity.
# 按照和第二相似度的25%作为threshold，大于这个值的就连边
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


def load_credit(dataset, id, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="../dataset/credit/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    path = f"/home/disk2/lhr/fairDomainAdaption/mine/dataset/{dataset}"
    if len(dataset) > 13:
        idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset[:-25])))
    else:
        idx_features_labels = pd.read_csv(os.path.join(path, f"{dataset}{id}.csv"))
    if 'Unnamed: 0' in idx_features_labels.columns:
        idx_features_labels = idx_features_labels.drop(['Unnamed: 0'], axis=1)
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')
    header.remove('user_id')
    n_cls = len(header)
    # header: ['Married', 'Age', 'EducationLevel', 'MaxBillAmountOverLast6Months', 'MaxPaymentAmountOverLast6Months', 'MonthsWithZeroBalanceOverLast6Months', 'MonthsWithLowSpendingOverLast6Months', 'MonthsWithHighSpendingOverLast6Months', 'MostRecentBillAmount', 'MostRecentPaymentAmount', 'TotalOverdueCounts', 'TotalMonthsOverdue', 'HistoryOfOverduePayments']
    # print(f"header: {header}")
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
    if os.path.exists(f'{path}/{dataset}{id}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}{id}_edges.txt').astype('int')
    else:
        raise NotImplementedError

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # print(f"features: {features}")
    labels = idx_features_labels[predict_attr].values

    # 行数
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    # print(f"idx_map: {idx_map}")
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

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:int(0.8 * len(label_idx_0))],
                          label_idx_1[:int(0.8 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.8 * len(label_idx_0)):int(0.9 * len(label_idx_0))],
                        label_idx_1[int(0.8 * len(label_idx_1)):int(0.9 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.9 * len(label_idx_0)):], label_idx_1[int(0.9 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, n_cls

def load_bail(dataset, id, sens_attr="WHITE", predict_attr="RECID", path="../dataset/bail/"):
    # print('Loading {} dataset from {}'.format(dataset, path))
    path = f"/home/disk2/lhr/fairDomainAdaption/mine/dataset/{dataset}"
    if len(dataset) > 9:
        idx_features_labels = pd.read_csv(
            os.path.join(path, "{}.csv".format(dataset[:-30])))
    else:
        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset + id)))
    if 'Unnamed: 0' in idx_features_labels.columns:
        idx_features_labels.drop(['Unnamed: 0'], axis=1)
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove("user_id")
    n_cls = len(header)
    
    # build relationship
    if os.path.exists(f'{path}/{dataset}{id}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}{id}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

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

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    
    idx_train = np.append(label_idx_0[:int(0.8 * len(label_idx_0))],
                          label_idx_1[: int(0.8 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.8 * len(label_idx_0)):int(0.9 * len(label_idx_0))], 
                        label_idx_1[int(0.8 * len(label_idx_1)):int(0.9 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.9 * len(label_idx_0)):], label_idx_1[int(0.9 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, n_cls


def load_german(dataset, id, sens_attr="Gender", predict_attr="GoodCustomer", path="dataset/german", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    path = f"/home/disk2/lhr/fairDomainAdaption/mine/dataset/{dataset}"
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset + id)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('user_id')
    header.remove(sens_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')
    n_cls = len(header)

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
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
        edges_unordered = np.genfromtxt(f'{path}/{dataset}{id}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}{id}_edges.txt', edges_unordered)


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

    idx_train = np.append(label_idx_0[:int(0.8 * len(label_idx_0))],
                          label_idx_1[:int(0.8 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.8 * len(label_idx_0)):int(0.9 * len(label_idx_0))],
                        label_idx_1[int(0.8 * len(label_idx_1)):int(0.9 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.9 * len(label_idx_0)):], label_idx_1[int(0.9 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, n_cls


def load_pokec(dataset, sens_attr, predict_attr, path="dataset/pokec/", label_number=1000, sens_number=500, seed=19,
               test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    if 'Unnamed: 0' in idx_features_labels.columns:
        idx_features_labels.drop(['Unnamed:0'], axis=1)
    header = list(pd.read_csv(os.path.join(path, "{}.csv".format("region_job_z"))).columns)
    header2 = list(pd.read_csv(os.path.join(path, "{}.csv".format("region_job_n"))).columns)
    header = [i for i in header if i in header2]
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)
    n_cls = len(header)
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset, )), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                      dtype=int).reshape(edges_unordered.shape)
    # edges = edges_unordered
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
    labels[labels > 1] = 1
    # labels[labels < 1] = 0

    import random
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]  # 找到label有效的集合
    random.shuffle(label_idx)

    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:int(0.8 * len(label_idx_0))],
                          label_idx_1[:int(0.8 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.8 * len(label_idx_0)):int(0.9 * len(label_idx_0))],
                        label_idx_1[int(0.8 * len(label_idx_1)):int(0.9 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.9 * len(label_idx_0)):], label_idx_1[int(0.9 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)

    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    # random.shuffle(sens_idx)

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, n_cls


def load_syn(dataset, id, path="/home/disk2/lhr/fairDomainAdaption/mine/dataset/syn"):
    features = pd.read_csv(os.path.join(path, "{}_feat.csv".format(dataset + id)), header=None)
    features = torch.FloatTensor(features.values.astype(np.float32))

    labels = pd.read_csv(os.path.join(path, "{}_label.txt".format(dataset + id)), header=None)
    labels = torch.LongTensor(labels.values.astype(int).squeeze())

    sens_labels = pd.read_csv(os.path.join(path, "{}_sens.txt".format(dataset + id)), header=None)
    sens_labels = torch.LongTensor(sens_labels.values.astype(int).squeeze())

    if os.path.exists(os.path.join(path, "{}_edges.txt".format(dataset + id))):
        edges_unordered = np.genfromtxt(os.path.join(path, "{}_edges.txt".format(dataset + id)), delimiter=',').astype('int')
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
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:int(0.8 * len(label_idx_0))],
                          label_idx_1[: int(0.8 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.8 * len(label_idx_0)):int(0.9 * len(label_idx_0))], 
                        label_idx_1[int(0.8 * len(label_idx_1)):int(0.9 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.9 * len(label_idx_0)):], label_idx_1[int(0.9 * len(label_idx_1)):])
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    n_cls = -1
    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens_labels, n_cls




# 表明训练节点从属哪个类别
def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label))
    idx_info = []
    n_data = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
        n_data.append(len(cls_indices))
    return idx_info, n_data

# 其实没有太多作用，因为一共也只有两个类别
# 在处理长尾数据时，通常选择的是通过增补少数类别的数据、调整训练策略或使用不同的模型架构来优化分类性能，而不是简单地去除长尾类别的数据。
# 这种方法可以更好地捕捉到各个类别之间的关系，避免信息的丢失，从而在类别不平衡的情况下提升模型的有效性和鲁棒性。
# edge_index: The edges of the graph represented in a sparse format.
# label: Class labels for each node in the graph.
# n_data: Number of nodes in each class.
# n_cls: Total number of classes.
# ratio: The long-tail ratio for imbalanced data.
# train_mask: A mask indicating which nodes are part of the training set.
def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    # Sort from major to minor
    n_data = torch.tensor(n_data)
    # The function sorts the number of nodes in descending order and prepares an index mapping to track the original order.
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1: # We does not remove any nodes of the most frequent class
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    # n_round 在 make_longtailed_data_remove 函数中起到了控制节点移除步骤的作用。
    # 通过分轮次进行节点移除，函数可以更细致地处理类别不平衡问题，确保在保持多数类别样本的同时，合理减少少数类别的样本数量。这种方法有助于提升模型在类别不平衡场景下的表现。
    
    # Compute the number of nodes which would be removed for each class
    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    # 移除的idx列表
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    # 移除之后mask会发生变化
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            # Compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            # Accumulation does not be problem since
            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask


def get_dataset(args, inid):
    dataname, top_k = args.dataset, args.top_k
    # load data
    if('credit' in dataname):
        load, label_num = load_credit, 6000
        adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, n_cls= load(dataset=dataname, id = inid, label_number=label_num)
    elif('bail' in dataname):
        adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, n_cls= load_bail(dataset=dataname, id = inid)
    elif('pokec' in dataname):
        adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, n_cls = load_pokec(dataset='region_job'+inid, sens_attr="region", 
                                                                                                      predict_attr="I_am_working_in_field", path="../dataset/pokec/",
                                                                                                      label_number=500, sens_number=200, seed=20, test_idx=False)
    elif ('syn' in dataname):
        load, label_num = load_syn, 1000
        adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, n_cls= load(dataset=dataname, id = inid)
    elif ('german' in dataname):
        load, label_num = load_german, 1000
        adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, n_cls= load(dataset=dataname, id = inid)

    if('credit' in dataname):
        sens_idx = 1
    elif('bail' in dataname):
        sens_idx = 0
    elif ('pokec' in dataname):
        sens_idx = 3
    else:
        sens_idx = 0

    x_max, x_min = torch.max(features, dim=0)[0], torch.min(features, dim=0)[0]
    norm_features = feature_norm(features)
    if(dataname != 'german' and dataname != 'syn') :
        norm_features[:, sens_idx] = features[:, sens_idx]
    features = norm_features


    # print(features[:5])
    # 提取出敏感特征和其他特征的相关性
    corr_matrix = sens_correlation(features, sens_idx)
    corr_idx = np.argsort(-np.abs(corr_matrix))
    #  If top_k is specified (greater than 0), the corr_idx array is truncated to keep only the indices of the top k most correlated features.
    if(top_k > 0):
        corr_idx = corr_idx[:top_k]

    # idx_info, n_data = get_idx_info(labels, n_cls, train_mask)
    # class_num_list, args.train_mask, args.idx_info, args.node_mask, args.edge_mask = make_longtailed_data_remove(edge_index, labels, n_data, n_cls, args.imb_ratio, train_mask.clone())
    # data = Data(x = features, edge_index = edge_index, adj_norm_sp = adj_norm_sp, y = labels.float(), train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, sens=sens)
    data = Data(x = features, edge_index = edge_index, adj_norm_sp = adj_norm_sp, y = labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, sens=sens)
    return data, sens_idx, corr_matrix, corr_idx, x_min, x_max
    # return data, sens_idx, corr_matrix, corr_idx, x_min, x_max, class_num_list, idx_info

def get_data2(args, data):
    data2 = []
    if args.ood == 1: # syn dataset
        if args.dataset == "bail":
            if args.outid == "_md0":
                args.strlist = ['_0.56_0.35_0.54_0.25_0.06_0.56', '_0.51_0.32_0.49_0.24_0.00_0.56', '_0.58_0.36_0.56_0.26_0.10_0.56',
                                '_0.49_0.30_0.47_0.25_0.00_0.56', '_0.63_0.39_0.61_0.30_0.20_0.56', '_0.45_0.27_0.43_0.29_0.00_0.56',
                                '_0.67_0.42_0.65_0.36_0.29_0.56', '_0.41_0.25_0.40_0.33_0.00_0.56', '_0.72_0.45_0.70_0.43_0.39_0.56',
                                '_0.37_0.23_0.37_0.37_0.00_0.44', '_0.76_0.48_0.74_0.51_0.48_0.56', '_0.34_0.21_0.34_0.41_0.00_0.44',
                                '_0.81_0.51_0.79_0.59_0.58_0.56', '_0.31_0.19_0.32_0.44_0.00_0.44', '_0.86_0.54_0.84_0.68_0.68_0.56',
                                '_0.29_0.17_0.30_0.47_0.00_0.44', '_0.90_0.57_0.89_0.78_0.79_0.56', '_0.27_0.16_0.28_0.50_0.00_0.44',
                                '_0.95_0.60_0.94_0.89_0.89_0.56', '_0.25_0.15_0.27_0.52_0.00_0.44', '_0.64_0.40_0.62_0.32_0.22_0.56',
                                '_0.43_0.26_0.42_0.30_0.00_0.56', '_0.69_0.43_0.67_0.38_0.32_0.56', '_0.40_0.24_0.39_0.34_0.00_0.44',
                                '_0.73_0.46_0.71_0.45_0.42_0.56', '_0.36_0.22_0.36_0.38_0.00_0.44', '_0.78_0.49_0.76_0.53_0.51_0.56',
                                '_0.33_0.20_0.34_0.42_0.00_0.44', '_0.82_0.52_0.80_0.62_0.61_0.56', '_0.30_0.18_0.31_0.45_0.00_0.44',
                                '_0.87_0.55_0.86_0.71_0.71_0.56', '_0.28_0.17_0.30_0.48_0.00_0.44', '_0.92_0.58_0.91_0.81_0.82_0.56',
                                '_0.26_0.16_0.28_0.51_0.00_0.44', '_0.97_0.61_0.96_0.92_0.92_0.56', '_0.24_0.14_0.27_0.53_0.00_0.44',
                                '_0.61_0.38_0.59_0.28_0.15_0.56', '_0.47_0.29_0.45_0.27_0.00_0.56', '_0.65_0.41_0.63_0.33_0.24_0.56',
                                '_0.43_0.26_0.41_0.31_0.00_0.56', '_0.70_0.43_0.68_0.39_0.34_0.56', '_0.39_0.24_0.38_0.35_0.00_0.44',
                                '_0.74_0.46_0.72_0.47_0.43_0.56', '_0.36_0.22_0.35_0.39_0.00_0.44', '_0.79_0.49_0.77_0.55_0.53_0.56',
                                '_0.33_0.20_0.33_0.43_0.00_0.44', '_0.83_0.52_0.81_0.64_0.63_0.56', '_0.30_0.18_0.31_0.46_0.00_0.44',
                                '_0.88_0.56_0.86_0.73_0.73_0.56', '_0.28_0.17_0.29_0.48_0.00_0.44', '_0.93_0.59_0.92_0.84_0.84_0.56',
                                '_0.26_0.15_0.28_0.51_0.00_0.44', '_0.98_0.62_0.97_0.94_0.94_0.56', '_0.24_0.14_0.26_0.53_0.00_0.44']
            elif args.outid == "_md3":
                args.strlist = ['_0.60_0.30_0.60_0.25_0.18_0.48', '_0.46_0.23_0.46_0.23_0.00_0.48', '_0.65_0.32_0.64_0.32_0.27_0.48',
                                '_0.42_0.21_0.42_0.28_0.00_0.48', '_0.69_0.35_0.69_0.40_0.37_0.48', '_0.38_0.19_0.39_0.32_0.00_0.48',
                                '_0.74_0.37_0.74_0.48_0.46_0.48', '_0.35_0.17_0.36_0.36_0.00_0.48', '_0.79_0.39_0.78_0.57_0.56_0.48',
                                '_0.32_0.16_0.34_0.40_0.00_0.48', '_0.83_0.42_0.83_0.66_0.66_0.48', '_0.30_0.15_0.32_0.44_0.00_0.48',
                                '_0.88_0.44_0.88_0.76_0.75_0.48', '_0.27_0.13_0.30_0.47_0.00_0.48', '_0.93_0.47_0.93_0.85_0.85_0.48',
                                '_0.25_0.12_0.28_0.49_0.00_0.48', '_0.98_0.49_0.98_0.95_0.95_0.48', '_0.24_0.12_0.27_0.52_0.00_0.48',
                                '_0.56_0.28_0.55_0.21_0.09_0.48', '_0.51_0.25_0.50_0.20_0.02_0.48', '_0.58_0.29_0.57_0.23_0.13_0.48',
                                '_0.49_0.24_0.48_0.21_0.00_0.48', '_0.62_0.31_0.62_0.28_0.23_0.48', '_0.44_0.22_0.44_0.26_0.00_0.48',
                                '_0.67_0.34_0.67_0.36_0.32_0.48', '_0.40_0.20_0.40_0.30_0.00_0.48', '_0.72_0.36_0.72_0.44_0.41_0.48',
                                '_0.37_0.18_0.37_0.34_0.00_0.48', '_0.76_0.38_0.76_0.53_0.51_0.48', '_0.34_0.17_0.35_0.38_0.00_0.48',
                                '_0.81_0.41_0.81_0.62_0.61_0.48', '_0.31_0.15_0.33_0.42_0.00_0.48', '_0.86_0.43_0.85_0.71_0.70_0.48',
                                '_0.29_0.14_0.31_0.45_0.00_0.48', '_0.90_0.45_0.90_0.80_0.80_0.48', '_0.26_0.13_0.29_0.48_0.00_0.48',
                                '_0.95_0.48_0.95_0.90_0.90_0.48', '_0.24_0.12_0.27_0.51_0.00_0.48', '_0.64_0.32_0.64_0.31_0.25_0.48',
                                '_0.43_0.21_0.43_0.27_0.00_0.48', '_0.68_0.34_0.68_0.38_0.35_0.48', '_0.39_0.19_0.39_0.32_0.00_0.48',
                                '_0.73_0.37_0.73_0.46_0.44_0.48', '_0.36_0.18_0.37_0.36_0.00_0.48', '_0.78_0.39_0.78_0.55_0.54_0.48',
                                '_0.33_0.16_0.34_0.40_0.00_0.48', '_0.82_0.41_0.82_0.64_0.64_0.48', '_0.30_0.15_0.32_0.43_0.00_0.48',
                                '_0.87_0.44_0.87_0.74_0.73_0.48', '_0.28_0.14_0.30_0.46_0.00_0.48', '_0.92_0.46_0.92_0.83_0.83_0.48',
                                '_0.26_0.13_0.28_0.49_0.00_0.48', '_0.97_0.49_0.97_0.93_0.93_0.48', '_0.24_0.12_0.27_0.51_0.00_0.48']

        data2 = None
        args.in_hom = [0 for i in range(len(args.strlist))]
        args.edge_hom = [0 for i in range(len(args.strlist))]
        args.node_hom = [0 for i in range(len(args.strlist))]
        args.class_hom = [0 for i in range(len(args.strlist))]
        args.agg_hom = [0 for i in range(len(args.strlist))]
        
    elif args.ood == 2: # real dataset
        if args.dataset == "credit":
            args.strlist = ['_C1', '_C2', '_C3', '_C4']
            for i in range(len(args.strlist)):
                datatmp, _, _, _, _, _ = get_dataset(args, args.strlist[i])
                data2.append(datatmp)
        elif args.dataset == "creditA":
            args.strlist = ['_1', '_2']
            for i in range(len(args.strlist)):
                datatmp, _, _, _, _, _ = get_dataset(
                    args,  args.strlist[i])
                data2.append(datatmp)
        elif args.dataset == "bail":
            args.strlist = ['_B1',  '_B2', '_B3', '_B4',]
            for i in range(len(args.strlist)):
                datatmp, _, _, _, _, _ = get_dataset(args, args.strlist[i])
                data2.append(datatmp)
        elif args.dataset == "bailA":
            args.strlist = ['_1',  '_2']
            for i in range(len(args.strlist)):
                datatmp, _, _, _, _, _ = get_dataset(
                    args,  args.strlist[i])
                data2.append(datatmp)     
        elif args.dataset == "pokec":
            args.strlist = ['_z', '_n',]
            args.inidIndex = args.strlist.index(args.inid)
            for i in range(len(args.strlist)):
                if args.inidIndex == i:
                    data2.append(data)
                    continue
                datatmp, _, _, _, _, _ = get_dataset(args, args.strlist[i])
                data2.append(datatmp)
        elif args.dataset == "german":
            args.strlist = ['_1',  '_2']
            for i in range(len(args.strlist)):
                datatmp, _, _, _, _, _ = get_dataset(
                    args,  args.strlist[i])
                data2.append(datatmp)    
        elif args.dataset == "syn":
            args.strlist = ['-1',  '-2']
            for i in range(len(args.strlist)):
                datatmp, _, _, _, _, _ = get_dataset(
                    args,  args.strlist[i])
                data2.append(datatmp)      
    else:
        data2 = None
    return data2


def process_dataset(args, data, data2):
    # print(sum(data.y==-1))
    args.num_features, args.num_classes = data.x.shape[1], len(data.y.unique())-1
    # if args.dataset == "pokec":
    args.num_classes = 1
    # args.train_ratio, args.val_ratio = torch.tensor([(data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), \
    #                                    torch.tensor([(data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    # args.train_ratio, args.val_ratio = torch.max(args.train_ratio) / args.train_ratio, \
    #                                    torch.max(args.val_ratio) / args.val_ratio
    # args.train_ratio, args.val_ratio = args.train_ratio[data.y[data.train_mask].long()], \
    #                                    args.val_ratio[data.y[data.val_mask].long()]

    # n_cls = data.y.max().item() + 1
    # idx_info, n_data = get_idx_info(data.y, n_cls, data.train_mask)
    # args.class_num_list = n_data
    # args.data_train_mask = data.train_mask
    # args.idx_info = idx_info
    # args.class_num_list, args.data_train_mask, args.idx_info, args.train_node_mask, args.train_edge_mask = make_longtailed_data_remove(data.edge_index, data.y, n_data, n_cls, args.imb_ratio, data.train_mask.clone())
    
    # if args.gdc=='ppr':
        # args.neighbor_dist_list = get_PPR_adj(data.x, data.edge_index, alpha=0.05, k=128, eps=None)
    # elif args.gdc=='hk':
    #     args.neighbor_dist_list = get_heat_adj(data.x, data.edge_index, t=5.0, k=None, eps=0.0001)
    # elif args.gdc=='none':
    #     args.neighbor_dist_list = get_ins_neighbor_dist(data.y.size(0), data.edge_index, data.train_mask, args.device)
    
    # args.train_idx = data.train_mask.nonzero().squeeze()
    # args.labels_local = data.y.view([-1])[args.train_idx]
    # args.train_idx_list = args.train_idx.cpu().tolist()
    # # 局部图和全局图的id变换
    # args.local2global = {i:args.train_idx_list[i] for i in range(len(args.train_idx_list))}
    # args.global2local = dict([val, key] for key, val in args.local2global.items())
    
    # args.idx_info_list = [item.cpu().tolist() for item in args.idx_info] 
    # args.idx_info_local = [torch.tensor(list(map(args.global2local.get, cls_idx))) for cls_idx in args.idx_info_list] 
    # args.neighbor_dist_list = args.neighbor_dist_list.to(args.device)