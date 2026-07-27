import torch
import os
import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_scatter import scatter_add
from scipy.spatial import distance_matrix
from scipy.sparse import load_npz

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    denom = (max_values - min_values).clamp_min(1e-12)
    return 2*(features - min_values).div(denom) - 1

def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1
    return mask

def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)
    return idx_map


def load_credit(dataset, id, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="../dataset_bak/credit/"):
    path = f"/home/disk2/lhr/EMBER-KDD-2026/dataset/{dataset}"
    idx_features_labels = pd.read_csv(os.path.join(path, f"{dataset}{id}.csv"))
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove('Single')
    header.remove(sens_attr) # sensitive feature removal
    header.remove(predict_attr)
    sens_labels = idx_features_labels[sens_attr].values.astype(int)
    sens_labels = torch.LongTensor(sens_labels)
    labels = idx_features_labels[predict_attr].values
    labels = torch.LongTensor(labels)
    features = idx_features_labels[header]
    features = torch.FloatTensor(np.array(features, dtype=np.float32))

    # build relationship
    if os.path.exists(f'{path}/{dataset}{id}_edges.npz'):
        adj = load_npz(f'{path}/{dataset}{id}_edges.npz')
    else:
        if os.path.exists(f'{path}/{dataset}{id}_edges.txt'):
            edges_unordered = np.genfromtxt(f'{path}/{dataset}{id}_edges.txt').astype('int')
        else:
            edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
            np.savetxt(f'{path}/{dataset}{id}_edges.txt', edges_unordered)

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

    edge_index, _ = from_scipy_sparse_matrix(adj)

    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:int(0.8 * len(label_idx_0))],
                          label_idx_1[:int(0.8 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.8 * len(label_idx_0)):int(0.9 * len(label_idx_0))], 
                        label_idx_1[int(0.8 * len(label_idx_1)):int(0.9 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.9 * len(label_idx_0)):], 
                         label_idx_1[int(0.9 * len(label_idx_1)):])
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))
    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask

def load_bail(dataset, id, sens_attr="WHITE", predict_attr="RECID", path="../dataset/bail/"):
    path = f"/home/disk2/lhr/EMBER-KDD-2026/dataset/{dataset}"
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset + id)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove("user_id")
    header.remove(sens_attr)
    labels = idx_features_labels[predict_attr].values
    labels = torch.LongTensor(labels)
    sens_labels = idx_features_labels[sens_attr].values.astype(int)
    sens_labels = torch.LongTensor(sens_labels)
    features = idx_features_labels[header]
    features = torch.FloatTensor(np.array(features, dtype=np.float32))

    if os.path.exists(f'{path}/{dataset}{id}_edges.npz'):
        adj = load_npz(f'{path}/{dataset}{id}_edges.npz')
    else:
        # build relationship
        if os.path.exists(f'{path}/{dataset}{id}_edges.txt'):
            edges_unordered = np.genfromtxt(f'{path}/{dataset}{id}_edges.txt').astype('int')
        else:
            edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
            np.savetxt(f'{path}/{dataset}{id}_edges.txt', edges_unordered)

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

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

    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask

def load_pokec(dataset, id, sens_attr="region", predict_attr="I_am_working_in_field", path="/home/disk2/lhr/EMBER-KDD-2026/dataset/pokec/"):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset + id)))
    header = list(pd.read_csv(os.path.join(path, "{}.csv".format("pokec_z"))).columns)
    header2 = list(pd.read_csv(os.path.join(path, "{}.csv".format("pokec_n"))).columns)
    header = [i for i in header if i in header2]
    header.remove("user_id")
    header.remove(sens_attr)
    header.remove(predict_attr)
    features = idx_features_labels[header]
    features = torch.FloatTensor(np.array(features, dtype=np.float32))
    labels = idx_features_labels[predict_attr].values
    labels = torch.LongTensor(labels)
    labels[labels > 1] = 1
    # labels[labels < 1] = 0
    sens_labels = idx_features_labels[sens_attr].values.astype(int)
    sens_labels = torch.FloatTensor(sens_labels)

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, f"{dataset}{id}_edges.txt"), dtype=int)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    edge_index, _ = from_scipy_sparse_matrix(adj)

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
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask

def load_german(dataset, id, sens_attr="Gender", predict_attr="GoodCustomer", path="../dataset/german/"):
    path = f"/home/disk2/lhr/EMBER-KDD-2026/dataset/{dataset}"
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset + id)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')
    header.remove(sens_attr)

    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
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

    sens_labels = idx_features_labels[sens_attr].values.astype(int)
    sens_labels = torch.FloatTensor(sens_labels)

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
    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:int(0.8 * len(label_idx_0))],
                          label_idx_1[:int(0.8 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.8 * len(label_idx_0)):int(0.9 * len(label_idx_0))],
                        label_idx_1[int(0.8 * len(label_idx_1)):int(0.9 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.9 * len(label_idx_0)):], label_idx_1[int(0.9 * len(label_idx_1)):])

    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))
   
    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask

def load_syn(dataset, id, path="/home/disk2/lhr/EMBER-KDD-2026/dataset/syn"):
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

    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask


def load_nba(dataset, id, sens_attr="country", predict_attr="SALARY", path="../dataset/nba/"):
    path = f"/home/disk2/lhr/fairDomainAdaption/mine/dataset/{dataset}"
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset + id)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove("user_id")
    header.remove(sens_attr)
    labels = idx_features_labels[predict_attr].values
    labels = torch.LongTensor(labels)
    labels[labels > 1] = 1
    sens_labels = idx_features_labels[sens_attr].values.astype(int)
    sens_labels = torch.LongTensor(sens_labels)
    features = idx_features_labels[header]
    features = torch.FloatTensor(np.array(features, dtype=np.float32))

    if os.path.exists(f'{path}/{dataset}{id}_edges.npz'):
        adj = load_npz(f'{path}/{dataset}{id}_edges.npz')
    else:
        # build relationship
        if os.path.exists(f'{path}/{dataset}{id}_edges.txt'):
            edges_unordered = np.genfromtxt(f'{path}/{dataset}{id}_edges.txt').astype('int')
        else:
            edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
            np.savetxt(f'{path}/{dataset}{id}_edges.txt', edges_unordered)

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

    edge_index, _ = from_scipy_sparse_matrix(adj)

    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:int(0.8 * len(label_idx_0))],
                          label_idx_1[:int(0.8 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.8 * len(label_idx_0)):int(0.9 * len(label_idx_0))],
                        label_idx_1[int(0.8 * len(label_idx_1)):int(0.9 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.9 * len(label_idx_0)):], label_idx_1[int(0.9 * len(label_idx_1)):])
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask


def get_dataset(args, inid):
    if('credit' in args.dataset):
        edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask = load_credit(dataset=args.dataset, id = inid)
    elif('bail' in args.dataset):
        edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask = load_bail(dataset=args.dataset, id = inid)
    elif('pokec'  in args.dataset):
        edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask= load_pokec(dataset=args.dataset, id = inid)
    elif('german'  in args.dataset):
        edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask= load_german(dataset=args.dataset, id = inid)
    elif('syn'  in args.dataset):
        edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask= load_syn(dataset=args.dataset, id = inid)
    elif('nba'  in args.dataset):
        edge_index, features, labels, sens_labels, train_mask, val_mask, test_mask= load_nba(dataset=args.dataset, id = inid)
    else:
        raise NotImplementedError
    features = feature_norm(features)

    data = Data(x = features, edge_index = edge_index, y = labels, sens_labels=sens_labels, 
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

def process_dataset(args, data, is_target=False):
    args.num_node, args.num_features = data.x.shape[0], data.x.shape[1]
    if is_target:
        # Target labels and sensitive attributes are reserved for the final
        # evaluator and must not be inspected during preprocessing/adaptation.
        return data.to(args.device)

    args.num_classes = len(data.y.unique()) - 1
    if args.dataset == "pokec":
        args.num_classes = 1
    label_idx_0 = np.where(data.y == 0)[0]
    label_idx_1 = np.where(data.y == 1)[0]
    print("==============the source label distribution==============")
    print(f"label = 0: {len(label_idx_0)}, ratio = {len(label_idx_0)/ len(data.y)}")
    print(f"label = 1: {len(label_idx_1)}, ratio = {len(label_idx_1)/ len(data.y)}")

    sens_idx_0 = np.where(data.sens_labels == 0)[0]
    sens_idx_1 = np.where(data.sens_labels == 1)[0]
    print("==============the sensitive label distribution==============")
    print(f"sens = 0: {len(sens_idx_0)}, ratio = {len(sens_idx_0)/ len(data.sens_labels)}")
    print(f"sens = 1: {len(sens_idx_1)}, ratio = {len(sens_idx_1)/ len(data.sens_labels)}")

    y0s0_idx = np.where((data.y == 0) & (data.sens_labels == 0))[0]
    y0s1_idx = np.where((data.y == 0) & (data.sens_labels == 1))[0]
    y1s0_idx = np.where((data.y == 1) & (data.sens_labels == 0))[0]
    y1s1_idx = np.where((data.y == 1) & (data.sens_labels == 1))[0]
    print("==============the sensitive label distribution in different source label group==============")
    print(f"sens = 0 in label 0: {len(y0s0_idx)}, ratio = {len(y0s0_idx)/ len(label_idx_0)}")
    print(f"sens = 1 in label 0: {len(y0s1_idx)}, ratio = {len(y0s1_idx)/ len(label_idx_0)}")
    print(f"sens = 0 in label 1: {len(y1s0_idx)}, ratio = {len(y1s0_idx)/ len(label_idx_1)}")
    print(f"sens = 1 in label 1: {len(y1s1_idx)}, ratio = {len(y1s1_idx)/ len(label_idx_1)}")


    print("==============the source label distribution in different sensitive label group==============")
    print(f"label = 0 in sensitive 0: {len(y0s0_idx)}, ratio = {len(y0s0_idx)/ len(sens_idx_0)}")
    print(f"label = 1 in sensitive 0: {len(y1s0_idx)}, ratio = {len(y1s0_idx)/ len(sens_idx_0)}")
    print(f"label = 0 in sensitive 1: {len(y0s1_idx)}, ratio = {len(y0s1_idx)/ len(sens_idx_1)}")
    print(f"label = 1 in sensitive 1: {len(y1s1_idx)}, ratio = {len(y1s1_idx)/ len(sens_idx_1)}")

    return data.to(args.device)
