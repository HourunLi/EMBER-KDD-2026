import os.path as osp
import random
import numpy as np
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F

@torch.no_grad()
def sampling_idx_individual_dst(args, class_num_list, idx_info):
    device = args.device

    # Selecting src & dst nodes
    max_num, n_cls = max(class_num_list), len(class_num_list)
    sampling_list = 1.2 * max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    new_class_num_list = torch.Tensor(class_num_list).to(device)

    # Compute # of source nodes
    
    sampling_src_idx =[cls_idx[torch.randint(len(cls_idx),(int(samp_num.item()),))]
                        for cls_idx, samp_num in zip(idx_info, sampling_list)]
    # 将所有类别的源节点索引合并为一个单一的张量。
    sampling_src_idx = torch.cat(sampling_src_idx)

    # Generate corresponding destination nodes
    # 数量越大，采样概率越低
    prob = torch.log(new_class_num_list.float())/ new_class_num_list.float()
    # prob: 通过重复扩展每个类别的概率，使其长度与类别中节点的总数相匹配。这样，prob 将对应每个节点的选择概率。
    prob = prob.repeat_interleave(new_class_num_list.long())
    # print(idx_info.device)
    temp_idx_info = torch.cat(idx_info).to(args.device)
    dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)
    # print(dst_idx.device)
    # print(temp_idx_info.device)
    sampling_dst_idx = temp_idx_info[dst_idx]

    # Sorting src idx with corresponding dst idx
    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]

    return sampling_src_idx, sampling_dst_idx

# class_num_list: 各个类别的节点数量列表。
# prev_out_local: 先前节点的输出，通常为模型的预测结果。
# idx_info_local: 每个类别的节点索引信息。
# train_idx: 训练节点的全局索引。
# tau: 用于温度缩放的参数，影响 softmax 的平滑程度。
# max_flag: 一个布尔值，用于控制最大值的处理方式。
# no_mask: 一个布尔值，用于决定是否对源节点的置信度进行掩码处理
@torch.no_grad()
def sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, tau=2, max_flag=False, no_mask=False):
    # print(idx_info_local)
    max_num, n_cls = max(class_num_list), len(class_num_list) 
    if not max_flag: # mean
        max_num = sum(class_num_list) / n_cls
    sampling_list = 1.2 * max_num * torch.ones(n_cls) - torch.tensor(class_num_list)

    prev_out_local = F.softmax(prev_out_local/tau, dim=1)
    prev_out_local = prev_out_local.cpu() 

    src_idx_all = []
    dst_idx_all = []
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0: 
            continue

        # first sampling
        # print(prev_out_local[idx_info_local[cls_idx]].size())
        # print(prev_out_local.size())
        prob = 1 - prev_out_local[idx_info_local[cls_idx]][:,cls_idx].squeeze() 
        src_idx_local = torch.multinomial(prob + 1e-12, num, replacement=True) 
        src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]] 

        # second sampling
        conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]] 
        if not no_mask:
            conf_src[:,cls_idx] = 0
        neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze().tolist() 

        # third sampling
        neighbor = [prev_out_local[idx_info_local[cls]][:,cls_idx] for cls in neighbor_cls] 
        dst_idx = []
        for i, item in enumerate(neighbor):
            dst_idx_local = torch.multinomial(item + 1e-12, 1)[0] 
            dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])
        dst_idx = torch.tensor(dst_idx).to(src_idx.device)

        src_idx_all.append(src_idx)
        dst_idx_all.append(dst_idx)
    
    src_idx_all = torch.cat(src_idx_all)
    dst_idx_all = torch.cat(dst_idx_all)
    
    return src_idx_all, dst_idx_all


@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx,neighbor_dist_list, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    
    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

# 根据源节点的采样索引，生成新的边连接以增强图的结构
# total_node: 图中节点的总数量。
# edge_index: 图的边列表，表示节点之间的连接。
# sampling_src_idx: 需要复制边的源节点索引。
@torch.no_grad()
def duplicate_neighbor(total_node, edge_index, sampling_src_idx):
    device = edge_index.device
    
    # Assign node index for augmented nodes
    row, col = edge_index[0], edge_index[1] 
    row, sort_idx = torch.sort(row)
    col = col[sort_idx]
    # 计算节点度数
    degree = scatter_add(torch.ones_like(row), row)
    # 赋予新的节点索引，然后重复度数次数，便于复制边
    new_row =(torch.arange(len(sampling_src_idx)).to(device)+ total_node).repeat_interleave(degree[sampling_src_idx])
    # 根据源节点的度数重复新索引，以匹配每个源节点的连接数量。
    temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)

    # 后面的处理就是复制边
    # Duplicate the edges of source nodes
    node_mask = torch.zeros(total_node, dtype=torch.bool).to(device)
    unique_src = torch.unique(sampling_src_idx)
    node_mask[unique_src] = True
    row_mask = node_mask[row]
    # 获取与源节点相关的目标节点索引。这些目标节点是需要被复制的。
    edge_mask = col[row_mask]
    # 确保每个源节点的索引按其度数重复。
    b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(degree[unique_src])
    #  使用 to_dense_batch 函数将边信息转换为稠密格式，
    # edge_dense 是通过从源节点复制边而生成的新边。表示的是原节点的度数
    edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
    if len(temp[temp!=0]) != edge_dense.shape[0]:
        cut_num =len(temp[temp!=0]) - edge_dense.shape[0]
        cut_temp = temp[temp!=0][:-cut_num]
    else:
        cut_temp = temp[temp!=0]
    edge_dense  = edge_dense.repeat_interleave(cut_temp, dim=0)
    new_col = edge_dense[edge_dense!= -1]
    # : 创建一个新的边索引，包含新生成的目标节点和对应的新源节点。
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    # 将原来的边和新生成的边拼接到一起
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
    new_src = x[sampling_src_idx.to(x.device), :].clone()
    new_dst = x[sampling_dst_idx.to(x.device), :].clone()
    lam = lam.to(x.device)

    mixed_node = lam * new_src + (1-lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim =0)
    return new_x

def mixup(args, epoch, data, prev_out):
    if epoch > args.warmup:
        # identifying source samples
        train_idx = data.train_mask.nonzero().squeeze()
        prev_out_local = prev_out[train_idx]
        sampling_src_idx, sampling_dst_idx = sampling_node_source(args.class_num_list, prev_out_local, args.idx_info_local, train_idx, args.tau, args.max_flag, args.no_mask) 
        # semimxup
        new_edge_index = neighbor_sampling(data.x.size(0), data.edge_index, sampling_src_idx, args.neighbor_dist_list)
        beta = torch.distributions.beta.Beta(1, 100)
        lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
        new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)
    else:
        # print(args.class_num_list, args.idx_info)
        sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(args, args.class_num_list, args.idx_info)
        beta = torch.distributions.beta.Beta(2, 2)
        # beta.sample((len(sampling_src_idx),)): 从 Beta 分布中抽取样本，样本数量与 sampling_src_idx 的长度相同。
        # 这些样本将用于权重（lam），用于在 Mixup 中调节源节点和目标节点特征的混合比例。
        lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
        new_edge_index = duplicate_neighbor(data.x.size(0), data.edge_index, sampling_src_idx)
        new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)
    return new_x, new_edge_index, sampling_src_idx, sampling_dst_idx