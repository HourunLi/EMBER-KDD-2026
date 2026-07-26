import argparse
import random
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import loss
import utils


# 调用指令类似于：sh train_uda_officehome.sh spa 0.2 1.0 DANNE 0.3 laplac1 gauss
def gauss_(v1, v2, sigma):
    norm_ = torch.norm(v1 - v2, p=2, dim=0)
    return torch.exp(-0.5 * norm_ / sigma ** 2)


def euc_(v1, v2):
    return torch.norm(v1 - v2, p=2, dim=0)


def adj_(s, t, ap='cos'):
    # s, t [bsize, dim], [bsize, dim] -> [bsize, bsize]
    if ap == 'cos':
        s_norm = s / torch.norm(s, p=2, dim=1, keepdim=True)
        t_norm = t / torch.norm(t, p=2, dim=1, keepdim=True)
        adj = s_norm @ t_norm.T
    elif ap == 'gauss':
        sigma_ = 1.5
        M, N = s.shape[0], t.shape[0]
        adj = torch.zeros([M, N], dtype=torch.float).cuda()
        for i in range(M):
            for j in range(i):
                adj[i][j] = adj[j][i] = gauss_(s[i], t[j], sigma_)
    elif ap == 'euc':
        M, N = s.shape[0], t.shape[0]
        adj = torch.zeros([M, N], dtype=torch.float).cuda()
        for i in range(M):
            for j in range(i):
                adj[i][j] = adj[j][i] = euc_(s[i], t[j])
    return adj


def laplacian_(A, ltype='laplac1'):
    v = torch.sum(A, dim=1)
    if ltype == 'laplac1':
        v_inv = 1 / v
        D_inv = torch.diag(v_inv).cuda()
        return -D_inv @ A
    elif ltype == 'laplac2':
        D = torch.diag(v).cuda()
        return D - A
    elif ltype == 'laplac3':
        v_sqrt = 1 / torch.sqrt(v)
        D_sqrt = torch.diag(v_sqrt).cuda()
        I = torch.eye(A.shape[0]).cuda()
        return I - D_sqrt @ A @ D_sqrt


def svd_loss_(s, t):
    # s, t [bsize, dim], [bsize, dim]
    s_matrix = adj_(s, s, args.ap)
    t_matrix = adj_(t, t, args.ap)
    s_matrix = laplacian_(s_matrix, args.laplac)
    t_matrix = laplacian_(t_matrix, args.laplac)
    _, s_v, _ = torch.svd(s_matrix)
    _, t_v, _ = torch.svd(t_matrix)
    svd_loss = torch.norm(s_v - t_v, p=2)
    return svd_loss


def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def data_load(args, labels=None):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    source_set = utils.ObjectImage(args.data_root, args.s_dset_path, train_transform, y=labels)
    target_set = utils.ObjectImage(args.data_root, args.t_dset_path, train_transform, ridx=True)
    test_set = utils.ObjectImage(args.data_root, args.test_dset_path, test_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size * 3,
                                                       shuffle=False, num_workers=args.worker, drop_last=False)
    return dset_loaders


def load_baila_mlp(csv_path, feature_columns=None, mean=None, std=None, load_labels=True):
    """Load BailA features and optionally omit labels from an adaptation stage."""
    if feature_columns is None:
        frame = pd.read_csv(csv_path)
    else:
        requested_columns = list(feature_columns)
        if load_labels:
            requested_columns.append('RECID')
        frame = pd.read_csv(csv_path, usecols=requested_columns)
    if load_labels and 'RECID' not in frame:
        raise ValueError(f'{csv_path} must contain the RECID label column.')
    if feature_columns is None:
        # user_id is an identifier, not an explanatory node feature.
        feature_columns = [col for col in frame.columns if col not in {'RECID', 'user_id'}]
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{csv_path} is missing feature columns: {sorted(missing)}')
    features = torch.tensor(frame.loc[:, feature_columns].to_numpy(dtype=np.float32))
    if mean is not None:
        features = (features - mean) / std
    labels = torch.tensor(frame['RECID'].to_numpy(dtype=np.int64)) if load_labels else None
    return features, labels, feature_columns


def _cross_entropy_label_smooth_gpu(logits, targets, num_classes, epsilon, reduction='mean'):
    """Label-smoothed cross entropy without moving labels away from the GPU."""
    if logits.device != targets.device:
        raise ValueError('logits and targets must be on the same device')
    if targets.dtype != torch.long:
        targets = targets.long()

    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    with torch.no_grad():
        smooth_targets = torch.full_like(log_probs, epsilon / num_classes)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - epsilon + epsilon / num_classes)
    per_sample_loss = -(smooth_targets * log_probs).sum(dim=1)

    if reduction == 'none':
        return per_sample_loss
    if reduction == 'sum':
        return per_sample_loss.sum()
    if reduction == 'mean':
        return per_sample_loss.mean()
    raise ValueError(f'Unsupported reduction: {reduction}')


@torch.no_grad()
def _baila_binary_metrics(class_probabilities, labels, sensitive):
    """Return accuracy, ROC-AUC, demographic-parity gap, and equality-of-opportunity gap."""
    if class_probabilities.ndim != 2 or class_probabilities.size(1) != 2:
        raise ValueError('BailA metrics require probabilities with shape [N, 2].')
    sample_count = class_probabilities.size(0)
    if labels.ndim != 1 or sensitive.ndim != 1:
        raise ValueError('labels and sensitive must be one-dimensional tensors.')
    if labels.numel() != sample_count or sensitive.numel() != sample_count:
        raise ValueError('Probabilities, labels, and sensitive attributes must have equal lengths.')
    if sample_count < 1:
        raise ValueError('At least one target sample is required to compute metrics.')
    if not torch.all((labels == 0) | (labels == 1)).item():
        raise ValueError('BailA ROC-AUC requires binary RECID labels in {0, 1}.')
    if not torch.all((sensitive == 0) | (sensitive == 1)).item():
        raise ValueError('BailA fairness metrics require WHITE values in {0, 1}.')

    predictions = class_probabilities.argmax(dim=1)
    positive_scores = class_probabilities[:, 1]
    positive_labels = labels == 1
    negative_labels = labels == 0
    accuracy = (predictions == labels).float().mean()

    # Mann-Whitney ROC-AUC with average ranks, so tied scores receive half credit.
    positive_count = int(positive_labels.sum().item())
    negative_count = int(negative_labels.sum().item())
    if positive_count == 0 or negative_count == 0:
        roc_auc = torch.full((), float('nan'), device=positive_scores.device)
    else:
        sorted_indices = torch.argsort(positive_scores)
        sorted_scores = positive_scores[sorted_indices]
        _, tie_counts = torch.unique_consecutive(sorted_scores, return_counts=True)
        tie_counts_float = tie_counts.to(dtype=positive_scores.dtype)
        rank_ends = tie_counts_float.cumsum(dim=0)
        rank_starts = rank_ends - tie_counts_float + 1.0
        average_ranks = (rank_starts + rank_ends) / 2.0
        sorted_ranks = torch.repeat_interleave(average_ranks, tie_counts)
        ranks = torch.empty_like(positive_scores)
        ranks[sorted_indices] = sorted_ranks
        positive_rank_sum = ranks[positive_labels].sum()
        roc_auc = (
            positive_rank_sum - positive_count * (positive_count + 1) / 2.0
        ) / (positive_count * negative_count)

    group_zero = sensitive == 0
    group_one = sensitive == 1

    def positive_prediction_rate(mask):
        if not mask.any().item():
            return torch.full((), float('nan'), device=class_probabilities.device)
        return (predictions[mask] == 1).float().mean()

    positive_rate_zero = positive_prediction_rate(group_zero)
    positive_rate_one = positive_prediction_rate(group_one)
    parity = torch.abs(positive_rate_zero - positive_rate_one)

    true_positive_group_zero = group_zero & positive_labels
    true_positive_group_one = group_one & positive_labels
    true_positive_rate_zero = positive_prediction_rate(true_positive_group_zero)
    true_positive_rate_one = positive_prediction_rate(true_positive_group_one)
    equality = torch.abs(true_positive_rate_zero - true_positive_rate_one)

    return {
        'accuracy': accuracy.item(),
        'roc_auc': roc_auc.item(),
        'parity': parity.item(),
        'equality': equality.item(),
        'positive_rate_white_0': positive_rate_zero.item(),
        'positive_rate_white_1': positive_rate_one.item(),
        'true_positive_rate_white_0': true_positive_rate_zero.item(),
        'true_positive_rate_white_1': true_positive_rate_one.item(),
        'predictions': predictions,
    }


def train_baila_mlp(args):
    """Two-stage source-free adaptation: source supervision, then target pseudo-labeling."""
    if not torch.cuda.is_available():
        raise RuntimeError('BailA MLP training requires an available CUDA device.')
    if args.max_epoch < 1:
        raise ValueError('--max_epoch must be at least 1.')
    if args.batch_size < 1:
        raise ValueError('--batch_size must be at least 1.')
    if args.pl != 'spa':
        raise ValueError('BailA SFDA requires --pl spa for target pseudo-label adaptation.')
    if args.K < 1:
        raise ValueError('--K must be at least 1 for target pseudo-label adaptation.')
    if not 0.0 <= args.momentum <= 1.0:
        raise ValueError('--momentum must be in [0, 1].')
    if args.tar_par <= 0.0:
        raise ValueError('--tar_par must be positive for target pseudo-label adaptation.')
    if args.log_interval < 1:
        raise ValueError('--log_interval must be at least 1.')
    if not 0.0 <= args.smooth < 1.0:
        raise ValueError('--smooth must be in [0, 1).')
    device = torch.device('cuda:0')
    class_num = args.class_num

    def write_log(message):
        print(message)
        if getattr(args, 'out_file', None) is not None:
            args.out_file.write(message + '\n')
            args.out_file.flush()

    def infer_in_batches(inputs, batch_size, return_features=False):
        base_network.eval()
        feature_chunks = []
        score_chunks = []
        with torch.no_grad():
            for start in range(0, inputs.size(0), batch_size):
                features, logits = base_network(inputs[start:start + batch_size])
                if return_features:
                    feature_chunks.append(features)
                score_chunks.append(torch.softmax(logits, dim=1))
        all_features = torch.cat(feature_chunks, dim=0) if return_features else None
        return all_features, torch.cat(score_chunks, dim=0)

    def state_dict_on_cpu(module):
        return {name: value.detach().cpu() for name, value in module.state_dict().items()}

    if args.method != 'srconly' or args.ifsvd:
        write_log(
            '[SFDA] --method and --ifsvd are ignored in the BailA target adaptation stage; '
            'DANN/CDAN and source-target SVD require source samples.'
        )

    # Stage 1: source-only supervised pretraining. The target CSV is not read here.
    source_x, source_y, columns = load_baila_mlp(args.source_csv)
    if source_y.numel() < 1:
        raise ValueError('The BailA source domain must contain at least one sample.')
    if source_y.min().item() < 0 or source_y.max().item() >= class_num:
        raise ValueError(f'Source labels must be in [0, {class_num - 1}].')
    source_mean = source_x.mean(dim=0)
    source_std = source_x.std(dim=0).clamp_min(1e-6)
    source_x = ((source_x - source_mean) / source_std).to(device)
    source_y = source_y.to(device)

    netG = utils.MLPEncoder(source_x.size(1), args.hidden, args.dropout).to(device)
    netF = utils.ResClassifier(class_num, netG.in_features, args.bottleneck_dim).to(device)
    base_network = nn.Sequential(netG, netF)

    source_batch_size = min(args.batch_size, source_y.numel())
    source_steps_per_epoch = max(1, int(np.ceil(source_y.numel() / source_batch_size)))
    source_max_iter = args.max_epoch * source_steps_per_epoch
    source_optimizer_g = optim.SGD(netG.parameters(), lr=args.lr * 0.1)
    source_optimizer_f = optim.SGD(netF.parameters(), lr=args.lr)
    source_iter = 0

    write_log(
        f'[Source] samples={source_y.numel()}, epochs={args.max_epoch}, '
        f'steps/epoch={source_steps_per_epoch}, total_iters={source_max_iter}'
    )
    for epoch in range(1, args.max_epoch + 1):
        source_order = torch.randperm(source_y.numel(), device=device)
        epoch_loss_sum = torch.zeros((), dtype=source_x.dtype, device=device)
        epoch_sample_count = 0

        for start in range(0, source_y.numel(), source_batch_size):
            source_iter += 1
            source_indices = source_order[start:start + source_batch_size]
            inputs_source = source_x[source_indices]
            labels_source = source_y[source_indices]

            base_network.train()
            lr_scheduler(source_optimizer_g, args.lr * 0.1, source_iter, source_max_iter)
            lr_scheduler(source_optimizer_f, args.lr, source_iter, source_max_iter)
            _, outputs_source = base_network(inputs_source)
            source_loss = _cross_entropy_label_smooth_gpu(
                outputs_source, labels_source, class_num, args.smooth, reduction='mean'
            )

            source_optimizer_g.zero_grad(set_to_none=True)
            source_optimizer_f.zero_grad(set_to_none=True)
            source_loss.backward()
            source_optimizer_g.step()
            source_optimizer_f.step()

            current_batch_size = labels_source.numel()
            epoch_loss_sum += source_loss.detach() * current_batch_size
            epoch_sample_count += current_batch_size
            if source_iter % args.log_interval == 0 or source_iter == source_max_iter:
                write_log(
                    f'[Source] Iter {source_iter}/{source_max_iter}: '
                    f'cls={source_loss.item():.5f}'
                )

        _, source_score = infer_in_batches(
            source_x, max(source_batch_size * 3, 1024), return_features=False
        )
        source_accuracy = (source_score.argmax(dim=1) == source_y).float().mean()
        source_entropy = loss.Entropy(source_score).mean()
        write_log(
            f'[Source] Epoch {epoch}/{args.max_epoch}: '
            f'loss={(epoch_loss_sum / epoch_sample_count).item():.5f}, '
            f'acc={source_accuracy.item() * 100:.2f}%, '
            f'entropy={source_entropy.item():.4f}'
        )

    source_checkpoint_path = osp.join(args.output_dir, 'source_model.pt')
    torch.save(
        {
            'netG': state_dict_on_cpu(netG),
            'netF': state_dict_on_cpu(netF),
            'feature_columns': list(columns),
            'source_mean': source_mean,
            'source_std': source_std,
            'class_num': class_num,
            'hidden': args.hidden,
            'bottleneck_dim': args.bottleneck_dim,
            'dropout': args.dropout,
            'source_epochs': args.max_epoch,
        },
        source_checkpoint_path,
    )
    write_log(f'[Source] checkpoint saved to: {source_checkpoint_path}')

    netG.zero_grad(set_to_none=True)
    netF.zero_grad(set_to_none=True)
    del source_x, source_y, source_score, source_order, source_indices
    del inputs_source, labels_source, outputs_source, source_loss
    del source_accuracy, source_entropy, epoch_loss_sum
    del source_optimizer_g, source_optimizer_f
    torch.cuda.empty_cache()

    # Stage 2: the source tensors no longer exist; load target features without RECID.
    target_x, _, _ = load_baila_mlp(
        args.target_csv, columns, source_mean, source_std, load_labels=False
    )
    if target_x.size(0) < 2:
        raise ValueError('Target pseudo-label adaptation requires at least two samples.')
    target_x = target_x.to(device)
    target_sample_count = target_x.size(0)
    target_batch_size = min(args.batch_size, target_sample_count)
    target_steps_per_epoch = max(1, int(np.ceil(target_sample_count / target_batch_size)))
    target_max_iter = args.max_epoch * target_steps_per_epoch
    args.max_iter = source_max_iter + target_max_iter

    # Initialize the memory bank with the source model instead of random features/classes.
    initial_features, initial_score = infer_in_batches(
        target_x, max(target_batch_size * 3, 1024), return_features=True
    )
    mem_fea = torch.nn.functional.normalize(initial_features, p=2, dim=1)
    mem_cls = initial_score.clone()
    del initial_features, initial_score

    # Both the encoder and classifier remain trainable, as requested.
    target_optimizer_g = optim.SGD(netG.parameters(), lr=args.lr * 0.1)
    target_optimizer_f = optim.SGD(netF.parameters(), lr=args.lr)
    target_iter = 0
    neighbour_count = min(args.K, target_sample_count - 1)

    write_log(
        f'[Target-SFDA] samples={target_sample_count}, epochs={args.max_epoch}, '
        f'steps/epoch={target_steps_per_epoch}, total_iters={target_max_iter}, '
        f'K={neighbour_count}'
    )
    for epoch in range(1, args.max_epoch + 1):
        target_order = torch.randperm(target_sample_count, device=device)
        epoch_loss_sum = torch.zeros((), dtype=target_x.dtype, device=device)
        epoch_confidence_sum = torch.zeros((), dtype=target_x.dtype, device=device)
        epoch_sample_count = 0

        for start in range(0, target_sample_count, target_batch_size):
            target_iter += 1
            target_indices = target_order[start:start + target_batch_size]
            inputs_target = target_x[target_indices]
            current_batch_size = target_indices.numel()

            base_network.train()
            lr_scheduler(target_optimizer_g, args.lr * 0.1, target_iter, target_max_iter)
            lr_scheduler(target_optimizer_f, args.lr, target_iter, target_max_iter)
            features_target, outputs_target = base_network(inputs_target)

            with torch.no_grad():
                normalized_target = torch.nn.functional.normalize(
                    features_target.detach(), p=2, dim=1
                )
                similarity = normalized_target @ mem_fea.T
                similarity[
                    torch.arange(current_batch_size, device=device), target_indices
                ] = -torch.inf
                neighbours = similarity.topk(k=neighbour_count, dim=1).indices
                neighbour_probability = mem_cls[neighbours].mean(dim=1)
                confidence, pseudo_labels = neighbour_probability.max(dim=1)

            per_sample_loss = nn.functional.cross_entropy(
                outputs_target, pseudo_labels, reduction='none'
            )
            pseudo_loss = (
                confidence * per_sample_loss
            ).sum() / confidence.sum().clamp_min(1e-8)
            adaptation_loss = args.tar_par * pseudo_loss

            target_optimizer_g.zero_grad(set_to_none=True)
            target_optimizer_f.zero_grad(set_to_none=True)
            adaptation_loss.backward()
            target_optimizer_g.step()
            target_optimizer_f.step()

            base_network.eval()
            with torch.no_grad():
                memory_features, memory_outputs = base_network(inputs_target)
                memory_features = torch.nn.functional.normalize(memory_features, p=2, dim=1)
                memory_probability = torch.softmax(memory_outputs, dim=1).square()
                memory_probability = memory_probability / memory_probability.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)

                mixed_features = (
                    (1.0 - args.momentum) * mem_fea[target_indices]
                    + args.momentum * memory_features
                )
                mem_fea[target_indices] = torch.nn.functional.normalize(
                    mixed_features, p=2, dim=1
                )
                mixed_probability = (
                    (1.0 - args.momentum) * mem_cls[target_indices]
                    + args.momentum * memory_probability
                )
                mem_cls[target_indices] = mixed_probability / mixed_probability.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)

            epoch_loss_sum += pseudo_loss.detach() * current_batch_size
            epoch_confidence_sum += confidence.mean() * current_batch_size
            epoch_sample_count += current_batch_size
            if target_iter % args.log_interval == 0 or target_iter == target_max_iter:
                write_log(
                    f'[Target-SFDA] Iter {target_iter}/{target_max_iter}: '
                    f'pseudo={pseudo_loss.item():.5f}, '
                    f'weighted={adaptation_loss.item():.5f}, '
                    f'confidence={confidence.mean().item():.4f}'
                )

        _, target_score = infer_in_batches(
            target_x, max(target_batch_size * 3, 1024), return_features=False
        )
        target_entropy = loss.Entropy(target_score).mean()
        predicted_count = torch.bincount(target_score.argmax(dim=1), minlength=class_num)
        write_log(
            f'[Target-SFDA] Epoch {epoch}/{args.max_epoch}: '
            f'pseudo={(epoch_loss_sum / epoch_sample_count).item():.5f}, '
            f'confidence={(epoch_confidence_sum / epoch_sample_count).item():.4f}, '
            f'entropy={target_entropy.item():.4f}, '
            f'predicted_count={predicted_count.tolist()}'
        )

    adapted_checkpoint_path = osp.join(args.output_dir, 'sfda_adapted_model.pt')
    torch.save(
        {
            'netG': state_dict_on_cpu(netG),
            'netF': state_dict_on_cpu(netF),
            'feature_columns': list(columns),
            'source_mean': source_mean,
            'source_std': source_std,
            'class_num': class_num,
            'hidden': args.hidden,
            'bottleneck_dim': args.bottleneck_dim,
            'dropout': args.dropout,
            'source_epochs': args.max_epoch,
            'target_epochs': args.max_epoch,
        },
        adapted_checkpoint_path,
    )
    write_log(f'[Target-SFDA] checkpoint saved to: {adapted_checkpoint_path}')

    # RECID and WHITE are loaded only after adaptation, exclusively for final evaluation.
    target_evaluation = pd.read_csv(args.target_csv, usecols=['RECID', 'WHITE'])
    target_y = torch.tensor(
        target_evaluation['RECID'].to_numpy(dtype=np.int64), device=device
    )
    target_sensitive = torch.tensor(
        target_evaluation['WHITE'].to_numpy(dtype=np.int64), device=device
    )
    _, final_score = infer_in_batches(
        target_x, max(target_batch_size * 3, 1024), return_features=False
    )
    final_metrics = _baila_binary_metrics(final_score, target_y, target_sensitive)
    final_prediction = final_metrics.pop('predictions')
    final_entropy = loss.Entropy(final_score).mean()
    write_log(
        f'[Target-Test] Accuracy={final_metrics["accuracy"] * 100:.2f}%, '
        f'ROC_AUC={final_metrics["roc_auc"]:.4f}, '
        f'Parity={final_metrics["parity"]:.4f}, '
        f'Equality={final_metrics["equality"]:.4f}, '
        f'Entropy={final_entropy.item():.4f}'
    )
    write_log(
        f'[Target-Groups] PositiveRate(WHITE=0)='
        f'{final_metrics["positive_rate_white_0"]:.4f}, '
        f'PositiveRate(WHITE=1)={final_metrics["positive_rate_white_1"]:.4f}, '
        f'TPR(WHITE=0)={final_metrics["true_positive_rate_white_0"]:.4f}, '
        f'TPR(WHITE=1)={final_metrics["true_positive_rate_white_1"]:.4f}'
    )
    return final_prediction.cpu().numpy().astype(np.int64)


# 自己改的
def get_dataset(args):
    if (args.dset == 'bailA'):
        adj_s, features_s, labels_s, _, _, _, sens_s = utils.load_bailA(
            'bailA_2', path=args.s_dset_path, sens_attr="WHITE", predict_attr="RECID")
        adj_t, features_t, labels_t, _, _, _, sens_t = utils.load_bailA(
            'bailA_1', path=args.t_dset_path, sens_attr="WHITE", predict_attr="RECID")

    return {
        "source": {
            "adj": adj_s,
            "features": features_s,
            "labels": labels_s
            # "idx_train": idx_train_s,
            # "idx_val": idx_val_s,
            # "idx_test": idx_test_s
        },

        "target": {
            "adj": adj_t,
            "features": features_t,
            "labels": labels_t
            # "idx_train": idx_train_t,
            # "idx_val": idx_val_t,
            # "idx_test": idx_test_t
        }
    }


def train(args, validate=False, label=None):
    ## set pre-process
    # dset_loaders = data_load(args, label)  # 要改
    dset_loaders = get_dataset(args)

    class_num = args.class_num  # 二分类写2就行，主函数里改了
    class_weight_src = torch.ones(class_num, ).cuda()
    ##################################################################################################
    ## set base network
    if args.net == 'resnet101':
        netG = utils.ResBase101().cuda()  # 特征提取器（要改GCN
    elif args.net == 'resnet50':
        netG = utils.ResBase50().cuda()

    netF = utils.ResClassifier(class_num=class_num, feature_dim=netG.in_features,
                               bottleneck_dim=args.bottleneck_dim).cuda()  # 分类器

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch * max_len

    ad_flag = False
    if args.method in {'DANN', 'DANNE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim, 1024, max_iter=args.max_iter).cuda()
        ad_flag = True
    if args.method in {'CDAN', 'CDANE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim * class_num, 1024, max_iter=args.max_iter).cuda()
        random_layer = None
        ad_flag = True

    optimizer_g = optim.SGD(netG.parameters(), lr=args.lr * 0.1)  # 更新ResNet（更新GCN）
    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr)  # 更新classifier
    if ad_flag:
        optimizer_d = optim.SGD(ad_net.parameters(), lr=args.lr)  # 更新Domain Discriminator

    base_network = nn.Sequential(netG, netF)  # 组合ResNet -> classifier（GCN -> classifier）

    mem_fea = torch.rand(len(dset_loaders["target"].dataset), args.bottleneck_dim).cuda()
    mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)  # Target Feature Memory
    mem_cls = torch.ones(len(dset_loaders["target"].dataset), class_num).cuda() / class_num  # Target Probability

    source_loader_iter = iter(dset_loaders["source"])
    target_loader_iter = iter(dset_loaders["target"])
    ####
    list_acc = []
    best_ent = 100
    for iter_num in range(1, args.max_iter + 1):  # epoch * batch
        base_network.train()
        lr_scheduler(optimizer_g, init_lr=args.lr * 0.1, iter_num=iter_num,
                     max_iter=args.max_iter)  # 更新Feature Extractor
        lr_scheduler(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)  # 更新Classifier
        if ad_flag:
            lr_scheduler(optimizer_d, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)  # 更新Discriminator

        try:
            inputs_source, labels_source = source_loader_iter.next()
        except:  # epoch结束，重新开始
            source_loader_iter = iter(dset_loaders["source"])
            inputs_source, labels_source = source_loader_iter.next()
        try:
            inputs_target, _, idx = target_loader_iter.next()
        except:
            target_loader_iter = iter(dset_loaders["target"])
            inputs_target, _, idx = target_loader_iter.next()

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        if args.method == 'srconly' and args.pl == 'none':
            features_source, outputs_source = base_network(inputs_source)
        else:
            features_source, outputs_source = base_network(inputs_source)
            features_target, outputs_target = base_network(inputs_target)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

        eff = utils.calc_coeff(iter_num, max_iter=args.max_iter)
        if args.method[-1] == 'E':
            entropy = loss.Entropy(softmax_out)
        else:
            entropy = None

        if args.method in {'CDAN', 'CDANE'}:
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, eff, random_layer)
        elif args.method in {'DANN', 'DANNE'}:
            transfer_loss = loss.DANN(features, ad_net, entropy, eff)
        elif args.method == 'srconly':
            transfer_loss = torch.tensor(0.0).cuda()
        else:
            raise ValueError('Method cannot be recognized.')

        src_ = loss.CrossEntropyLabelSmooth(reduction='none', num_classes=class_num, epsilon=args.smooth)(
            outputs_source, labels_source)
        weight_src = class_weight_src[labels_source].unsqueeze(0)
        classifier_loss = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())
        total_loss = transfer_loss + classifier_loss  # 目前只有两个

        eff = iter_num / args.max_iter

        if args.ifcorrect:
            features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
        dis = -torch.mm(features_target.detach(), mem_fea.t())  # Cosine Distance
        for di in range(dis.size(0)):
            dis[di, idx[di]] = torch.max(dis)  # 防止自己找自己
        _, p1 = torch.sort(dis, dim=1)

        w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
        for wi in range(w.size(0)):
            for wj in range(args.K):
                w[wi][p1[wi, wj]] = 1 / args.K
        weight_, pred = torch.max(w.mm(mem_cls), 1)  # pred就是Pseudo Label

        loss_ = nn.CrossEntropyLoss(reduction='none')(outputs_target, pred)
        classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item())
        pl_loss = args.tar_par * eff * classifier_loss  # Pseudo Label Loss
        if args.pl != 'none':
            total_loss += pl_loss  # 加上Pseudo Label Loss

        if args.ifsvd:
            # svd loss
            f_s = features_source
            f_t = features_target
            # svd_loss = args.svd_par * eff * svd_loss_(f_s, f_t)
            svd_loss = args.svd_par * svd_loss_(f_s, f_t)
            total_loss += svd_loss  # 加上Spectral Alignment loss

        # 反向传播
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        if ad_flag:
            optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        if ad_flag:
            optimizer_d.step()

        # 更新 Memory Bank
        base_network.eval()
        with torch.no_grad():
            features_target, outputs_target = base_network(inputs_target)
            features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            outputs_target = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))

            mem_fea[idx] = (1.0 - args.momentum) * mem_fea[idx] + args.momentum * features_target.clone()
            mem_cls[idx] = (1.0 - args.momentum) * mem_cls[idx] + args.momentum * outputs_target.clone()

        if iter_num % 10 == 0:
            iter_str = 'total:{:.5f}, trans: {:.5f}, cls: {:.5f}'.format(total_loss.item(), transfer_loss.item(),
                                                                         classifier_loss.item())
            if args.pl != 'none':
                iter_str += ', pl:{:.5f}'.format(pl_loss.item())
            if args.ifsvd:
                iter_str += ', svd:{:.5f}'.format(svd_loss.item())
            print(iter_str)

        if iter_num % int(max_len) == 0:
            base_network.eval()
            if args.dset == 'visda2017':
                acc, py, score, y, tacc = utils.cal_acc_visda(dset_loaders["test"], base_network)
                args.out_file.write(tacc + '\n')
                args.out_file.flush()
                print(tacc)

                _ent = loss.Entropy(score)
                mean_ent = 0
                for ci in range(args.class_num):
                    mean_ent += _ent[py == ci].mean()
                mean_ent /= args.class_num
            else:
                acc, py, score, y = utils.cal_acc(dset_loaders["test"], base_network)
                mean_ent = torch.mean(loss.Entropy(score))

            list_acc.append(acc * 100)
            if best_ent > mean_ent:
                best_ent = mean_ent
                val_acc = acc * 100
                best_y = y
                best_py = py
                best_score = score

            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.name, iter_num,
                                                                                           args.max_iter, acc * 100,
                                                                                           mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

    idx = np.argmax(np.array(list_acc))
    max_acc = list_acc[idx]
    final_acc = list_acc[-1]

    log_str = '\n==========================================\n'
    log_str += '\nVal Acc = {:.2f}\nMax Acc = {:.2f}\nFin Acc = {:.2f}\n'.format(val_acc, max_acc, final_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return best_y.cpu().numpy().astype(np.int64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation Methods')
    parser.add_argument('--method', type=str, default='srconly', choices=['srconly', 'CDAN', 'CDANE', 'DANN', 'DANNE'])
    parser.add_argument('--pl', type=str, default='none', choices=['none', 'spa', 'npl', 'bsp'])

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=256)

    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--tar_par', type=float, default=1.0)
    parser.add_argument('--validate', action='store_true')

    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet50", "resnet101", "mlp"])
    parser.add_argument('--dset', type=str, default='office_home',
                        choices=['domain_net', 'multi', 'visda2017', 'office31', 'office_home', 'bailA'], help="dataset used")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    parser.add_argument('--ifcorrect', action='store_true')
    parser.add_argument('--ifsvd', action='store_true')
    parser.add_argument('--svd_par', type=float, default=1.0)
    parser.add_argument('--laplac', type=str, default='laplac1', choices=["laplac1", "laplac2", "laplac3"])
    parser.add_argument('--ap', type=str, default='euc', choices=['cos', 'gauss', 'euc'])
    # 自己加的
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')  # GCN隐藏层
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--source_csv', type=str, default='./dataset/bailA/bailA_2.csv')
    parser.add_argument('--target_csv', type=str, default='./dataset/bailA/bailA_1.csv')
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()
    args.output = args.output.strip()

    args.eval_epoch = args.max_epoch / 10

    # if args.dset == 'office_home':
    #     names = ['Art', 'Clipart', 'Product', 'Real']
    #     args.class_num = 65
    #     args.data_root = ''
    # if args.dset == 'office31':
    #     names = ['amazon', 'dslr', 'webcam']
    #     args.class_num = 31
    #     args.data_root = ''
    # if args.dset == 'visda2017':
    #     names = ['train', 'validation']
    #     args.class_num = 12
    # if args.dset == 'multi': # DomainNet-126
    #     names = ['clipart', 'painting', 'real', 'sketch']
    #     args.class_num = 126
    #     args.data_root = '/data/domain_net/'
    # if args.dset == 'domain_net':
    #     names = ['clipart_train', 'painting_train', 'real_train', 'sketch_train']
    #     tests = ['clipart_test', 'painting_test', 'real_test', 'sketch_test']
    #     args.class_num = 345
    #     args.data_root = ''

    # 自己加的
    if args.dset == 'bailA':
        if args.net != 'mlp':
            parser.error('--dset bailA requires --net mlp')
        names = ['bailA_2', 'bailA_1']
        args.class_num = 2
        args.data_root = ''
        args.s_dset_path = './dataset/bailA'
        args.t_dset_path = './dataset/bailA'  # 记得回来改
        args.test_dset_path = args.t_dset_path

    # 以下两部分合并到上面
    # args.s_dset_path = './data/uda/' + args.dset + '/' + names[args.s] + '.txt'
    # args.t_dset_path = './data/uda/' + args.dset + '/' + names[args.t] + '.txt'

    # if args.dset == 'domain_net':
    #     args.test_dset_path = './data/' + args.dset + '/' + tests[args.t] + '.txt'
    # else:
    #     args.test_dset_path = args.t_dset_path

    if args.pl == 'none':
        args.output_dir = osp.join(args.output, args.pl, args.dset,
                                   names[args.s][0].upper() + names[args.t][0].upper())
    else:
        args.output_dir = osp.join(args.output, args.pl + '_' + str(args.tar_par), args.dset,
                                   names[args.s][0].upper() + names[args.t][0].upper())

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    # 创建结果文件夹（同时兼容 Windows 和 Linux）。
    os.makedirs(args.output_dir, exist_ok=True)

    args.log = args.method
    args.out_file = open(osp.join(args.output_dir, "{:}.txt".format(args.log)), "w")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    utils.print_args(args)
    if args.dset == 'bailA':
        train_baila_mlp(args)
    else:
        label = train(args)
        if args.validate:
            train(args, validate=True, label=label)
