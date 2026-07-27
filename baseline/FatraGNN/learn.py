import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils import fair_metric, InfoNCE, random_aug, consis_loss
import numpy as np


def _ember_fair_metric(pred, labels, sens):
    """Compute DP and EO with the same definition used by EMBER."""
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y0 = np.bitwise_and(idx_s0, labels == 0)
    idx_s1_y0 = np.bitwise_and(idx_s1, labels == 0)
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)

    def conditional_gap(left, right):
        if left.size == 0 or right.size == 0:
            return float('nan')
        value = abs(float(left.mean()) - float(right.mean()))
        return value if np.isfinite(value) else float('nan')

    dp = conditional_gap(pred[idx_s0], pred[idx_s1])
    eo_y0 = conditional_gap(pred[idx_s0_y0] == 0, pred[idx_s1_y0] == 0)
    eo_y1 = conditional_gap(pred[idx_s0_y1] == 1, pred[idx_s1_y1] == 1)
    return float(dp), float(0.5 * (eo_y0 + eo_y1))


def _ember_metrics(logits, labels, sens):
    """Return Acc, AUC, DP, and EO using EMBER/learn.py's convention."""
    probs = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    sens = sens.detach().cpu().numpy()
    pred = (probs > 0.5).astype(int)

    target_metrics = []
    has_both_classes = len(set(y_true)) == 2
    for yval in np.unique(y_true):
        idx = y_true == yval
        target_metrics.append({
            'acc': accuracy_score(y_true[idx], pred[idx]) * 100,
            'auc': roc_auc_score(
                (y_true == yval).astype(int),
                probs if yval == 1 else 1 - probs,
            ) * 100 if has_both_classes else float('nan'),
        })

    dp, eo = _ember_fair_metric(pred, y_true, sens)
    return (
        np.nanmean([metrics['acc'] for metrics in target_metrics]),
        np.nanmean([metrics['auc'] for metrics in target_metrics]),
        dp * 100,
        eo * 100,
    )

def train(model, data, optimizer, args):
    model.train()
    optimizer.zero_grad()

    output, h = model(data.x, data.edge_index)
    preds = (output.squeeze() > 0).type_as(data.y)

    loss = {}
    loss['train'] = F.binary_cross_entropy_with_logits(
        output[data.train_mask], data.y[data.train_mask].unsqueeze(1).float().to(args.device))
    loss['val'] = F.binary_cross_entropy_with_logits(
        output[data.val_mask], data.y[data.val_mask].unsqueeze(1).float().to(args.device), weight=args.val_ratio)

    loss['train'].backward()
    optimizer.step()

    return loss


def evaluate(model, data, args):
    model.eval()

    with torch.no_grad():
        x_flip, edge_index1, edge_index2, mask1, mask2 = random_aug(
            data.x, data.edge_index, args)
        output, h = model(x_flip, data.edge_index, mask=torch.ones_like(
            data.edge_index[0, :], dtype=torch.bool))

        loss_ce = F.binary_cross_entropy_with_logits(
            output[data.val_mask], data.y[data.val_mask].unsqueeze(1), weight=args.val_ratio)

        # loss_cl = InfoNCE(h[data.train_mask], h[data.train_mask],
        #                   args.label_mask_pos, args.label_mask_neg, tau=0.5)

        loss_val = loss_ce

    accs, auc_rocs, F1s = {}, {}, {}
    all_mask = data.train_mask | data.val_mask | data.test_mask
    pred_all= (output[all_mask].squeeze() > 0).type_as(data.y)
    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['all'] = pred_val.eq(
        data.y[all_mask]).sum().item() / all_mask.sum()
    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum()

    F1s['all'] = f1_score(data.y[all_mask].cpu().numpy(), pred_all.cpu().numpy())

    F1s['val'] = f1_score(data.y[data.val_mask].cpu().numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu().numpy(), pred_test.cpu().numpy())

    auc_rocs['all'] = roc_auc_score(
        data.y[all_mask].cpu().numpy(), output[all_mask].detach().cpu().numpy())
    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    parity, equality = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, parity, equality, loss_val


def evaluate_finetune(encoder, classifier, data, args):
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        h = encoder(data.x, data.edge_index)
        output = classifier(h)

    accs, auc_rocs, F1s = {}, {}, {}

    loss_val = F.binary_cross_entropy_with_logits(
        output[data.val_mask], data.y[data.val_mask].unsqueeze(1).float().to(args.device))

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    parity, equality = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, parity, equality, loss_val


def evaluate_exploration(x, model, data, args):
    model.eval()

    with torch.no_grad():
        outputs, loss_ce = [], 0
        for k in range(args.K):
            x = data.x.clone()
            # print(data.x.unique())
            x[:, args.corr_idx] = (torch.rand(
                len(args.corr_idx)) * (args.x_max[args.corr_idx] - args.x_min[args.corr_idx]) + args.x_min[args.corr_idx]).to(args.device)

            output, h2 = model(x, data.edge_index)
            outputs.append(output)

            loss_ce += F.binary_cross_entropy_with_logits(
                output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

        loss_val = loss_ce / args.K

        # output1, h1 = model(data.x, data.edge_index)
        # output2, h2 = model(x, data.edge_index)

        # loss_ce = F.binary_cross_entropy_with_logits(
        #     output2[data.val_mask], data.y[data.val_mask].unsqueeze(1))

        # loss_val = loss_ce

    output = torch.stack(outputs).mean(dim=0)

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys, loss_val


def evaluate_ged(x, classifier, discriminator, generator, encoder, data, args):
    classifier.eval()
    generator.eval()
    discriminator.eval()
    encoder.eval()

    with torch.no_grad():
        if(args.f_mask == 'yes'):
            outputs, loss_e = [], 0
            feature_weights = generator()
            for k in range(args.K):
                x = data.x * F.gumbel_softmax(
                    feature_weights, tau=1, hard=False)[:, 0]

                h = encoder(x, data.edge_index)
                output = classifier(h)
                output2 = discriminator(h)

                if(args.adv == 'yes'):
                    loss_e += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + args.sup_alpha * F.binary_cross_entropy_with_logits(
                        output[data.val_mask], data.y[data.val_mask].unsqueeze(1))
                else:
                    loss_e += F.binary_cross_entropy_with_logits(
                        output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

                outputs.append(output)

            loss_val = loss_e / args.K

            output = torch.stack(outputs).mean(dim=0)
        else:
            h = encoder(data.x, data.edge_index)
            output, h = classifier(h)

            if(args.adv == 'yes'):
                loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + args.sup_alpha * F.binary_cross_entropy_with_logits(
                    output[data.val_mask], data.y[data.val_mask].unsqueeze(1))
            else:
                loss_val = F.binary_cross_entropy_with_logits(
                    output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys, loss_val


def evaluate_ged2(x, classifier, discriminator, generator, encoder, data, args):
    classifier.eval()
    generator.eval()
    encoder.eval()

    with torch.no_grad():
        if(args.f_mask == 'yes'):
            outputs, loss = [], 0
            feature_weights = generator()
            for k in range(args.K):
                x = data.x * F.gumbel_softmax(
                    feature_weights, tau=1, hard=False)[:, 0]

                h = encoder(x, data.edge_index)
                output = classifier(h)
                output2 = discriminator(h)

                # loss += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
                #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

                outputs.append(output)

            loss_val = loss / args.K

            output = torch.stack(outputs).mean(dim=0)
        else:
            h = encoder(data.x, data.edge_index)
            output = classifier(h)
            output2 = discriminator(h)

            # loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
            #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys


def evaluate_ged3(x, classifier, discriminator, generator, encoder, data, args):
    sens_labels = data.sens
    test_labels = data.y[data.test_mask]

    t_idx_s0 = sens_labels[data.test_mask] == 0
    t_idx_s1 = sens_labels[data.test_mask] == 1
    t_idx_s0_y1 = torch.logical_and(t_idx_s0, test_labels == 1)
    t_idx_s1_y1 = torch.logical_and(t_idx_s1, test_labels == 1)
    t_idx_s0_y0 = torch.logical_and(t_idx_s0, test_labels == 0)
    t_idx_s1_y0 = torch.logical_and(t_idx_s1, test_labels == 0)

    classifier.eval()
    generator.eval()
    encoder.eval()
    with torch.no_grad():
        if(args.f_mask == 'yes'):
            outputs, loss = [], 0
            feature_weights = generator()
            for k in range(args.K):
                x = data.x * F.gumbel_softmax(
                    feature_weights, tau=1, hard=True)[:, 0]

                h = encoder(x, data.edge_index, data.adj_norm_sp)
                feat = h
                output = classifier(h)
                # output2 = discriminator(h)

                # loss += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
                #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

                outputs.append(output)

            # loss_val = loss / args.K

            output = torch.stack(outputs).mean(dim=0)
        else:
            h = encoder(data.x, data.edge_index, data.adj_norm_sp)
            feat = h
            output = classifier(h)
            # output2 = discriminator(h)

            # loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
            #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    representation_np = feat[data.test_mask].detach().cpu().numpy()
    labels = torch.full((test_labels.shape[0],), -1, dtype=torch.int64)

    # 赋值类别标签
    labels[t_idx_s0_y1] = 0
    labels[t_idx_s1_y1] = 1
    labels[t_idx_s0_y0] = 2
    labels[t_idx_s1_y0] = 3
    labels_np = labels.cpu().numpy()

    np.savez(f"pokec_feat.npz", representations=representation_np)
    np.savez(f"pokec_labels.npz", labels=labels_np)

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}
    all_mask = data.train_mask | data.val_mask | data.test_mask

    pred_all = (output[all_mask].squeeze() > 0).type_as(data.y)
    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)


    F1s['all'] = f1_score(data.y[all_mask].cpu().numpy(), pred_all.cpu().numpy())
    F1s['val'] = f1_score(data.y[data.val_mask].cpu().numpy(), pred_val.cpu().numpy())
    F1s['test'] = f1_score(data.y[data.test_mask].cpu().numpy(), pred_test.cpu().numpy())

    for split_name, mask in {
        'all': all_mask,
        'val': data.val_mask,
        'test': data.test_mask,
    }.items():
        accs[split_name], auc_rocs[split_name], paritys[split_name], equalitys[split_name] = _ember_metrics(
            output[mask], data.y[mask], data.sens[mask]
        )

    return accs, auc_rocs, F1s, paritys, equalitys


# def evaluate_ged4(x, classifier, discriminator, generator, encoder, data, args, ood):
#     classifier.eval()
#     generator.eval()
#     encoder.eval()
#
#     if ood == 2:
#         data.test_mask =
#
#     with torch.no_grad():
#         if(args.f_mask == 'yes'):
#             outputs, loss = [], 0
#             feature_weights = generator()
#             for k in range(args.K):
#                 x = data.x * F.gumbel_softmax(
#                     feature_weights, tau=1, hard=True)[:, 0]
#
#                 h = encoder(x, data.edge_index, data.adj_norm_sp)
#                 output = classifier(h)
#                 # output2 = discriminator(h)
#
#                 # loss += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
#                 #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))
#
#                 outputs.append(output)
#
#             # loss_val = loss / args.K
#
#             output = torch.stack(outputs).mean(dim=0)
#         else:
#             h = encoder(data.x, data.edge_index, data.adj_norm_sp)
#             output = classifier(h)
#             # output2 = discriminator(h)
#
#             # loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
#             #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))
#
#     accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}
#
#     pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
#     pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)
#
#     accs['val'] = pred_val.eq(
#         data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
#     accs['test'] = pred_test.eq(
#         data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
#
#     F1s['val'] = f1_score(data.y[data.val_mask].cpu(
#     ).numpy(), pred_val.cpu().numpy())
#
#     F1s['test'] = f1_score(data.y[data.test_mask].cpu(
#     ).numpy(), pred_test.cpu().numpy())
#
#     auc_rocs['val'] = roc_auc_score(
#         data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
#     auc_rocs['test'] = roc_auc_score(
#         data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())
#
#     paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
#     ).numpy(), data.sens[data.val_mask].cpu().numpy())
#
#     paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
#     ).numpy(), data.sens[data.test_mask].cpu().numpy())
#
#     return accs, auc_rocs, F1s, paritys, equalitys
