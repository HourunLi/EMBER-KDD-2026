import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils import fair_metric, InfoNCE, random_aug, consis_loss


def evaluate_per_target_class(output, data):
    """Evaluate logits with the per-target-class protocol from ``learn(1).py``.

    Accuracy and F1 are computed independently for every target class and then
    macro-averaged.  AUC is likewise computed one-vs-rest for each target class;
    for class 0 its score is ``1 - P(y=1)``.  DP and EO are computed from the
    thresholded predictions.  All returned values are percentages.
    """
    probabilities = torch.sigmoid(output.view(-1)).detach().cpu().numpy()
    labels = data.y.detach().cpu().numpy()
    sensitive = data.sens.detach().cpu().numpy()
    all_mask = data.train_mask | data.val_mask | data.test_mask
    splits = {
        'all': all_mask.detach().cpu().numpy(),
        'train': data.train_mask.detach().cpu().numpy(),
        'val': data.val_mask.detach().cpu().numpy(),
        'test': data.test_mask.detach().cpu().numpy(),
    }

    accs, auc_rocs, f1s, paritys, equalitys = {}, {}, {}, {}, {}
    for split_name, mask in splits.items():
        y_true = labels[mask]
        sens = sensitive[mask]
        prob = probabilities[mask]
        pred = (prob > 0.5).astype(int)
        has_both_classes = len(np.unique(y_true)) == 2

        target_metrics = []
        for target in np.unique(y_true):
            target_idx = y_true == target
            target_truth = target_idx.astype(int)
            target_pred = (pred == target).astype(int)
            target_prob = prob if target == 1 else 1 - prob
            target_metrics.append({
                'acc': accuracy_score(
                    y_true[target_idx], pred[target_idx]) * 100,
                'auc': roc_auc_score(target_truth, target_prob) * 100
                       if has_both_classes else float('nan'),
                'f1': f1_score(
                    target_truth, target_pred, zero_division=0) * 100,
            })

        dp, eo = fair_metric(pred, y_true, sens)
        accs[split_name] = float(np.nanmean(
            [metrics['acc'] for metrics in target_metrics]))
        auc_rocs[split_name] = float(np.nanmean(
            [metrics['auc'] for metrics in target_metrics]))
        f1s[split_name] = float(np.nanmean(
            [metrics['f1'] for metrics in target_metrics]))
        paritys[split_name] = dp * 100
        equalitys[split_name] = eo * 100

    return accs, auc_rocs, f1s, paritys, equalitys


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
                output = classifier(h)
                # output2 = discriminator(h)

                # loss += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
                #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

                outputs.append(output)

            # loss_val = loss / args.K

            output = torch.stack(outputs).mean(dim=0)
        else:
            h = encoder(data.x, data.edge_index, data.adj_norm_sp)
            output = classifier(h)
            # output2 = discriminator(h)

            # loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
            #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    return evaluate_per_target_class(output, data)
