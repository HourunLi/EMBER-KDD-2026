import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score, roc_auc_score
from utils import fair_metric, InfoNCE, random_aug, consis_loss


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

    # accs['val'] = pred_val.eq(
    #     data.y[data.val_mask]).sum().item() / data.val_mask.sum()
    # accs['test'] = pred_test.eq(
    #     data.y[data.test_mask]).sum().item() / data.test_mask.sum()
    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    y_val = data.y[data.val_mask]
    tmp_acc = []
    for group in [0, 1]:
        group_mask = (y_val == group)
        tmp_acc.append(pred_val[group_mask].eq(y_val[group_mask]).sum().item() / group_mask.sum().item())
    accs['val'] = sum(tmp_acc) / len(tmp_acc)

    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)
    y_test = data.y[data.test_mask]
    tmp_acc = []
    for group in [0, 1]:
        group_mask = (y_test == group)
        tmp_acc.append(pred_test[group_mask].eq(y_test[group_mask]).sum().item() / group_mask.sum().item())
    accs['test'] = sum(tmp_acc) / len(tmp_acc)

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
                x = data.x * F.gumbel_softmax(feature_weights, tau=1, hard=False)[:, 0]

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

                h = encoder(x, data.edge_index)
                output = classifier(h)
                # output2 = discriminator(h)

                # loss += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
                #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

                outputs.append(output)

            # loss_val = loss / args.K

            output = torch.stack(outputs).mean(dim=0)
        else:
            h = encoder(data.x, data.edge_index)
            output = classifier(h)
            # output2 = discriminator(h)

            # loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
            #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    # print(torch.min(output[data.test_mask]))
    # print("****************************evaluate****************************")
    # print("label == 0")
    # output_0 = output[data.test_mask][data.y[data.test_mask] == 0].to(args.device)
    # quantiles_0 = torch.quantile(output_0, torch.tensor([0.25, 0.5, 0.75]).to(args.device))
    # print(f"min, max, mean, std, quantiles:, {torch.min(output_0)}, {torch.max(output_0)}, {torch.mean(output_0)}, {torch.std(output_0)}, {quantiles_0}")
    # print("label == 1")
    # output_1 = output[data.test_mask][data.y[data.test_mask] == 1].to(args.device)
    # quantiles_1 = torch.quantile(output_1, torch.tensor([0.25, 0.5, 0.75]).to(args.device))
    # print(f"min, max, mean, std, quantiles:, {torch.min(output_1)}, {torch.max(output_1)}, {torch.mean(output_1)}, {torch.std(output_1)}, {quantiles_1}")
    
    pred_val = (output[data.val_mask].squeeze() > 0.5).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0.5).type_as(data.y)
    # pred_val = torch.argmax(output[data.val_mask], dim=1) 
    # pred_test = torch.argmax(output[data.test_mask], dim=1) 
    
    # print(sum(pred_test == 0), sum(data.y[data.test_mask]==0))
    accs['val'] = pred_val.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    accs['test'] = pred_test.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu().numpy(), pred_val.cpu().numpy())
    F1s['test'] = f1_score(data.y[data.test_mask].cpu().numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu().numpy(), data.sens[data.val_mask].cpu().numpy())
    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu().numpy(), data.sens[data.test_mask].cpu().numpy())

    # print(accs, auc_rocs, F1s, paritys, equalitys)
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
