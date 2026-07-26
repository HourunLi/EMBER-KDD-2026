from dataset import *
from model import *
from utils import *
from learn import *
import argparse
from tqdm import tqdm
from torch import tensor
import warnings
warnings.filterwarnings('ignore')
import math
from constraint import Constraint_SAGE
import scipy.optimize as sopt
from torch.optim.lr_scheduler import ExponentialLR
import time
import copy
import json
import os
import subprocess
import sys
import tempfile
import numpy as np

cnst_config = {
    'lr_scheduler': True,
    'continue_training': False,
    # 'with_constraint': True,
    'nit': 100,
    'criterion': 100,  # 100
    'cnst': 0.01,
    'alpha': 2.1,
    'rho': 20 # 50, 100
}

CROSS_DOMAIN_PAIRS = {
    'bailA': ('bailA_2', 'bailA_1'),
    'germanA': ('germanA_2', 'germanA_1'),
    'pokec': ('pokec_z', 'pokec_n'),
    'syn': ('syn-2', 'syn-1'),
}

def loss_cvar(loss_vector, alpha):
  batch_size = len(loss_vector)
  n = int(alpha * batch_size)
  rk = torch.argsort(loss_vector, descending=True)
  loss = loss_vector[rk[:n]].mean()  
  return loss


def loss_chisq(loss_vector, alpha):
  max_l = 10.
  C = math.sqrt(1 + (1 / alpha - 1) ** 2)
  foo = lambda eta: C * math.sqrt((F.relu(loss_vector - eta) ** 2).mean().item()) + eta
  opt_eta = sopt.brent(foo, brack=(0, max_l))
  loss = C * torch.sqrt((F.relu(loss_vector - opt_eta) ** 2).mean()) + opt_eta
  return loss


def cvar_doro(loss_vector, alpha, eps):
  gamma = eps + alpha * (1 - eps)
  batch_size = len(loss_vector)
  n1 = int(gamma * batch_size)
  n2 = int(eps * batch_size)
  rk = torch.argsort(loss_vector, descending=True)
  loss = loss_vector[rk[n2:n1]].sum() / alpha / (batch_size - n2)  
  return loss

def chisq_doro(loss_vector, alpha, eps):
  max_l = 10.
  batch_size = len(loss_vector)
  C = math.sqrt(1 + (1 / alpha - 1) ** 2)
  n = int(eps * batch_size)
  rk = torch.argsort(loss_vector, descending=True)
  l0 = loss_vector[rk[n:]]
  foo = lambda eta: C * math.sqrt((F.relu(l0 - eta) ** 2).mean().item()) + eta
  opt_eta = sopt.brent(foo, brack=(0, max_l))
  loss = C * torch.sqrt((F.relu(l0 - opt_eta) ** 2).mean()) + opt_eta
  return loss


def checkpoint_path_for_run(checkpoint_path, run_index, total_runs):
    if total_runs == 1:
        return checkpoint_path

    root, extension = os.path.splitext(checkpoint_path)
    if extension == '':
        extension = '.pt'
    return '{}_run{}{}'.format(root, run_index, extension)


def run(data, args, target_data=None, source_only_train=False):
    pbar = tqdm(range(args.runs), unit='run')
    criterion = nn.BCELoss()
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)
    if source_only_train:
        final_data = None
        final_mask = None
        final_split_name = None
    elif target_data is None:
        final_data = data
        final_mask = data.test_mask
        final_split_name = 'Source Test'
    else:
        final_data = target_data.to(args.device)
        final_mask = final_data.test_mask
        final_split_name = 'Target Test'

    generator = channel_masker(args).to(args.device)
    optimizer_g = torch.optim.Adam([
        dict(params=generator.weights, weight_decay=args.g_wd)], lr=args.g_lr)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    if(args.encoder == 'MLP'):
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GCN'):
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.bias, weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GIN'):
        encoder = GIN_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'SAGE'):
        neurons_per_layer= [args.num_features, args.hidden, args.hidden]
        encoder = SAGE_encoder(args, neurons_per_layer).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)  

    os.makedirs('logs', exist_ok=True)
    ResLogFile = os.path.join('logs', args.experiment_name) + "_ResLog.txt"
    for count in pbar:
        seed_everything(count + args.seed)
        generator.reset_parameters()
        discriminator.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()

        # Optimizer moments must not leak from one repeated run into another.
        optimizer_g.state.clear()
        optimizer_d.state.clear()
        optimizer_c.state.clear()
        optimizer_e.state.clear()

        best_val_tradeoff = -math.inf
        best_epoch = -1
        best_states = None
        cnt = 0
        for epoch in range(0, args.epochs):
            if(args.f_mask == 'yes'):
                generator.eval()
                feature_weights, masks, = generator(), []
                for k in range(args.K):
                    mask = F.gumbel_softmax(feature_weights, tau=1, hard=False)[:, 0]
                    masks.append(mask)

            # train discriminator to recognize the sensitive group
            discriminator.train()
            encoder.train()
            for epoch_d in range(0, args.d_epochs):
                optimizer_d.zero_grad()
                optimizer_e.zero_grad()

                if(args.f_mask == 'yes'):
                    loss_d = 0

                    for k in range(args.K):
                        x = data.x * masks[k].detach()
                        # h = encoder(x, data.edge_index, data.adj_norm_sp)
                        h = encoder(x, data.edge_index)
                        output = discriminator(h)

                        loss_d += criterion(output.view(-1), data.x[:, args.sens_idx])

                    loss_d = loss_d / args.K
                else:
                    # h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    h = encoder(data.x, data.edge_index)
                    output = discriminator(h)
                    
                    loss_d = criterion(output.view(-1), data.x[:, args.sens_idx])

                loss_d.backward()
                optimizer_d.step()
                optimizer_e.step()

            # train classifier
            classifier.train()
            encoder.train()
            for epoch_c in range(0, args.c_epochs):
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()

                if(args.f_mask == 'yes'):
                    # # Save previous weights
                    # num_layers = len(encoder.gcn_stack)
                    # old_weights = [None] * num_layers
                    # for i, gcn_block in enumerate(encoder.gcn_stack):
                    #     old_weights[i] = {'lin_r': torch.zeros_like(gcn_block.lin_r.weight.data),
                    #                     'lin_l': torch.zeros_like(gcn_block.lin_l.weight.data)}                          
                    loss_c = 0
                    for k in range(args.K):
                        x = data.x * masks[k].detach()
                        # h = encoder(x, data.edge_index, data.adj_norm_sp)
                        h = encoder(x, data.edge_index)
                        output = classifier(h)
                        
                        # for i, gcn_block in enumerate(encoder.gcn_stack):
                        #         w0 = torch.clone(gcn_block.lin_r.weight.detach().data)
                        #         w1 = torch.clone(gcn_block.lin_l.weight.detach().data)
                        #         old_weights[i]['lin_r'] += w0
                        #         old_weights[i]['lin_l'] += w1

                        loss_c += F.binary_cross_entropy_with_logits(
                            output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                    # # Average the accumulated weights
                    # for i in range(num_layers):
                    #         old_weights[i]['lin_r'] /= args.K
                    #         old_weights[i]['lin_l'] /= args.K
                    #         old_weights[i] = [old_weights[i]['lin_r'], old_weights[i]['lin_l']]
                            
                    loss_c = loss_c / args.K

                else:
                    # h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    h = encoder(data.x, data.edge_index)
                    output = classifier(h)

                    loss_c = F.binary_cross_entropy_with_logits(
                        output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                loss_c.backward()

                optimizer_e.step()
                optimizer_c.step()  
                
                # # Apply constraint
                # cnst_config['num_layers'] = len(neurons_per_layer) - 1
                # cnst_config['neurons_per_layer'] = neurons_per_layer
                # constraint = Constraint_SAGE(encoder, parameters=cnst_config, device=args.device, with_constraint=args.with_constraint)
                # if epoch >= 100:
                #     constraint.on_batch_end(old_weights)    

            # train generator to fool discriminator
            generator.train()
            encoder.train()
            discriminator.eval()
            for epoch_g in range(0, args.g_epochs):
                optimizer_g.zero_grad()
                optimizer_e.zero_grad()

                if(args.f_mask == 'yes'):
                    # Save previous weights
                    num_layers = len(encoder.gcn_stack)
                    old_weights = [None] * num_layers
                    generator_weights = None
                    for i, gcn_block in enumerate(encoder.gcn_stack):
                        old_weights[i] = {'lin_r': torch.zeros_like(gcn_block.lin_r.weight.data),
                                        'lin_l': torch.zeros_like(gcn_block.lin_l.weight.data)}
                    
                    loss_g = 0
                    loss_g1 = 0
                    loss_g2 = 0
                    feature_weights = generator()
                    for k in range(args.K):
                        mask = F.gumbel_softmax(feature_weights, tau=1, hard=False)[:, 0]

                        x = data.x * mask
                        # h = encoder(x, data.edge_index, data.adj_norm_sp)
                        h = encoder(x, data.edge_index)
                        output = discriminator(h)
                        
                        
                        generator_weights = generator.weights[:, 0]
                        for i, gcn_block in enumerate(encoder.gcn_stack):
                                w0 = torch.clone(gcn_block.lin_r.weight.detach().data)
                                w1 = torch.clone(gcn_block.lin_l.weight.detach().data)
                                old_weights[i]['lin_r'] += w0
                                old_weights[i]['lin_l'] += w1

                        loss_g1 += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1)), reduction='none') 
                        loss_g2 += args.ratio * F.mse_loss(mask.view(-1), torch.ones_like(mask.view(-1)))
                        loss_g = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + \
                            args.ratio * F.mse_loss(mask.view(-1), torch.ones_like(mask.view(-1))) 
                    
                    # Average the accumulated weights
                    for i in range(num_layers):
                        old_weights[i]['lin_r'] /= args.K
                        old_weights[i]['lin_l'] /= args.K
                        old_weights[i] = [old_weights[i]['lin_r'], old_weights[i]['lin_l']]
                    
                    loss_g1 = loss_g1 / args.K
                    loss_g2 = loss_g2 / args.K
                    # loss_g = loss_g / args.K
                    # solve outlier
                    loss_g = loss_chisq(loss_g1, args.loss_alpha) + loss_g2
                    # loss_g = cvar_doro(loss_g1, args.loss_alpha, args.eps) + loss_g2
                    
                    
                else:
                    # h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    h = encoder(data.x, data.edge_index)
                    output = discriminator(h)
                    
                    loss_g = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1)))

                loss_g.backward()

                optimizer_g.step()
                optimizer_e.step()
                
                # Apply constraint
                cnst_config['num_layers'] = len(neurons_per_layer) - 1
                cnst_config['neurons_per_layer'] = neurons_per_layer
                cnst_config['rho'] = args.rho
                constraint = Constraint_SAGE(encoder, parameters=cnst_config, device=args.device, with_constraint=args.with_constraint)
                constraint.on_batch_end(old_weights, generator_weights)    
        
            
            if(args.weight_clip == 'yes'):
                if(args.f_mask == 'yes'):
                    weights = torch.stack(masks).mean(dim=0)
                else:
                    weights = torch.ones_like(data.x[0])

                encoder.clip_parameters(weights)

            if source_only_train:
                print(epoch, 'Source Train -- supervised nodes:',
                      data.train_mask.sum().item())
                continue

            # Model selection uses source-domain validation data only. Target
            # labels are excluded from fitting and checkpoint selection.
            val_metrics = evaluate_ged3_on_mask(
                classifier, generator, encoder, data, data.val_mask, args)

            print(epoch, 'Source Val -- Acc:', val_metrics['acc'],
                  'AUC_ROC:', val_metrics['auc_roc'], 'F1:', val_metrics['f1'],
                  'Parity:', val_metrics['parity'],
                  'Equality:', val_metrics['equality'])
            with open(ResLogFile, "a+") as file:
                now = time.localtime()
                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
                strLog = formatted_time + \
                    "epoch {} Source Val -- ACC: {:.4f} AUC_ROC:{:.4f} F1:{:.4f} DP:{:.4f} EO:{:.4f}".format(
                        epoch, val_metrics['acc'], val_metrics['auc_roc'],
                        val_metrics['f1'], val_metrics['parity'],
                        val_metrics['equality'])
                file.write(strLog + "\n")


            # visualize mask weights
            # ResLogFile1 = os.path.join('logs', args.dataset) + "_ResLog_mask-origin.txt"
            # with open(ResLogFile1, "a+") as file:
            #     now = time.localtime()
            #     formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
            #     strLog = formatted_time + str(generator.weights[:, 0])
            #     file.write(strLog +"\n") 
            
            # visualize gnn encoder weights
            # import seaborn as sns
            # sns.heatmap(encoder.gcn_stack[1].lin_r.weight.detach().data[3:13,3:13].cpu().detach().numpy(), annot=True, cmap='YlGnBu') 
            # plt.savefig("./maps/ct-heatmap-" + str(epoch) + ".png")
            # # plt.show()
            # plt.clf()  # 清除当前图形      
            
            # Early stopping and checkpoint selection depend only on the
            # source-domain validation trade-off.
            cur_val_tradeoff = (
                val_metrics['auc_roc'] + val_metrics['f1'] + val_metrics['acc']
                - args.alpha * (val_metrics['parity'] + val_metrics['equality']))
            if cur_val_tradeoff > best_val_tradeoff:
                best_val_tradeoff = cur_val_tradeoff
                best_epoch = epoch
                cnt = 0
                best_states = {
                    'generator': copy.deepcopy(generator.state_dict()),
                    'discriminator': copy.deepcopy(discriminator.state_dict()),
                    'classifier': copy.deepcopy(classifier.state_dict()),
                    'encoder': copy.deepcopy(encoder.state_dict()),
                }
            elif epoch >= args.pretrain:
                cnt += 1

            if epoch >= args.pretrain and cnt >= args.early_stopping:
                print("Early Stopping at epoch {}, best source-validation epoch {}".format(
                    epoch, best_epoch))
                break

        if source_only_train:
            run_checkpoint_path = checkpoint_path_for_run(
                args.checkpoint, count, args.runs)
            checkpoint_directory = os.path.dirname(run_checkpoint_path)
            if checkpoint_directory:
                os.makedirs(checkpoint_directory, exist_ok=True)

            # Store inference model parameters only. The target process does
            # not need source data, optimizers, losses, or normalization stats.
            torch.save({
                'generator': generator.state_dict(),
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
            }, run_checkpoint_path)
            print("Saved source-trained model parameters to '{}'".format(
                run_checkpoint_path))
            continue

        if best_states is None:
            raise RuntimeError("No source-domain validation checkpoint was recorded.")

        generator.load_state_dict(best_states['generator'])
        discriminator.load_state_dict(best_states['discriminator'])
        classifier.load_state_dict(best_states['classifier'])
        encoder.load_state_dict(best_states['encoder'])

        # In a cross-domain run target labels are used only for this final
        # evaluation. For same-domain runs, final_data is the source test set.
        final_metrics = evaluate_ged3_on_mask(
            classifier, generator, encoder, final_data, final_mask, args)
        print('===== {} (best source-validation epoch {}) ====='.format(
            final_split_name, best_epoch))
        print('Acc:', final_metrics['acc'], 'AUC_ROC:', final_metrics['auc_roc'],
              'F1:', final_metrics['f1'], 'Parity:', final_metrics['parity'],
              'Equality:', final_metrics['equality'])

        if args.with_constraint:
            theta_bar = get_Lips_constant_upper(encoder)
            print("######get_Lips_constant_upper", theta_bar)
        else:
            theta_bar = get_Lips_constant(encoder)
            print("######get_Lips_constant", theta_bar)
        
        acc[count] = final_metrics['acc']
        f1[count] = final_metrics['f1']
        auc_roc[count] = final_metrics['auc_roc']
        parity[count] = final_metrics['parity']
        equality[count] = final_metrics['equality']

        # print('auc_roc:', np.mean(auc_roc[:(count + 1)]))
        # print('f1:', np.mean(f1[:(count + 1)]))
        # print('acc:', np.mean(acc[:(count + 1)]))
        # print('Statistical parity:', np.mean(parity[:(count + 1)]))
        # print('Equal Opportunity:', np.mean(equality[:(count + 1)]))
        with open(ResLogFile, "a+") as file:
            now = time.localtime()
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
            strLog = formatted_time + \
                "best {} -- ACC: {:.4f} AUC_ROC:{:.4f} F1:{:.4f} DP:{:.4f} EO:{:.4f}".format(
                    final_split_name, acc[count], auc_roc[count], f1[count],
                    parity[count], equality[count])
            file.write(strLog + "\n")

    return acc, f1, auc_roc, parity, equality


def encode_prediction_sensitive_groups(predictions, sensitive_attributes):
    """Encode binary (predicted Y, S) pairs for the visualization scripts.

    The encoding follows visualization/export_utils.py in the zyt branch:
      0 -> predicted Y=1, S=0
      1 -> predicted Y=1, S=1
      2 -> predicted Y=0, S=0
      3 -> predicted Y=0, S=1
    """
    predictions = np.asarray(predictions).reshape(-1)
    sensitive_attributes = np.asarray(sensitive_attributes).reshape(-1)

    if predictions.shape[0] != sensitive_attributes.shape[0]:
        raise ValueError(
            "Predictions and sensitive attributes must have the same length, "
            "got {} and {}.".format(
                predictions.shape[0], sensitive_attributes.shape[0]))
    if not np.isin(predictions, [0, 1]).all():
        raise ValueError(
            "Target predictions used for visualization must be binary (0/1).")
    if not np.isin(sensitive_attributes, [0, 1]).all():
        raise ValueError(
            "Target sensitive attributes used for visualization must be "
            "binary (0/1).")

    predictions = predictions.astype(np.int64, copy=False)
    sensitive_attributes = sensitive_attributes.astype(np.int64, copy=False)
    labels = np.full(predictions.shape[0], -1, dtype=np.int64)
    labels[(predictions == 1) & (sensitive_attributes == 0)] = 0
    labels[(predictions == 1) & (sensitive_attributes == 1)] = 1
    labels[(predictions == 0) & (sensitive_attributes == 0)] = 2
    labels[(predictions == 0) & (sensitive_attributes == 1)] = 3

    if (labels < 0).any() or not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError(
            "Failed to encode every target node into one of groups 0, 1, 2, 3.")
    return labels


def target_visualization_paths():
    """Return the standard artifact paths in the directory containing sfg.py."""
    output_directory = os.path.dirname(os.path.abspath(__file__))
    return (
        os.path.join(output_directory, 'feat.npz'),
        os.path.join(output_directory, 'labels.npz'),
    )


def save_target_visualization_artifacts(artifacts):
    """Validate and save one seed's target representations and group labels."""
    required_keys = {
        'representations', 'predictions', 'sensitive_attributes'}
    missing_keys = required_keys.difference(artifacts.keys())
    if missing_keys:
        raise KeyError(
            "Target visualization artifacts are missing keys: {}."
            .format(sorted(missing_keys)))

    representations = (
        artifacts['representations'].detach().cpu().numpy())
    predictions = artifacts['predictions'].detach().cpu().numpy()
    sensitive_attributes = (
        artifacts['sensitive_attributes'].detach().cpu().numpy())
    labels = encode_prediction_sensitive_groups(
        predictions, sensitive_attributes)

    if representations.ndim != 2:
        raise ValueError(
            "representations must have shape [num_valid_target_all_nodes, "
            "feature_dim], got {}.".format(representations.shape))
    if representations.shape[0] != labels.shape[0]:
        raise ValueError(
            "representations and labels length mismatch: {} vs {}."
            .format(representations.shape[0], labels.shape[0]))
    if not np.isfinite(representations).all():
        invalid_count = representations.size - int(
            np.isfinite(representations).sum())
        raise ValueError(
            "representations contains {} NaN or Inf values; refusing to save."
            .format(invalid_count))
    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape [num_valid_target_all_nodes], got {}."
            .format(labels.shape))
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError(
            "labels may contain only the joint groups 0, 1, 2, 3.")

    feat_path, labels_path = target_visualization_paths()
    np.savez_compressed(
        feat_path, representations=representations)
    np.savez_compressed(
        labels_path, labels=labels.astype(np.int64, copy=False))
    return feat_path, labels_path


def test_target_domain(data, args):
    """Load inference parameters and evaluate a target graph without updates."""
    if args.encoder != 'SAGE':
        raise ValueError("Target-only SFG evaluation currently requires --encoder='SAGE'.")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            "Checkpoint '{}' does not exist.".format(args.checkpoint))

    seed_everything(args.seed)
    data = data.to(args.device)

    generator = channel_masker(args).to(args.device)
    encoder = SAGE_encoder(
        args, [args.num_features, args.hidden, args.hidden]).to(args.device)
    classifier = MLP_classifier(args).to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    required_keys = {'generator', 'encoder', 'classifier'}
    missing_keys = required_keys.difference(checkpoint.keys())
    if missing_keys:
        raise KeyError(
            "Checkpoint '{}' is missing model parameters: {}."
            .format(args.checkpoint, sorted(missing_keys)))

    generator.load_state_dict(checkpoint['generator'], strict=True)
    encoder.load_state_dict(checkpoint['encoder'], strict=True)
    classifier.load_state_dict(checkpoint['classifier'], strict=True)

    for model in (generator, encoder, classifier):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    metrics, artifacts = evaluate_ged3_on_mask(
        classifier, generator, encoder, data, data.test_mask, args,
        return_artifacts=True)
    feat_path, labels_path = save_target_visualization_artifacts(artifacts)
    print('===== Target Test (%): {} ====='.format(args.target_domain))
    print('Acc:', metrics['acc'] * 100, 'AUC_ROC:', metrics['auc_roc'] * 100,
          'F1:', metrics['f1'] * 100, 'Parity:', metrics['parity'] * 100,
          'Equality:', metrics['equality'] * 100)
    print("Saved target representations to '{}' with shape {}.".format(
        feat_path, tuple(artifacts['representations'].shape)))
    print("Saved target joint-group labels to '{}' with shape {}.".format(
        labels_path, tuple(artifacts['predictions'].shape)))
    return metrics


def checkpoint_path_for_seed(checkpoint_path, seed):
    root, extension = os.path.splitext(checkpoint_path)
    if extension == '':
        extension = '.pt'
    return '{}_seed{}{}'.format(root, seed, extension)


def append_cli_arguments(command, args, argument_names):
    for argument_name in argument_names:
        command.extend([
            '--{}'.format(argument_name),
            str(getattr(args, argument_name)),
        ])


def run_cross_domain_seeds(args):
    """Coordinate isolated source-train and target-test subprocesses."""
    supported_families = set(CROSS_DOMAIN_PAIRS)
    if args.dataset not in supported_families:
        raise ValueError(
            "cross_domain supports {}, got '{}'.".format(
                sorted(supported_families), args.dataset))
    if args.source_domain is None or args.target_domain is None:
        raise ValueError(
            "cross_domain requires both --source_domain and --target_domain.")
    if args.encoder != 'SAGE':
        raise ValueError("Cross-domain SFG requires --encoder='SAGE'.")
    expected_pair = CROSS_DOMAIN_PAIRS[args.dataset]
    actual_pair = (args.source_domain, args.target_domain)
    if actual_pair != expected_pair:
        raise ValueError(
            "Dataset family '{}' expects source/target domains {}, got {}."
            .format(args.dataset, expected_pair, actual_pair))

    script_path = os.path.abspath(__file__)
    training_argument_names = [
        'dataset', 'encoder', 'epochs', 'd_epochs', 'g_epochs', 'c_epochs',
        'g_lr', 'g_wd', 'd_lr', 'd_wd', 'c_lr', 'c_wd', 'e_lr', 'e_wd',
        'prop', 'dropout', 'hidden', 'K', 'top_k', 'clip_e', 'f_mask',
        'weight_clip', 'ratio', 'alpha', 'loss_alpha', 'eps', 'rho',
    ]
    testing_argument_names = [
        'dataset', 'encoder', 'prop', 'dropout', 'hidden', 'K', 'top_k',
        'f_mask',
    ]

    seeds = range(1, 6)
    per_seed_results = []
    with tempfile.TemporaryDirectory(prefix='sfg_cross_domain_') as metrics_directory:
        for seed in seeds:
            seed_checkpoint = checkpoint_path_for_seed(args.checkpoint, seed)
            seed_metrics_path = os.path.join(
                metrics_directory, 'seed_{}_metrics.json'.format(seed))

            train_command = [
                sys.executable, script_path,
                '--mode', 'source_train',
                '--source_domain', args.source_domain,
                '--checkpoint', seed_checkpoint,
                '--seed', str(seed),
                '--runs', '1',
            ]
            append_cli_arguments(
                train_command, args, training_argument_names)
            if args.with_constraint:
                train_command.append('--with_constraint')

            print('\n===== Seed {}: source training ====='.format(seed), flush=True)
            subprocess.run(train_command, check=True)

            test_command = [
                sys.executable, script_path,
                '--mode', 'target_test',
                '--target_domain', args.target_domain,
                '--checkpoint', seed_checkpoint,
                '--metrics_output', seed_metrics_path,
                '--seed', str(seed),
            ]
            append_cli_arguments(
                test_command, args, testing_argument_names)

            print('\n===== Seed {}: isolated target testing ====='.format(seed), flush=True)
            subprocess.run(test_command, check=True)

            with open(seed_metrics_path, 'r') as metrics_file:
                metrics = json.load(metrics_file)
            per_seed_results.append({
                'seed': seed,
                'accuracy': float(metrics['acc']) * 100,
                'roc_auc': float(metrics['auc_roc']) * 100,
                'parity': float(metrics['parity']) * 100,
                'equality': float(metrics['equality']) * 100,
            })

    summary = {}
    for metric_name in ('accuracy', 'roc_auc', 'parity', 'equality'):
        metric_values = np.asarray(
            [result[metric_name] for result in per_seed_results],
            dtype=np.float64)
        summary[metric_name] = {
            'mean': float(np.mean(metric_values)),
            'std': float(np.std(metric_values)),
        }

    print('\n===== Five-seed target-domain results (%) =====')
    for result in per_seed_results:
        print(
            "Seed {seed}: Accuracy={accuracy:.6f}, ROC_AUC={roc_auc:.6f}, "
            "Parity={parity:.6f}, Equality={equality:.6f}".format(**result))

    print('\n===== Mean +/- Std (%) =====')
    display_names = {
        'accuracy': 'Accuracy',
        'roc_auc': 'ROC_AUC',
        'parity': 'Parity',
        'equality': 'Equality',
    }
    for metric_name in ('accuracy', 'roc_auc', 'parity', 'equality'):
        print('{}: {:.2f} +/- {:.2f}'.format(
            display_names[metric_name],
            summary[metric_name]['mean'],
            summary[metric_name]['std']))

    formatted_summary = {
        display_names[metric_name]: '{:.2f} +/- {:.2f}'.format(
            summary[metric_name]['mean'], summary[metric_name]['std'])
        for metric_name in ('accuracy', 'roc_auc', 'parity', 'equality')
    }

    summary_output = args.summary_output
    if summary_output is None:
        os.makedirs('logs', exist_ok=True)
        summary_output = os.path.join(
            'logs',
            '{}_to_{}_five_seeds.json'.format(
                args.source_domain, args.target_domain))
    else:
        summary_directory = os.path.dirname(summary_output)
        if summary_directory:
            os.makedirs(summary_directory, exist_ok=True)

    with open(summary_output, 'w') as summary_file:
        json.dump({
            'source_domain': args.source_domain,
            'target_domain': args.target_domain,
            'seeds': per_seed_results,
            'summary': summary,
            'formatted_summary': formatted_summary,
            'unit': 'percentage points',
            'std_definition': 'population standard deviation (ddof=0)',
        }, summary_file, indent=2)
    print("Saved five-seed summary to '{}'".format(summary_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--g_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=5)
    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--g_wd', type=float, default=0)
    parser.add_argument('--d_lr', type=float, default=0.001)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.001)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.001)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--clip_e', type=float, default=0.1)
    parser.add_argument('--f_mask', type=str, default='yes')
    parser.add_argument('--weight_clip', type=str, default='no')
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--with_constraint', action='store_true', default=False)
    parser.add_argument('--pretrain', type=int, default=200)
    parser.add_argument('--loss_alpha', type=float, default=1)
    parser.add_argument('--eps', type=float, default=0)
    parser.add_argument('--rho', type=float, default=100)
    parser.add_argument('--mode', choices=[
        'standard', 'source_train', 'target_test', 'cross_domain'],
                        default='standard')
    parser.add_argument('--source_domain', type=str, default=None,
                        help="Source-domain file prefix, e.g. bailA_2.")
    parser.add_argument('--target_domain', type=str, default=None,
                        help="Target-domain file prefix, e.g. bailA_1.")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sfg.pt',
                        help="Model-parameter file written by source_train or read by target_test.")
    parser.add_argument('--metrics_output', type=str, default=None,
                        help="Optional JSON output used by isolated target_test runs.")
    parser.add_argument('--summary_output', type=str, default=None,
                        help="Optional JSON output for the five-seed cross-domain summary.")
    parser.add_argument('--split_seed', type=int, default=20,
                        help="Fixed seed for the standard-mode train/validation split.")
    

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'cross_domain':
        run_cross_domain_seeds(args)

    elif args.mode == 'source_train':
        supported_families = set(CROSS_DOMAIN_PAIRS)
        if args.dataset not in supported_families:
            raise ValueError(
                "source_train supports {}, got '{}'.".format(
                    sorted(supported_families), args.dataset))
        if args.source_domain is None:
            parser.error("source_train requires --source_domain.")
        if args.target_domain is not None:
            parser.error(
                "source_train must not receive --target_domain; the target domain is not loaded.")
        if args.encoder != 'SAGE':
            raise ValueError("Source-domain SFG training requires --encoder='SAGE'.")

        data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_domain_dataset(
            args.dataset, args.source_domain, 'full_train', args.top_k)
        expected_supervised_nodes = data.valid_label_mask.sum().item()
        if (data.train_mask.sum().item() != expected_supervised_nodes
                or data.val_mask.any().item() or data.test_mask.any().item()):
            raise RuntimeError(
                "source_train requires every labeled source node in train_mask "
                "and no hold-out nodes.")
        args.num_features, args.num_classes = data.x.shape[1], 1
        args.experiment_name = args.source_domain

        print("Source-only training domain: '{}'".format(args.source_domain))
        print("All {} labeled source nodes (out of {} total nodes) are included "
              "in the supervised train mask.".format(
                  data.train_mask.sum().item(), data.x.shape[0]))
        run(data, args, source_only_train=True)

    elif args.mode == 'target_test':
        supported_families = set(CROSS_DOMAIN_PAIRS)
        if args.dataset not in supported_families:
            raise ValueError(
                "target_test supports {}, got '{}'.".format(
                    sorted(supported_families), args.dataset))
        if args.target_domain is None:
            parser.error("target_test requires --target_domain.")
        if args.source_domain is not None:
            parser.error(
                "target_test must not receive --source_domain; source data is not loaded.")

        data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_domain_dataset(
            args.dataset, args.target_domain, 'full_test', args.top_k)
        expected_test_nodes = torch.logical_and(
            data.valid_label_mask, data.valid_sensitive_mask).sum().item()
        if (data.test_mask.sum().item() != expected_test_nodes
                or data.train_mask.any().item() or data.val_mask.any().item()):
            raise RuntimeError(
                "target_test requires every evaluable target node in test_mask "
                "and no training nodes.")
        args.num_features, args.num_classes = data.x.shape[1], 1
        args.experiment_name = args.target_domain
        print("Testing {} labeled target nodes out of {} total nodes.".format(
            data.test_mask.sum().item(), data.x.shape[0]))

        metrics = test_target_domain(data, args)
        if args.metrics_output is not None:
            metrics_directory = os.path.dirname(args.metrics_output)
            if metrics_directory:
                os.makedirs(metrics_directory, exist_ok=True)
            with open(args.metrics_output, 'w') as metrics_file:
                json.dump(
                    {key: float(value) for key, value in metrics.items()},
                    metrics_file, indent=2)
        os.makedirs('logs', exist_ok=True)
        ResLogFile = os.path.join('logs', args.target_domain) + "_TargetTest_ResLog.txt"
        with open(ResLogFile, "a+") as file:
            now = time.localtime()
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
            strLog = formatted_time + \
                "Target Test -- ACC: {:.4f} AUC_ROC:{:.4f} F1:{:.4f} DP:{:.4f} EO:{:.4f}".format(
                    metrics['acc'] * 100, metrics['auc_roc'] * 100,
                    metrics['f1'] * 100, metrics['parity'] * 100,
                    metrics['equality'] * 100)
            file.write(strLog + "\n")

    else:
        if args.source_domain is not None or args.target_domain is not None:
            parser.error(
                "standard mode does not accept --source_domain or --target_domain.")
        data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(
            args.dataset, args.top_k)
        args.experiment_name = args.dataset
        args.num_features, args.num_classes = data.x.shape[1], 1

        args.train_ratio, args.val_ratio = torch.tensor([
            (data.y[data.train_mask] == 0).sum(),
            (data.y[data.train_mask] == 1).sum()]), torch.tensor([
                (data.y[data.val_mask] == 0).sum(),
                (data.y[data.val_mask] == 1).sum()])
        args.train_ratio, args.val_ratio = torch.max(
            args.train_ratio) / args.train_ratio, torch.max(args.val_ratio) / args.val_ratio
        args.train_ratio, args.val_ratio = args.train_ratio[
            data.y[data.train_mask].long()], args.val_ratio[data.y[data.val_mask].long()]

        acc, f1, auc_roc, parity, equality = run(data, args)
        print('======' + args.experiment_name + '-' + args.encoder + '======')
        print('auc_roc:', np.mean(auc_roc) * 100, np.std(auc_roc) * 100)
        print('Acc:', np.mean(acc) * 100, np.std(acc) * 100)
        print('f1:', np.mean(f1) * 100, np.std(f1) * 100)
        print('parity:', np.mean(parity) * 100, np.std(parity) * 100)
        print('equality:', np.mean(equality) * 100, np.std(equality) * 100)
        ResLogFile = os.path.join('logs', args.experiment_name) + "_" + str(args.with_constraint) +  "_" + str(args.loss_alpha) +  "_" + str(args.eps) + "_ResLog.txt"
        with open(ResLogFile, "a+") as file:
            now = time.localtime()
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S ", now)
            strLog = formatted_time + "final Test -- ACC: {:.2f}-+-{:.2f} AUC_ROC:{:.2f}-+-{:.2f} F1:{:.2f}-+-{:.2f} DP:{:.2f}-+-{:.2f} EO:{:.2f}-+-{:.2f}".format(
                np.mean(acc) * 100, np.std(acc) * 100,
                np.mean(auc_roc) * 100, np.std(auc_roc) * 100,
                np.mean(f1) * 100, np.std(f1) * 100,
                np.mean(parity) * 100, np.std(parity) * 100,
                np.mean(equality) * 100, np.std(equality) * 100)
            file.write(strLog + "\n")
