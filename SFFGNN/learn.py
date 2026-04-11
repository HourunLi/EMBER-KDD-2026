import torch
import numpy as np
import torch.nn.functional as F
from utils import fair_metric
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


def evaluate_per_class(args, data, encoder):
    """
    Evaluate encoder on all splits (all/train/val/test).

    Returns:
      accs, auc_rocs, paritys, equalitys  — each a dict keyed by split name
    """
    sens_labels = data.sens_labels
    test_labels = data.y[data.test_mask]

    t_idx_s0 = sens_labels[data.test_mask] == 0
    t_idx_s1 = sens_labels[data.test_mask] == 1
    t_idx_s0_y1 = torch.logical_and(t_idx_s0, test_labels == 1)
    t_idx_s1_y1 = torch.logical_and(t_idx_s1, test_labels == 1)
    t_idx_s0_y0 = torch.logical_and(t_idx_s0, test_labels == 0)
    t_idx_s1_y0 = torch.logical_and(t_idx_s1, test_labels == 0)

    accs, auc_rocs, f1s, paritys, equalitys = {}, {}, {}, {}, {}

    encoder.eval()
    with torch.no_grad():
        feat, output = encoder(data.x, data.edge_index)
        probs   = torch.sigmoid(output.squeeze()).cpu().numpy()
        y_all   = data.y.cpu().numpy()
        sens_all = data.sens_labels.cpu().numpy()
        all_mask = data.train_mask | data.val_mask | data.test_mask

        splits = {
            'all':   all_mask.cpu().numpy(),
            'train': data.train_mask.cpu().numpy(),
            'val':   data.val_mask.cpu().numpy(),
            'test':  data.test_mask.cpu().numpy(),
        }

        # Save embeddings for analysis
        labels = torch.full((test_labels.shape[0],), -1, dtype=torch.int64)
        labels[t_idx_s0_y1] = 0
        labels[t_idx_s1_y1] = 1
        labels[t_idx_s0_y0] = 2
        labels[t_idx_s1_y0] = 3
        np.savez(f"{args.dataset}_feat.npz",
                 representations=feat[data.test_mask].cpu().numpy())
        np.savez(f"{args.dataset}_labels.npz",
                 labels=labels.cpu().numpy())

        result = {}
        for split_name, mask in splits.items():
            y_true = y_all[mask]
            sens   = sens_all[mask]
            prob   = probs[mask]
            pred   = (prob > 0.5).astype(int)

            acc_total = accuracy_score(y_true, pred) * 100
            auc_total = roc_auc_score(y_true, prob) * 100 if len(set(y_true)) == 2 else float('nan')
            f1_total  = f1_score(y_true, pred, zero_division=0) * 100

            # Per sensitive group
            sens_metrics = {}
            for s in np.unique(sens):
                idx = sens == s
                sens_metrics[int(s)] = {
                    'acc': accuracy_score(y_true[idx], pred[idx]) * 100,
                    'auc': roc_auc_score(y_true[idx], prob[idx]) * 100
                          if len(np.unique(y_true[idx])) == 2 else float('nan'),
                    'f1':  f1_score(y_true[idx], pred[idx], zero_division=0) * 100,
                }

            # Per target class
            y_metrics = {}
            for yval in np.unique(y_true):
                idx = y_true == yval
                y_metrics[int(yval)] = {
                    'acc': accuracy_score(y_true[idx], pred[idx]) * 100,
                    'auc': roc_auc_score((y_true == yval).astype(int),
                                         prob if yval == 1 else 1 - prob) * 100
                          if len(set(y_true)) == 2 else float('nan'),
                    'f1':  f1_score((y_true == yval).astype(int),
                                    (pred == yval).astype(int), zero_division=0) * 100,
                }

            dp, eo = fair_metric(pred, y_true, sens)
            result[split_name] = {
                'overall':      {'acc': acc_total, 'auc': auc_total, 'f1': f1_total},
                'sens_group':   sens_metrics,
                'target_group': y_metrics,
                'fairness':     {'dp': dp * 100, 'eo': eo * 100},
            }

    for split_name in splits:
        target_vals = result[split_name]['target_group'].values()
        accs[split_name]     = np.nanmean([v['acc'] for v in target_vals])
        auc_rocs[split_name] = np.nanmean([v['auc'] for v in target_vals])
        f1s[split_name]      = np.nanmean([v['f1']  for v in target_vals])
        paritys[split_name]  = result[split_name]['fairness']['dp']
        equalitys[split_name]= result[split_name]['fairness']['eo']

    return accs, auc_rocs, paritys, equalitys
