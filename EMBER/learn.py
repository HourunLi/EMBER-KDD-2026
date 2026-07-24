import torch
import numpy as np
from utils import fair_metric
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def evaluate_per_class(args, data, encoder):
    """Evaluate source predictions with the paper's ACC, AUC, DP, and EO."""
    accs, auc_rocs, paritys, equalitys = {}, {}, {}, {}
    diagnostics = {
        'balanced_acc': {},
        'macro_f1': {},
        'positive_rate': {},
        'predicted_class_count': {},
    }

    encoder.eval()
    with torch.no_grad():
        feat, output = encoder(data.x, data.edge_index)
        probs = torch.sigmoid(output.view(-1)).cpu().numpy()
        y_all = data.y.cpu().numpy()
        sens_all = data.sens_labels.cpu().numpy()
        valid_label_mask = (data.y == 0) | (data.y == 1)
        all_mask = (
            data.train_mask | data.val_mask | data.test_mask
        ) & valid_label_mask

        splits = {
            'all':   all_mask.cpu().numpy(),
            'train': (data.train_mask & valid_label_mask).cpu().numpy(),
            'val':   (data.val_mask & valid_label_mask).cpu().numpy(),
            'test':  (data.test_mask & valid_label_mask).cpu().numpy(),
        }

        if not getattr(args, 'disable_embedding_export', False):
            export_mask = data.test_mask & valid_label_mask
            export_y = data.y[export_mask]
            export_sens = data.sens_labels[export_mask]
            labels = torch.full_like(export_y, -1, dtype=torch.int64)
            labels[(export_y == 1) & (export_sens == 0)] = 0
            labels[(export_y == 1) & (export_sens == 1)] = 1
            labels[(export_y == 0) & (export_sens == 0)] = 2
            labels[(export_y == 0) & (export_sens == 1)] = 3
            np.savez(f"{args.dataset}_feat.npz",
                     representations=feat[export_mask].cpu().numpy())
            np.savez(f"{args.dataset}_labels.npz",
                     labels=labels.cpu().numpy())

        for split_name, mask in splits.items():
            y_true = y_all[mask]
            if y_true.size == 0:
                accs[split_name] = 0.0
                auc_rocs[split_name] = 50.0
                paritys[split_name] = float("nan")
                equalitys[split_name] = float("nan")
                diagnostics['balanced_acc'][split_name] = float("nan")
                diagnostics['macro_f1'][split_name] = float("nan")
                diagnostics['positive_rate'][split_name] = float("nan")
                diagnostics['predicted_class_count'][split_name] = 0
                continue

            sens = sens_all[mask]
            prob = probs[mask]
            pred = (prob > 0.5).astype(int)
            accs[split_name] = accuracy_score(y_true, pred) * 100
            auc_rocs[split_name] = (
                roc_auc_score(y_true, prob) * 100
                if len(np.unique(y_true)) == 2
                else 50.0
            )
            valid_sensitive = (sens == 0) | (sens == 1)
            dp, eo = fair_metric(
                pred[valid_sensitive],
                y_true[valid_sensitive],
                sens[valid_sensitive],
            )
            paritys[split_name] = dp * 100
            equalitys[split_name] = eo * 100
            diagnostics['balanced_acc'][split_name] = (
                balanced_accuracy_score(y_true, pred) * 100
            )
            diagnostics['macro_f1'][split_name] = f1_score(
                y_true,
                pred,
                labels=[0, 1],
                average='macro',
                zero_division=0,
            ) * 100
            diagnostics['positive_rate'][split_name] = float(pred.mean() * 100)
            diagnostics['predicted_class_count'][split_name] = int(
                np.unique(pred).size
            )

    return accs, auc_rocs, paritys, equalitys, diagnostics
