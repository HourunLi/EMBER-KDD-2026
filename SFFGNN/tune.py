import copy
import itertools

import numpy as np
import torch

from dataset import process_dataset, get_dataset
from runner import train_and_adapt
from config import args
from utils import seed_everything


SEARCH_TRIALS = {
    "hidden_dim": [16, 24, 32, 64],
    "n_layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.2],
    "lr": [0.0005],
    "lr2_reg": [0.0],
    "lambda_fair": [1.0, 2.0],
}


def compute_joint_metric(result):
    """
    Joint score on the adapted target domain.

    Higher Acc/AUC are rewarded, while higher DP/EO are penalized.
    All four metrics are already on a 0-100 scale, so a weighted linear
    combination keeps the ranking simple and easy to inspect.
    """
    return float( result['ada_acc'] + result['ada_auc'] - result['ada_dp'] - result['ada_eo'])


def run_once(run_args):
    # Reuse the same main training/evaluation pipeline as main.py.
    seed_everything(run_args.seed)

    source_data = get_dataset(run_args, run_args.inid)
    target_data = get_dataset(run_args, run_args.outid)

    print("********************process source data********************")
    source_data = process_dataset(run_args, source_data)
    print("********************process target data********************")
    target_data = process_dataset(run_args, target_data)

    (src_acc, src_auc_roc, src_parity, src_equality,
     tgt_acc, tgt_auc_roc, tgt_parity, tgt_equality,
     ada_acc, ada_auc_roc, ada_parity, ada_equality) = train_and_adapt(
        run_args, source_data, target_data
    )

    print(f"=========== {run_args.inid} (Source) ===========")
    print(f"Acc:      {np.mean(src_acc):.2f} ± {np.std(src_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(src_auc_roc):.2f} ± {np.std(src_auc_roc):.2f}")
    print(f"Parity:   {np.mean(src_parity):.2f} ± {np.std(src_parity):.2f}")
    print(f"Equality: {np.mean(src_equality):.2f} ± {np.std(src_equality):.2f}")

    print(f"\n=========== {run_args.outid} (Target, before adapt) ===========")
    print(f"Acc:      {np.mean(tgt_acc):.2f} ± {np.std(tgt_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(tgt_auc_roc):.2f} ± {np.std(tgt_auc_roc):.2f}")
    print(f"Parity:   {np.mean(tgt_parity):.2f} ± {np.std(tgt_parity):.2f}")
    print(f"Equality: {np.mean(tgt_equality):.2f} ± {np.std(tgt_equality):.2f}")

    print(f"\n=========== {run_args.outid} (Target, after SFDA adapt) ===========")
    print(f"Acc:      {np.mean(ada_acc):.2f} ± {np.std(ada_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(ada_auc_roc):.2f} ± {np.std(ada_auc_roc):.2f}")
    print(f"Parity:   {np.mean(ada_parity):.2f} ± {np.std(ada_parity):.2f}")
    print(f"Equality: {np.mean(ada_equality):.2f} ± {np.std(ada_equality):.2f}")

    return {
        'src_acc': float(np.mean(src_acc)),
        'src_auc': float(np.mean(src_auc_roc)),
        'src_dp': float(np.mean(src_parity)),
        'src_eo': float(np.mean(src_equality)),
        'tgt_acc': float(np.mean(tgt_acc)),
        'tgt_auc': float(np.mean(tgt_auc_roc)),
        'tgt_dp': float(np.mean(tgt_parity)),
        'tgt_eo': float(np.mean(tgt_equality)),
        'ada_acc': float(np.mean(ada_acc)),
        'ada_auc': float(np.mean(ada_auc_roc)),
        'ada_dp': float(np.mean(ada_parity)),
        'ada_eo': float(np.mean(ada_equality)),
    }


if __name__ == '__main__':
    if not SEARCH_TRIALS:
        run_once(args)
    else:
        base_args = copy.deepcopy(args)
        keys = list(SEARCH_TRIALS.keys())
        values = [SEARCH_TRIALS[k] for k in keys]
        trials = list(itertools.product(*values))

        best_metric = -float('inf')
        best_overrides = None
        best_result = None

        for trial_idx, trial_values in enumerate(trials):
            overrides = dict(zip(keys, trial_values))
            trial_args = copy.deepcopy(base_args)
            for key, value in overrides.items():
                setattr(trial_args, key, value)

            if int(trial_args.device_id) >= 0 and torch.cuda.is_available():
                trial_args.device = torch.device('cuda:{}'.format(trial_args.device_id))
            else:
                trial_args.device = torch.device('cpu')

            print("\n" + "=" * 72)
            print(f"Trial {trial_idx + 1}/{len(trials)}: {overrides}")
            print("=" * 72)

            result = run_once(trial_args)
            joint_metric = compute_joint_metric(result)
            print(f"Joint score (Target after adapt): {joint_metric:.4f}")

            if joint_metric > best_metric:
                best_metric = joint_metric
                best_overrides = overrides
                best_result = result

        print("\n" + "=" * 72)
        print("Best trial (by Target after adapt joint metric):")
        print(f"Best joint score: {best_metric:.4f}")
        print(best_overrides)
        print(best_result)
        print("=" * 72)
