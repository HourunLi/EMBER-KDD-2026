import copy
import itertools

import numpy as np
import torch

from dataset import process_dataset, get_dataset
from runner import train_and_adapt
from config import args
from utils import seed_everything


# 为空时，行为与 main.py 一致，只跑当前 args。
# 非空时，按笛卡尔积依次覆盖参数做微调搜索。
SEARCH_TRIALS = {
}


def run_once(run_args):
    # 这里完全复用 main.py 的主流程，避免 tune.py 和正式实验逻辑漂移。
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

            # 若调 device_id，则同步刷新 device，避免仍沿用旧设备。
            if int(trial_args.device_id) >= 0 and torch.cuda.is_available():
                trial_args.device = torch.device('cuda:{}'.format(trial_args.device_id))
            else:
                trial_args.device = torch.device('cpu')

            print("\n" + "=" * 72)
            print(f"Trial {trial_idx + 1}/{len(trials)}: {overrides}")
            print("=" * 72)

            result = run_once(trial_args)

            # 默认按目标域适配后 AUC 选最优 trial。
            if result['ada_auc'] > best_metric:
                best_metric = result['ada_auc']
                best_overrides = overrides
                best_result = result

        print("\n" + "=" * 72)
        print("Best trial (by Target after adapt AUC-ROC):")
        print(best_overrides)
        print(best_result)
        print("=" * 72)
