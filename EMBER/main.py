import json
import os
from copy import copy

import numpy as np

from config import args
from dataset import get_dataset, process_dataset
from runner import train_and_adapt
from utils import seed_everything


RUN_SEED_STEP = 1111
TARGET_SEED_OFFSET = 100000


def _values(array):
    return np.asarray(array).reshape(-1).astype(float).tolist()


def _run_seed_pair(base_source_seed, base_target_seed, run_idx):
    """Return the independently aligned source/target seeds for one run."""
    source_seed = int(base_source_seed) + run_idx * RUN_SEED_STEP
    target_seed = int(base_target_seed) + run_idx * RUN_SEED_STEP
    return source_seed, target_seed


def _run_once(run_args):
    """Load data and execute one source/target seed pair."""
    # Dataset construction shuffles source and target splits.  Seed each load
    # independently so that a run has the same source/target data split in
    # the main experiment and in every ablation subprocess.
    seed_everything(run_args.seed)
    source_data = get_dataset(run_args, run_args.inid)
    target_data = None
    if not run_args.source_only:
        seed_everything(run_args.target_seed)
        target_data = get_dataset(run_args, run_args.outid)

    print("********************process source data********************")
    source_data = process_dataset(run_args, source_data)
    if target_data is not None:
        print("********************process target data********************")
        target_data = process_dataset(run_args, target_data, is_target=True)

    return train_and_adapt(run_args, source_data, target_data)


def _combine_run_outputs(run_outputs):
    """Concatenate one-run metric arrays into the established result shape."""
    metric_outputs = tuple(
        np.concatenate([output[index] for output in run_outputs], axis=0)
        for index in range(12)
    )
    diagnostic_names = run_outputs[0][12]
    diagnostics = {
        name: np.concatenate(
            [output[12][name] for output in run_outputs], axis=0
        )
        for name in diagnostic_names
    }
    return (*metric_outputs, diagnostics)


if __name__ == '__main__':
    base_source_seed = int(args.seed)
    base_target_seed = (
        int(args.target_seed)
        if args.target_seed is not None
        else base_source_seed + TARGET_SEED_OFFSET
    )
    requested_runs = int(args.runs)
    if requested_runs < 1:
        raise ValueError("runs must be positive")

    run_outputs = []
    for run_idx in range(requested_runs):
        source_seed, target_seed = _run_seed_pair(
            base_source_seed,
            base_target_seed,
            run_idx,
        )
        run_args = copy(args)
        run_args.runs = 1
        run_args.seed = source_seed
        run_args.target_seed = target_seed
        print(
            f"[Run {run_idx}] source_seed={source_seed} "
            f"target_seed={target_seed}"
        )
        run_outputs.append(_run_once(run_args))

    (src_acc, src_auc_roc, src_parity, src_equality,
     tgt_acc, tgt_auc_roc, tgt_parity, tgt_equality,
     ada_acc, ada_auc_roc, ada_parity, ada_equality,
     src_diagnostics) = _combine_run_outputs(run_outputs)

    source_label = "Source validation" if args.source_only else "Source"
    print(f"=========== {args.inid} ({source_label}) ===========")
    print(f"Acc:      {np.mean(src_acc):.2f} +/- {np.std(src_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(src_auc_roc):.2f} +/- {np.std(src_auc_roc):.2f}")
    print(f"Parity:   {np.mean(src_parity):.2f} +/- {np.std(src_parity):.2f}")
    print(f"Equality: {np.mean(src_equality):.2f} +/- {np.std(src_equality):.2f}")

    if not args.source_only:
        print(f"\n=========== {args.outid} (Target, before adapt) ===========")
        print(f"Acc:      {np.mean(tgt_acc):.2f} +/- {np.std(tgt_acc):.2f}")
        print(f"AUC-ROC:  {np.mean(tgt_auc_roc):.2f} +/- {np.std(tgt_auc_roc):.2f}")
        print(f"Parity:   {np.mean(tgt_parity):.2f} +/- {np.std(tgt_parity):.2f}")
        print(f"Equality: {np.mean(tgt_equality):.2f} +/- {np.std(tgt_equality):.2f}")

        print(f"\n=========== {args.outid} (Target, after SFDA adapt) ===========")
        print(f"Acc:      {np.mean(ada_acc):.2f} +/- {np.std(ada_acc):.2f}")
        print(f"AUC-ROC:  {np.mean(ada_auc_roc):.2f} +/- {np.std(ada_auc_roc):.2f}")
        print(f"Parity:   {np.mean(ada_parity):.2f} +/- {np.std(ada_parity):.2f}")
        print(f"Equality: {np.mean(ada_equality):.2f} +/- {np.std(ada_equality):.2f}")

    if args.result_path:
        source_stage = 'source_val' if args.source_only else 'source'
        payload = {
            'dataset': args.dataset,
            'inid': args.inid,
            'outid': args.outid,
            'seed': base_source_seed,
            'target_seed': base_target_seed,
            'source_seeds': [
                _run_seed_pair(base_source_seed, base_target_seed, run_idx)[0]
                for run_idx in range(requested_runs)
            ],
            'target_seeds': [
                _run_seed_pair(base_source_seed, base_target_seed, run_idx)[1]
                for run_idx in range(requested_runs)
            ],
            'ablation': args.ablation,
            'stage': 'source' if args.source_only else 'full',
            'metrics': {
                source_stage: {
                    'acc': _values(src_acc),
                    'auc': _values(src_auc_roc),
                    'dp': _values(src_parity),
                    'eo': _values(src_equality),
                },
            },
            'diagnostics': {
                source_stage: {
                    name: _values(values)
                    for name, values in src_diagnostics.items()
                },
            },
        }
        if not args.source_only:
            payload['metrics'].update({
                'target_before': {
                    'acc': _values(tgt_acc),
                    'auc': _values(tgt_auc_roc),
                    'dp': _values(tgt_parity),
                    'eo': _values(tgt_equality),
                },
                'target_after': {
                    'acc': _values(ada_acc),
                    'auc': _values(ada_auc_roc),
                    'dp': _values(ada_parity),
                    'eo': _values(ada_equality),
                },
            })

        result_dir = os.path.dirname(os.path.abspath(args.result_path))
        os.makedirs(result_dir, exist_ok=True)
        with open(args.result_path, 'w', encoding='utf-8') as result_file:
            json.dump(payload, result_file, ensure_ascii=False, indent=2)
