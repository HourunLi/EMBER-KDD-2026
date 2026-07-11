import csv
import copy
import itertools
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


SEARCH_TRIALS = {
    'tau_c': {0.1, 0.2, 0.35, 0.5, 0.7},
    'proto_temp': {0.5, 1.0, 1.6, 2.5, 3.0},
    'alpha_pi': {0.7, 0.9, 0.99},
}

AVAILABLE_GPUS = ['0', '1', '2', '3', '4', '5', '6', '7']
GPU_IDLE_MAX_MEMORY_MB = 5000
GPU_IDLE_MAX_UTILIZATION = 10
GPU_POLL_INTERVAL_SECONDS = 60
GPU_IDLE_CONFIRMATIONS = 0


def compute_joint_metric(result):
    """
    Joint score on the adapted target domain.

    Higher Acc/AUC are rewarded, while higher DP/EO are penalized.
    All four metrics are already on a 0-100 scale, so a weighted linear
    combination keeps the ranking simple and easy to inspect.
    """
    return float( result['ada_acc'] + result['ada_auc'] - result['ada_dp'] - result['ada_eo'])


def run_once(run_args):
    from dataset import process_dataset, get_dataset
    from runner import train_and_adapt
    from utils import seed_everything

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


def build_trial_args(base_args, overrides, gpu_id):
    trial_args = copy.deepcopy(base_args)
    for key, value in overrides.items():
        setattr(trial_args, key, value)

    trial_args.device_id = str(gpu_id)
    if str(gpu_id) != 'cpu' and int(trial_args.device_id) >= 0 and torch.cuda.is_available():
        trial_args.device = torch.device('cuda:{}'.format(trial_args.device_id))
    else:
        trial_args.device = torch.device('cpu')
    return trial_args


def format_trial_summary(record):
    result = record['result']
    return (
        f"Trial {record['trial_idx'] + 1} | GPU {record['gpu_id']} | "
        f"joint_metric={record['joint_metric']:.4f} | overrides={record['overrides']}\n"
        f"  source:      Acc={result['src_acc']:.2f}, AUC={result['src_auc']:.2f}, "
        f"DP={result['src_dp']:.2f}, EO={result['src_eo']:.2f}\n"
        f"  after_adapt: Acc={result['ada_acc']:.2f}, AUC={result['ada_auc']:.2f}, "
        f"DP={result['ada_dp']:.2f}, EO={result['ada_eo']:.2f}"
    )


def save_result_summary(base_args, records):
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{base_args.dataset}.csv")

    fieldnames = [
        'rank', 'trial_idx', 'gpu_id', 'joint_metric', 'overrides',
        'src_acc', 'src_auc', 'src_dp', 'src_eo',
        'ada_acc', 'ada_auc', 'ada_dp', 'ada_eo',
    ]
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            result = record['result']
            writer.writerow({
                'rank': rank,
                'trial_idx': record['trial_idx'] + 1,
                'gpu_id': record['gpu_id'],
                'joint_metric': record['joint_metric'],
                'overrides': json.dumps(record['overrides'], ensure_ascii=False, sort_keys=True),
                'src_acc': result['src_acc'],
                'src_auc': result['src_auc'],
                'src_dp': result['src_dp'],
                'src_eo': result['src_eo'],
                'ada_acc': result['ada_acc'],
                'ada_auc': result['ada_auc'],
                'ada_dp': result['ada_dp'],
                'ada_eo': result['ada_eo'],
            })
    return result_path


def parse_gpu_value(value):
    try:
        return int(value.strip())
    except ValueError:
        return None


def query_gpu_statuses():
    output = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
        ],
        stderr=subprocess.STDOUT,
        text=True,
    )
    statuses = {}
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3:
            continue
        statuses[parts[0]] = {
            'memory_used': parse_gpu_value(parts[1]),
            'utilization': parse_gpu_value(parts[2]),
        }
    return statuses


def get_gpu_status(gpu_id):
    statuses = query_gpu_statuses()
    status = statuses.get(str(gpu_id))
    if status is None:
        raise RuntimeError(f"GPU {gpu_id} was not reported by nvidia-smi.")
    return status


def sleep_with_stop(stop_event):
    for _ in range(GPU_POLL_INTERVAL_SECONDS):
        if stop_event is not None and stop_event.is_set():
            return False
        time.sleep(1)
    return True


def wait_for_gpu(gpu_id, stop_event=None):
    if str(gpu_id) == 'cpu':
        return True

    confirmations = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            return False

        try:
            status = get_gpu_status(gpu_id)
        except Exception as exc:
            confirmations = 0
            print(f"Waiting for GPU {gpu_id}: unable to query nvidia-smi ({exc})")
            if not sleep_with_stop(stop_event):
                return False
            continue

        memory_used = status['memory_used']
        utilization = status['utilization']
        idle = (
            memory_used is not None
            and utilization is not None
            and memory_used <= GPU_IDLE_MAX_MEMORY_MB
            and utilization <= GPU_IDLE_MAX_UTILIZATION
        )

        if idle:
            confirmations += 1
            print(
                f"GPU {gpu_id} idle check {confirmations}/{GPU_IDLE_CONFIRMATIONS}: "
                f"memory={memory_used} MiB, util={utilization}%"
            )
        else:
            confirmations = 0
            print(
                f"Waiting for GPU {gpu_id}: memory={memory_used} MiB, "
                f"util={utilization}%"
            )

        if confirmations >= GPU_IDLE_CONFIRMATIONS:
            print(f"GPU {gpu_id} is idle; starting next trial.")
            return True
        if not sleep_with_stop(stop_event):
            return False


def release_cuda_cache(gpu_id):
    if str(gpu_id) == 'cpu' or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


def set_worker_log_path(base_args, gpu_id):
    log_path = getattr(base_args, 'log_path', None)
    if not log_path:
        return
    root, ext = os.path.splitext(log_path)
    worker_log_path = f"{root}.gpu{gpu_id}{ext or '.log'}"
    sys.argv = sys.argv + ['--log_path', worker_log_path]


def run_worker(gpu_id, base_args, task_queue, result_queue, stop_event):
    set_worker_log_path(base_args, gpu_id)
    while True:
        if not wait_for_gpu(gpu_id, stop_event):
            break

        item = task_queue.get()
        if item is None:
            break

        trial_idx, overrides = item
        try:
            trial_args = build_trial_args(base_args, overrides, gpu_id)

            print("\n" + "=" * 72)
            print(f"Trial {trial_idx + 1}: {overrides} on GPU {gpu_id}")
            print("=" * 72)

            result = run_once(trial_args)
            joint_metric = compute_joint_metric(result)
            print(f"Joint score (Target after adapt): {joint_metric:.4f}")
            result_queue.put({
                'trial_idx': trial_idx,
                'gpu_id': gpu_id,
                'overrides': overrides,
                'result': result,
                'joint_metric': joint_metric,
            })
        except Exception:
            result_queue.put({
                'trial_idx': trial_idx,
                'gpu_id': gpu_id,
                'overrides': overrides,
                'error': traceback.format_exc(),
            })
        finally:
            release_cuda_cache(gpu_id)


if __name__ == '__main__':
    from config import args

    if not SEARCH_TRIALS:
        run_once(args)
    else:
        base_args = copy.deepcopy(args)
        keys = list(SEARCH_TRIALS.keys())
        values = [SEARCH_TRIALS[k] for k in keys]
        trials = list(itertools.product(*values))

        if torch.cuda.is_available():
            worker_gpus = [str(gpu_id) for gpu_id in AVAILABLE_GPUS]
            if not worker_gpus:
                worker_gpus = [str(base_args.device_id)]
        else:
            worker_gpus = ['cpu']
        worker_gpus = worker_gpus[:min(len(worker_gpus), len(trials))]
        gpu_worker_ids = [gpu_id for gpu_id in worker_gpus if gpu_id != 'cpu']
        if gpu_worker_ids:
            statuses = query_gpu_statuses()
            missing_gpus = [gpu_id for gpu_id in gpu_worker_ids if gpu_id not in statuses]
            if missing_gpus:
                raise RuntimeError(f"GPU(s) not reported by nvidia-smi: {missing_gpus}")

        ctx = mp.get_context('spawn')
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        stop_event = ctx.Event()

        for trial_idx, trial_values in enumerate(trials):
            overrides = dict(zip(keys, trial_values))
            task_queue.put((trial_idx, overrides))
        for _ in worker_gpus:
            task_queue.put(None)

        workers = [
            ctx.Process(target=run_worker, args=(gpu_id, base_args, task_queue, result_queue, stop_event))
            for gpu_id in worker_gpus
        ]
        for worker in workers:
            worker.start()

        try:
            records = [result_queue.get() for _ in trials]
        finally:
            stop_event.set()
            for worker in workers:
                worker.join()

        errors = [record for record in records if 'error' in record]
        records = [record for record in records if 'error' not in record]

        print("\n" + "=" * 72)
        print("Trials sorted by joint_metric (source + after_adapt):")
        sorted_records = sorted(records, key=lambda item: item['joint_metric'], reverse=True)
        result_path = save_result_summary(base_args, sorted_records)
        for record in sorted_records:
            print(format_trial_summary(record))
        print(f"Saved summary to: {result_path}")
        if errors:
            print("\nFailed trials:")
            for record in errors:
                print(f"Trial {record['trial_idx'] + 1} | GPU {record['gpu_id']} | overrides={record['overrides']}")
                print(record['error'])
            raise RuntimeError(f"{len(errors)} trial(s) failed.")
        print("=" * 72)
