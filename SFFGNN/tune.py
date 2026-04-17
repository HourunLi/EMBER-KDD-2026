"""
Validation-split metrics: same training + SFDA as runner.train_and_adapt, but reported
scalars use split ``metric_split`` (default ``all``).

CLI matches main.py (config.py). Default entry matches main.py except:
  - calls ``train_and_adapt_metric_split`` instead of ``train_and_adapt``;
  - section titles include ``[val]`` (or chosen METRIC_SPLIT) so you know which split the
    printed means/std refer to.

Optional: set SEARCH_TRIALS to a non-empty list to run multiple hyperparameter overrides
on top of CLI + config.yaml (see file bottom).
"""

import copy
import itertools
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call
from tqdm import tqdm

from dataset import process_dataset, get_dataset
from models import *
from utils import *
from runner import (
    mmd_loss,
    _init_fair_gnn,
    extract_source_knowledge,
    _save_checkpoint,
    _load_checkpoint,
    adapt_target,
    evaluate_per_class,
    evaluate_after,
)
from config import args

# Allowed keys from learn.evaluate_per_class / runner.evaluate_after.
_VALID_SPLITS = frozenset({"all", "train", "val", "test"})

# Grid search space.
# Recommended format: dict[param_name] = [candidate values]
# Example below means 2*3 = 6 trials.
# Keep it empty ({}) to run once like main.py (same seed, same args).
SEARCH_TRIALS = {
}


def train_and_adapt_metric_split(args, source_data, target_data, metric_split="all"):
    """
    Same logic as runner.train_and_adapt; only differences:
      - indexing uses ``metric_split`` instead of hard-coded ``'all'`` in prints and arrays;
      - optional guard when ``metric_split`` is invalid.
    """
    if metric_split not in _VALID_SPLITS:
        raise ValueError(f"metric_split must be one of {_VALID_SPLITS}, got {metric_split!r}")

    src_acc = np.zeros([args.runs, 1])
    src_auc_roc = np.zeros([args.runs, 1])
    src_parity = np.zeros([args.runs, 1])
    src_equality = np.zeros([args.runs, 1])
    tgt_acc = np.zeros([args.runs, 1])
    tgt_auc_roc = np.zeros([args.runs, 1])
    tgt_parity = np.zeros([args.runs, 1])
    tgt_equality = np.zeros([args.runs, 1])
    ada_acc = np.zeros([args.runs, 1])
    ada_auc_roc = np.zeros([args.runs, 1])
    ada_parity = np.zeros([args.runs, 1])
    ada_equality = np.zeros([args.runs, 1])

    m = metric_split

    source_data = source_data.to(args.device)
    target_data = target_data.to(args.device)

    cls_labels = source_data.y.float().to(args.device)
    sens_labels = source_data.sens_labels.to(args.device)
    train_mask = source_data.train_mask

    criterion = nn.BCEWithLogitsLoss()

    for run_idx in tqdm(range(args.runs), unit="run"):
        model, knowledge = (None, None)
        if getattr(args, "use_checkpoint", True):
            model, knowledge = _load_checkpoint(args, run_idx)
        skip_training = model is not None and knowledge is not None

        if skip_training:
            print(f"[Run {run_idx}] Loaded checkpoint, skipping training.")
        else:
            if getattr(args, "use_checkpoint", True):
                print(f"[Run {run_idx}] No checkpoint found, training from scratch.")
            else:
                print(f"[Run {run_idx}] Checkpoint loading disabled, training from scratch.")
            model, optimizer, scheduler = _init_fair_gnn(args)
            meta_lr_src = getattr(args, "meta_lr_src", 0.01)

            for epoch in tqdm(range(args.train_epochs), desc=f"Run {run_idx} [train]", leave=False):
                model.train()

                _, cls_logit = model(source_data.x, source_data.edge_index)
                L_cls = criterion(cls_logit[train_mask].view(-1), cls_labels[train_mask])

                backbone_params = list(model.backbone.parameters())
                grad_cls = torch.autograd.grad(
                    L_cls, backbone_params, create_graph=True, retain_graph=True
                )
                params_prime = [p - meta_lr_src * g for p, g in zip(backbone_params, grad_cls)]

                param_dict = dict(zip([n for n, _ in model.backbone.named_parameters()], params_prime))
                emb_prime = functional_call(
                    model.backbone, param_dict, (source_data.x, source_data.edge_index)
                )
                emb_prime_train = emb_prime[train_mask]

                L_mmd = mmd_loss(
                    emb_prime_train,
                    sens_labels[train_mask].long(),
                    chunk_size=getattr(args, "mmd_chunk_size", 1024),
                )

                loss = L_cls + args.lambda_fair * L_mmd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (epoch + 1) % 50 == 0:
                    print(
                        f"[Run {run_idx} | Epoch {epoch+1:03d}] "
                        f"L_cls={L_cls.item():.4f}  "
                        f"L_mmd={L_mmd.item():.4f}  "
                        f"Total={loss.item():.4f}"
                    )

            knowledge = extract_source_knowledge(args, source_data, model)
            _save_checkpoint(args, run_idx, model, knowledge)

        accs, auc_rocs, tmp_parity, tmp_equality = evaluate_per_class(args, source_data, model)
        print(
            f"[Run {run_idx}] {args.dataset}  Source ({args.inid}) [{m}] | "
            f"Acc={accs[m]:.2f}  AUC={auc_rocs[m]:.2f}  "
            f"DP={tmp_parity[m]:.2f}  EO={tmp_equality[m]:.2f}"
        )
        src_acc[run_idx] = accs[m]
        src_auc_roc[run_idx] = auc_rocs[m]
        src_parity[run_idx] = tmp_parity[m]
        src_equality[run_idx] = tmp_equality[m]

        t_accs, t_auc_rocs, t_parity, t_equality = evaluate_per_class(args, target_data, model)
        print(
            f"[Run {run_idx}] {args.dataset}  Target before adapt ({args.outid}) [{m}] | "
            f"Acc={t_accs[m]:.2f}  AUC={t_auc_rocs[m]:.2f}  "
            f"DP={t_parity[m]:.2f}  EO={t_equality[m]:.2f}"
        )
        tgt_acc[run_idx] = t_accs[m]
        tgt_auc_roc[run_idx] = t_auc_rocs[m]
        tgt_parity[run_idx] = t_parity[m]
        tgt_equality[run_idx] = t_equality[m]

        adapted_model, state = adapt_target(args, target_data, knowledge)

        a_accs, a_aucs, a_par, a_eq = evaluate_after(args, target_data, adapted_model, state)
        print(
            f"[Run {run_idx}] {args.dataset}  Target after adapt ({args.outid}) [{m}] | "
            f"Acc={a_accs[m]:.2f}  AUC={a_aucs[m]:.2f}  "
            f"DP={a_par[m]:.2f}  EO={a_eq[m]:.2f}"
        )
        ada_acc[run_idx] = a_accs[m]
        ada_auc_roc[run_idx] = a_aucs[m]
        ada_parity[run_idx] = a_par[m]
        ada_equality[run_idx] = a_eq[m]

    return (
        src_acc,
        src_auc_roc,
        src_parity,
        src_equality,
        tgt_acc,
        tgt_auc_roc,
        tgt_parity,
        tgt_equality,
        ada_acc,
        ada_auc_roc,
        ada_parity,
        ada_equality,
    )


def _apply_overrides(ns, overrides):
    """Apply SEARCH_TRIALS dict on a copy of args (grid mode only)."""
    for k, v in overrides.items():
        setattr(ns, k, v)


def _refresh_device(ns):
    """After overriding device_id in grid mode, rebuild torch.device like config.py."""
    if int(ns.device_id) >= 0 and torch.cuda.is_available():
        ns.device = torch.device("cuda:{}".format(ns.device_id))
    else:
        ns.device = torch.device("cpu")


def _expand_search_trials(search_trials):
    """
    Normalize SEARCH_TRIALS into a list[dict] of overrides.

    Supported input formats:
      1) dict[str, list]  -> auto Cartesian product grid
      2) list[dict]       -> explicit manual trials (backward compatible)
    """
    if not search_trials:
        return []

    if isinstance(search_trials, list):
        return search_trials

    if isinstance(search_trials, dict):
        keys = list(search_trials.keys())
        values_lists = [search_trials[k] for k in keys]
        if any(not isinstance(vs, (list, tuple)) or len(vs) == 0 for vs in values_lists):
            raise ValueError("For dict SEARCH_TRIALS, every key must map to a non-empty list/tuple.")
        combos = itertools.product(*values_lists)
        return [dict(zip(keys, combo)) for combo in combos]

    raise TypeError("SEARCH_TRIALS must be either dict[param->list] or list[dict].")


def _metrics_mean_std(acc, auc, parity, equality):
    """Pack four metric arrays (shape [runs, 1]) into mean/std dicts for logging."""
    return {
        "acc": (float(np.mean(acc)), float(np.std(acc))),
        "auc": (float(np.mean(auc)), float(np.std(auc))),
        "dp": (float(np.mean(parity)), float(np.std(parity))),
        "eo": (float(np.mean(equality)), float(np.std(equality))),
    }


def _format_metrics_line(label, m):
    """m from _metrics_mean_std; one line Acc/AUC/DP/EO as mean±std."""
    def cell(k):
        mu, sd = m[k]
        return f"{mu:.2f}±{sd:.2f}"

    return (
        f"{label}  Acc={cell('acc')}  AUC={cell('auc')}  "
        f"DP={cell('dp')}  EO={cell('eo')}"
    )


def _print_trials_final_summary(results, metric_split):
    """
    After all SEARCH_TRIALS, print one block per trial: Source / Tgt pre / Tgt post,
    each with the four metrics (Acc, AUC-ROC, DP, EO) as mean±std over runs.
    """
    if not results:
        return ""
    lines = []
    lines.append("=" * 72)
    lines.append(
        f"[{metric_split}] Summary: Each hyperparameter combination "
        "Source / Target Before / Target After Four metrics (mean±std across runs)"
    )
    lines.append("=" * 72)
    for r in results:
        lines.append("")
        lines.append(f"--- trial {r['trial']}  overrides={r['overrides']} ---")
        lines.append(_format_metrics_line("  Source:          ", r["src"]))
        lines.append(_format_metrics_line("  Target Before: ", r["tgt"]))
        lines.append(_format_metrics_line("  Target After: ", r["ada"]))
    lines.append("")
    lines.append("=" * 72)
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    return summary_text


def _print_summary(run_args, ms, src_acc, src_auc_roc, src_parity, src_equality,
                   tgt_acc, tgt_auc_roc, tgt_parity, tgt_equality,
                   ada_acc, ada_auc_roc, ada_parity, ada_equality, trial_note=""):
    """Same three blocks as main.py; titles note split ``ms`` and optional ``trial_note``."""
    ds = run_args.dataset
    print(f"=========== {ds} {run_args.inid} (Source) [{ms}]{trial_note} ===========")
    print(f"Acc:      {np.mean(src_acc):.2f} ± {np.std(src_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(src_auc_roc):.2f} ± {np.std(src_auc_roc):.2f}")
    print(f"Parity:   {np.mean(src_parity):.2f} ± {np.std(src_parity):.2f}")
    print(f"Equality: {np.mean(src_equality):.2f} ± {np.std(src_equality):.2f}")

    print(f"\n=========== {ds} {run_args.outid} (Target, before adapt) [{ms}]{trial_note} ===========")
    print(f"Acc:      {np.mean(tgt_acc):.2f} ± {np.std(tgt_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(tgt_auc_roc):.2f} ± {np.std(tgt_auc_roc):.2f}")
    print(f"Parity:   {np.mean(tgt_parity):.2f} ± {np.std(tgt_parity):.2f}")
    print(f"Equality: {np.mean(tgt_equality):.2f} ± {np.std(tgt_equality):.2f}")

    print(f"\n=========== {ds} {run_args.outid} (Target, after SFDA adapt) [{ms}]{trial_note} ===========")
    print(f"Acc:      {np.mean(ada_acc):.2f} ± {np.std(ada_acc):.2f}")
    print(f"AUC-ROC:  {np.mean(ada_auc_roc):.2f} ± {np.std(ada_auc_roc):.2f}")
    print(f"Parity:   {np.mean(ada_parity):.2f} ± {np.std(ada_parity):.2f}")
    print(f"Equality: {np.mean(ada_equality):.2f} ± {np.std(ada_equality):.2f}")


def _build_single_run_summary_text(run_args, ms, src_acc, src_auc_roc, src_parity, src_equality,
                                   tgt_acc, tgt_auc_roc, tgt_parity, tgt_equality,
                                   ada_acc, ada_auc_roc, ada_parity, ada_equality, trial_note=""):
    """Build the same three-block final summary text for optional log appending."""
    ds = run_args.dataset
    lines = [
        f"=========== {ds} {run_args.inid} (Source) [{ms}]{trial_note} ===========",
        f"Acc:      {np.mean(src_acc):.2f} ± {np.std(src_acc):.2f}",
        f"AUC-ROC:  {np.mean(src_auc_roc):.2f} ± {np.std(src_auc_roc):.2f}",
        f"Parity:   {np.mean(src_parity):.2f} ± {np.std(src_parity):.2f}",
        f"Equality: {np.mean(src_equality):.2f} ± {np.std(src_equality):.2f}",
        "",
        f"=========== {ds} {run_args.outid} (Target, before adapt) [{ms}]{trial_note} ===========",
        f"Acc:      {np.mean(tgt_acc):.2f} ± {np.std(tgt_acc):.2f}",
        f"AUC-ROC:  {np.mean(tgt_auc_roc):.2f} ± {np.std(tgt_auc_roc):.2f}",
        f"Parity:   {np.mean(tgt_parity):.2f} ± {np.std(tgt_parity):.2f}",
        f"Equality: {np.mean(tgt_equality):.2f} ± {np.std(tgt_equality):.2f}",
        "",
        f"=========== {ds} {run_args.outid} (Target, after SFDA adapt) [{ms}]{trial_note} ===========",
        f"Acc:      {np.mean(ada_acc):.2f} ± {np.std(ada_acc):.2f}",
        f"AUC-ROC:  {np.mean(ada_auc_roc):.2f} ± {np.std(ada_auc_roc):.2f}",
        f"Parity:   {np.mean(ada_parity):.2f} ± {np.std(ada_parity):.2f}",
        f"Equality: {np.mean(ada_equality):.2f} ± {np.std(ada_equality):.2f}",
    ]
    return "\n".join(lines)


def _read_config_yaml_text_for_log():
    """Read ``config/config.yaml`` as text (same path as ``config.read_config``)."""
    root = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(root, "config", "config.yaml")
    if not os.path.isfile(yaml_path):
        return yaml_path, f"(file not found: {yaml_path})"
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml_path, f.read()


def _append_summary_to_log(log_path, summary_text, header="tune.py final summary"):
    """Append config.yaml snapshot then final summary to tune2.log."""
    if not summary_text:
        return
    d = os.path.dirname(log_path)
    if d:
        os.makedirs(d, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ypath, ytext = _read_config_yaml_text_for_log()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"[{ts}] {header}\n")
        f.write("=" * 60 + "\n")
        f.write(f"config.yaml ({ypath})\n")
        f.write("-" * 60 + "\n")
        f.write(ytext)
        if ytext and not ytext.endswith("\n"):
            f.write("\n")
        f.write("-" * 60 + "\n\n")
        f.write(summary_text + "\n")


def _tune_log_stamp(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"\n{'=' * 60}\n[{ts}] {msg}\n{'=' * 60}\n")


def _print_args(ns, title="args"):
    """Print all Namespace fields in stable key order."""
    print(f"{title}:")
    for k in sorted(vars(ns)):
        print(f"  {k}: {getattr(ns, k)}")


if __name__ == "__main__":
    _tune_log_path = os.path.join('logs', "whole_{}.log".format(args.dataset))
    _tune_log_stamp("tune.py run start")

    METRIC_SPLIT = "all"
    key_auc = f"ada_{METRIC_SPLIT}_auc_mean"

    expanded_trials = _expand_search_trials(SEARCH_TRIALS)

    if not expanded_trials:
        _print_args(args, title="run args")
        seed_everything(args.seed)

        source_data = get_dataset(args, args.inid)
        target_data = get_dataset(args, args.outid)

        print("********************process source data********************")
        source_data = process_dataset(args, source_data)
        print("********************process target data********************")
        target_data = process_dataset(args, target_data)

        (
            src_acc,
            src_auc_roc,
            src_parity,
            src_equality,
            tgt_acc,
            tgt_auc_roc,
            tgt_parity,
            tgt_equality,
            ada_acc,
            ada_auc_roc,
            ada_parity,
            ada_equality,
        ) = train_and_adapt_metric_split(args, source_data, target_data, metric_split=METRIC_SPLIT)

        _print_summary(
            args,
            METRIC_SPLIT,
            src_acc,
            src_auc_roc,
            src_parity,
            src_equality,
            tgt_acc,
            tgt_auc_roc,
            tgt_parity,
            tgt_equality,
            ada_acc,
            ada_auc_roc,
            ada_parity,
            ada_equality,
        )
        final_summary_text = _build_single_run_summary_text(
            args,
            METRIC_SPLIT,
            src_acc,
            src_auc_roc,
            src_parity,
            src_equality,
            tgt_acc,
            tgt_auc_roc,
            tgt_parity,
            tgt_equality,
            ada_acc,
            ada_auc_roc,
            ada_parity,
            ada_equality,
        )
        _append_summary_to_log(_tune_log_path, final_summary_text)
    else:
        base_args = copy.deepcopy(args)
        results = []

        for ti, overrides in enumerate(expanded_trials):
            _tune_log_stamp(f"tune.py trial {ti + 1}/{len(expanded_trials)} start")

            trial_args = copy.deepcopy(base_args)
            _apply_overrides(trial_args, overrides)
            _refresh_device(trial_args)
            _print_args(trial_args, title=f"trial {ti} args")

            print("\n" + "=" * 72)
            print(
                f"[SEARCH_TRIALS] dataset={trial_args.dataset}  "
                f"{ti + 1}/{len(expanded_trials)}  overrides={overrides}"
            )
            print("=" * 72)

            seed_everything(trial_args.seed)

            source_data = get_dataset(trial_args, trial_args.inid)
            target_data = get_dataset(trial_args, trial_args.outid)

            print("********************process source data********************")
            source_data = process_dataset(trial_args, source_data)
            print("********************process target data********************")
            target_data = process_dataset(trial_args, target_data)

            (
                src_acc,
                src_auc_roc,
                src_parity,
                src_equality,
                tgt_acc,
                tgt_auc_roc,
                tgt_parity,
                tgt_equality,
                ada_acc,
                ada_auc_roc,
                ada_parity,
                ada_equality,
            ) = train_and_adapt_metric_split(
                trial_args, source_data, target_data, metric_split=METRIC_SPLIT
            )

            trial_note = f"  trial {ti}"
            _print_summary(
                trial_args,
                METRIC_SPLIT,
                src_acc,
                src_auc_roc,
                src_parity,
                src_equality,
                tgt_acc,
                tgt_auc_roc,
                tgt_parity,
                tgt_equality,
                ada_acc,
                ada_auc_roc,
                ada_parity,
                ada_equality,
                trial_note=trial_note,
            )

            results.append(
                {
                    "trial": ti,
                    "metric_split": METRIC_SPLIT,
                    "overrides": overrides,
                    "src": _metrics_mean_std(src_acc, src_auc_roc, src_parity, src_equality),
                    "tgt": _metrics_mean_std(tgt_acc, tgt_auc_roc, tgt_parity, tgt_equality),
                    "ada": _metrics_mean_std(ada_acc, ada_auc_roc, ada_parity, ada_equality),
                    "ada_acc_mean": float(np.mean(ada_acc)),
                    key_auc: float(np.mean(ada_auc_roc)),
                    "ada_dp_mean": float(np.mean(ada_parity)),
                    "ada_eo_mean": float(np.mean(ada_equality)),
                }
            )

        if len(results) > 1:
            best = max(results, key=lambda r: r[key_auc])
            print("\n" + "=" * 72)
            print(f"Best trial by mean adapted-target [{METRIC_SPLIT}] AUC-ROC (across runs):")
            print(best)
            print("=" * 72)

        final_summary_text = _print_trials_final_summary(results, METRIC_SPLIT)
        _append_summary_to_log(_tune_log_path, final_summary_text)
