# coding: utf-8
"""Source-only/source-free inference runner for FatraGNN.

FatraGNN already trains on a source graph and evaluates the trained model
under distribution shift.  This entry point keeps that minimal protocol:
train the original encoder/classifier/sensitive-discriminator components on
the labelled source graph, save only their parameters, release the source
graph, then run one direct forward pass on the target graph.  There is no
target optimizer, entropy minimization, pseudo-label, or target-validation
model selection.  Target annotations are used only by the final all-valid
evaluator and visualization exporter.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

# Import FatraGNN first.  model.py intentionally imports this directory's
# top-level utils.py, while SFFGNN/dataset.py only needs the compatible
# sens_correlation helper from that module.
from model import GCN_encoder_scatter, MLP_classifier, MLP_discriminator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from EMBER.dataset import get_dataset
from visualization.export_utils import save_visualization_embeddings

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


METHOD_NAME = "FatraGNN-SF"
VISUALIZATION_METHOD = "FatraGNN"
DATASET_ID_MAP = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

RESULT_DIR = SCRIPT_DIR / "results" / "fatragnn_sf"
RUN_DIR = RESULT_DIR / "runs"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
LOG_DIR = RESULT_DIR / "logs"
SUMMARY_FILE = RESULT_DIR / "FatraGNN-SF_summary.md"
METRIC_NAMES = ("Acc", "AUC", "DP", "EO")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cuda_id: int) -> torch.device:
    if cuda_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(cuda_id)
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def load_graph(dataset: str, domain_id: str, device: torch.device):
    """Use exactly the feature normalization and graph loader of SFFGNN."""
    data = get_dataset(SimpleNamespace(dataset=dataset), domain_id)
    return data.to(device)


def valid_binary_mask(data) -> torch.Tensor:
    """Keep only labels 0/1; in particular, exclude Pokec y=-1 nodes."""
    return (data.y == 0) | (data.y == 1)


def source_train_mask(data) -> torch.Tensor:
    return data.train_mask.bool() & valid_binary_mask(data)


def evaluation_mask(data) -> torch.Tensor:
    split_mask = data.train_mask.bool() | data.val_mask.bool() | data.test_mask.bool()
    return split_mask & valid_binary_mask(data)


def configure_model_args(args, num_features: int) -> SimpleNamespace:
    # The reused FatraGNN modules read these attributes from one namespace.
    return SimpleNamespace(
        num_features=num_features,
        hidden=args.hidden,
        num_classes=1,
        clip_e=args.clip_e,
        clip_c=args.clip_c,
        dropout=args.dropout,
    )


def build_models(num_features: int, args, device: torch.device):
    model_args = configure_model_args(args, num_features)
    encoder = GCN_encoder_scatter(model_args).to(device)
    classifier = MLP_classifier(model_args).to(device)
    discriminator = MLP_discriminator(model_args).to(device)
    # GCN_encoder_scatter owns a raw bias Parameter and, like the original
    # fatragnn.py run loop, must be reset explicitly before first use.
    encoder.reset_parameters()
    classifier.reset_parameters()
    discriminator.reset_parameters()
    return encoder, classifier, discriminator


def encode(encoder, data) -> torch.Tensor:
    # GCN_encoder_scatter keeps the original three-argument interface, though
    # its implementation uses edge_index and does not require adj_norm_sp.
    return encoder(data.x, data.edge_index, None)


def encode_view(
    encoder, target_x: torch.Tensor, target_edge_index: torch.Tensor
) -> torch.Tensor:
    """Encode a label-free graph view without accepting a full Data object."""
    return encoder(target_x, target_edge_index, None)


def checkpoint_path(dataset: str, seed: int) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{dataset}_seed{seed}_source.pt"


def run_result_path(dataset: str, seed: int) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"{dataset}_seed{seed}.json"


def log_path(dataset: str, seed: int) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{dataset}_seed{seed}.log"


def save_source_checkpoint(
    path: Path,
    encoder,
    classifier,
    discriminator,
    dataset: str,
    source_id: str,
    args,
) -> None:
    torch.save(
        {
            "method": METHOD_NAME,
            "dataset": dataset,
            "source_id": source_id,
            "seed": args.seed,
            "num_features": encoder.args.num_features,
            "hidden": args.hidden,
            "encoder_state": {
                name: value.detach().cpu()
                for name, value in encoder.state_dict().items()
            },
            "classifier_state": {
                name: value.detach().cpu()
                for name, value in classifier.state_dict().items()
            },
            "discriminator_state": {
                name: value.detach().cpu()
                for name, value in discriminator.state_dict().items()
            },
        },
        path,
    )


def stage1_pretrain_source(
    dataset: str,
    source_id: str,
    args,
    device: torch.device,
) -> Path:
    """Train FatraGNN on source labels/sensitive groups and export a checkpoint."""
    source_data = load_graph(dataset, source_id, device)
    encoder, classifier, discriminator = build_models(
        source_data.x.size(1), args, device
    )

    train_mask = source_train_mask(source_data)
    if int(train_mask.sum().item()) == 0:
        raise ValueError(f"{dataset} source split has no valid binary labels")

    sens_targets = source_data.sens_labels.float().view(-1)

    optimizer_encoder = torch.optim.Adam(
        encoder.parameters(), lr=args.encoder_lr, weight_decay=args.weight_decay
    )
    optimizer_classifier = torch.optim.Adam(
        classifier.parameters(),
        lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.discriminator_lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.source_epochs + 1):
        encoder.eval()
        discriminator.train()
        disc_value = float("nan")
        for _ in range(args.discriminator_steps):
            optimizer_discriminator.zero_grad()
            with torch.no_grad():
                detached_features = encode(encoder, source_data)
            sens_probability = discriminator(detached_features).view(-1)
            discriminator_loss = F.binary_cross_entropy(
                sens_probability, sens_targets
            )
            discriminator_loss.backward()
            optimizer_discriminator.step()
            disc_value = discriminator_loss.item()

        # The task update and the discriminator-confusion update are combined
        # into one encoder step.  This is the same min-max signal as the
        # original alternating FatraGNN loop without carrying unused gradients.
        encoder.train()
        classifier.train()
        discriminator.eval()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)

        optimizer_encoder.zero_grad()
        optimizer_classifier.zero_grad()
        features = encode(encoder, source_data)
        logits = classifier(features).view(-1)
        classification_loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], source_data.y[train_mask].float()
        )
        sens_probability = discriminator(features).view(-1)
        confusion_loss = F.mse_loss(
            sens_probability, torch.full_like(sens_probability, 0.5)
        )
        total_loss = classification_loss + args.source_fair_weight * confusion_loss
        total_loss.backward()
        optimizer_encoder.step()
        optimizer_classifier.step()

        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

        if args.verbose and (
            epoch % args.print_every == 0 or epoch == args.source_epochs
        ):
            print(
                f"[{dataset} seed={args.seed}] source epoch={epoch} "
                f"L_cls={classification_loss.item():.6f} "
                f"L_disc={disc_value:.6f} L_conf={confusion_loss.item():.6f}"
            )

    path = checkpoint_path(dataset, args.seed)
    save_source_checkpoint(
        path,
        encoder,
        classifier,
        discriminator,
        dataset,
        source_id,
        args,
    )
    print(f"[source checkpoint] {path}")

    # This explicit deletion is the source-free boundary.  stage2 receives
    # only a checkpoint path and independently loads the target graph.
    del source_data, encoder, classifier, discriminator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def load_source_models(
    checkpoint: Path,
    num_target_features: int,
    args,
    device: torch.device,
):
    artifact = torch.load(checkpoint, map_location=device)
    num_source_features = int(artifact["num_features"])
    if num_source_features != num_target_features:
        raise ValueError(
            "Source/target feature dimensions differ: "
            f"{num_source_features} vs {num_target_features}"
        )
    if int(artifact["hidden"]) != args.hidden:
        raise ValueError(
            f"Checkpoint hidden={artifact['hidden']} does not match --hidden={args.hidden}"
        )

    encoder, classifier, discriminator = build_models(
        num_target_features, args, device
    )
    encoder.load_state_dict(artifact["encoder_state"])
    classifier.load_state_dict(artifact["classifier_state"])
    discriminator.load_state_dict(artifact["discriminator_state"])
    encoder.eval()
    classifier.eval()
    discriminator.eval()
    return encoder, classifier


def stage2_evaluate_target(
    dataset: str,
    target_id: str,
    checkpoint: Path,
    args,
    device: torch.device,
):
    """Load target data and evaluate the frozen source-trained model once."""
    target_data = load_graph(dataset, target_id, device)
    encoder, classifier = load_source_models(
        checkpoint, target_data.x.size(1), args, device
    )

    with torch.no_grad():
        target_features = encode_view(
            encoder, target_data.x, target_data.edge_index
        )
        target_logits = classifier(target_features).view(-1)
    metrics = metrics_from_logits(target_logits, target_data)

    if args.save_visualization_embeddings:
        mask = evaluation_mask(target_data)
        feat_path, labels_path = save_visualization_embeddings(
            REPO_ROOT / "visualization" / "embeddings",
            VISUALIZATION_METHOD,
            dataset,
            target_features[mask].detach().cpu().numpy(),
            y=target_data.y[mask].detach().cpu().numpy(),
            sens=target_data.sens_labels[mask].detach().cpu().numpy(),
        )
        print(f"[embedding] {feat_path}")
        print(f"[embedding labels] {labels_path}")

    return metrics


def safe_gap(left: np.ndarray, right: np.ndarray) -> float:
    """Match SFFGNN.utils.fair_metric, including its empty-group policy."""
    if left.size == 0 or right.size == 0:
        return 0.0
    value = abs(float(left.mean()) - float(right.mean()))
    return value if np.isfinite(value) else 0.0


def fairness_metrics(
    predictions: np.ndarray, labels: np.ndarray, sensitive: np.ndarray
) -> tuple[float, float]:
    group0 = sensitive == 0
    group1 = sensitive == 1
    group0_negative = group0 & (labels == 0)
    group1_negative = group1 & (labels == 0)
    group0_positive = group0 & (labels == 1)
    group1_positive = group1 & (labels == 1)

    dp = safe_gap(predictions[group0], predictions[group1])
    # SFFGNN Eq. (2): average the sensitive-group gap in correct
    # predictions for y=0 and y=1 (equalized odds, not only TPR parity).
    y0_gap = safe_gap(
        predictions[group0_negative] == 0,
        predictions[group1_negative] == 0,
    )
    y1_gap = safe_gap(
        predictions[group0_positive] == 1,
        predictions[group1_positive] == 1,
    )
    eo = 0.5 * (y0_gap + y1_gap)
    return dp * 100.0, eo * 100.0


@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, data) -> dict[str, float]:
    mask = evaluation_mask(data)
    if int(mask.sum().item()) == 0:
        raise ValueError("Target evaluation union has no valid binary labels")

    probabilities = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    predictions = (probabilities > 0.5).astype(np.int64)
    labels = data.y[mask].detach().cpu().numpy().astype(np.int64)
    sensitive = (
        data.sens_labels[mask].detach().cpu().numpy().astype(np.int64)
    )

    accuracy = float((predictions == labels).mean() * 100.0)
    if roc_auc_score is not None and np.unique(labels).size == 2:
        auc = float(roc_auc_score(labels, probabilities) * 100.0)
    else:
        auc = float("nan")
    dp, eo = fairness_metrics(predictions, labels, sensitive)
    return {"Acc": accuracy, "AUC": auc, "DP": dp, "EO": eo}


def write_run_result(
    dataset: str,
    source_id: str,
    target_id: str,
    metrics: dict[str, float],
    args,
) -> Path:
    path = run_result_path(dataset, args.seed)
    payload = {
        "method": METHOD_NAME,
        "dataset": dataset,
        "source_id": source_id,
        "target_id": target_id,
        "seed": args.seed,
        "evaluation_protocol": "source_only_direct_target_inference",
        "eval_split": "train|val|test with y in {0,1}",
        "metrics": metrics,
        "hyperparameters": {
            "hidden": args.hidden,
            "source_epochs": args.source_epochs,
            "encoder_lr": args.encoder_lr,
            "classifier_lr": args.classifier_lr,
            "discriminator_lr": args.discriminator_lr,
            "source_fair_weight": args.source_fair_weight,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    return path


def run_single(args) -> Path:
    source_id, target_id = DATASET_ID_MAP[args.dataset]
    seed_everything(args.seed)
    device = get_device(args.cuda)
    print(
        f"[{METHOD_NAME}] dataset={args.dataset} seed={args.seed} "
        f"source={source_id} target={target_id} device={device}"
    )

    checkpoint = stage1_pretrain_source(
        args.dataset, source_id, args, device
    )
    metrics = stage2_evaluate_target(
        args.dataset, target_id, checkpoint, args, device
    )
    result = write_run_result(
        args.dataset,
        source_id,
        target_id,
        metrics,
        args,
    )
    print(f"[run result] {result}")
    print(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )
    return result


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_single_with_log(args) -> Path:
    """Run one seed and always keep stdout/stderr in its own log file."""
    path = log_path(args.dataset, args.seed)
    with path.open("w", encoding="utf-8") as handle:
        tee_out = Tee(sys.stdout, handle)
        tee_err = Tee(sys.stderr, handle)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            try:
                result = run_single(args)
            except Exception:
                traceback.print_exc()
                raise
    return result


def finite_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "variance": float(np.nanvar(array)),
    }


def aggregate_all(datasets: list[str], seeds: list[int]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    raw_rows = []
    for dataset in datasets:
        metrics = []
        for seed in seeds:
            path = run_result_path(dataset, seed)
            if not path.exists():
                raise FileNotFoundError(f"Missing run result: {path}")
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)["metrics"]
            metrics.append(row)
            raw_rows.append((dataset, seed, row))
        summaries[dataset] = {
            metric: finite_summary([row[metric] for row in metrics])
            for metric in METRIC_NAMES
        }

    with SUMMARY_FILE.open("w", encoding="utf-8") as handle:
        handle.write("# FatraGNN-SF: 5-run summary\n\n")
        handle.write(
            "Values are percentages. Main cells report population mean +/- "
            "population standard deviation; the following table reports "
            "population variance explicitly.\n\n"
        )
        handle.write("| Dataset | ACC | AUC | DP | EO |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for dataset in datasets:
            cells = []
            for metric in METRIC_NAMES:
                stats = summaries[dataset][metric]
                cells.append(f"{stats['mean']:.2f} +/- {stats['std']:.2f}")
            handle.write(f"| {dataset} | " + " | ".join(cells) + " |\n")

        handle.write("\n## Population variance\n\n")
        handle.write("| Dataset | ACC var | AUC var | DP var | EO var |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for dataset in datasets:
            cells = [
                f"{summaries[dataset][metric]['variance']:.4f}"
                for metric in METRIC_NAMES
            ]
            handle.write(f"| {dataset} | " + " | ".join(cells) + " |\n")

        handle.write("\n## Raw runs\n\n")
        handle.write("| Dataset | Seed | ACC | AUC | DP | EO |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for dataset, seed, row in raw_rows:
            cells = [f"{row[metric]:.4f}" for metric in METRIC_NAMES]
            handle.write(
                f"| {dataset} | {seed} | " + " | ".join(cells) + " |\n"
            )
    return SUMMARY_FILE


def parse_gpu_ids(value: str) -> list[int]:
    ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(-1 if item.lower() == "cpu" else int(item))
    if not torch.cuda.is_available():
        # Keep several independent workers on CPU-only hosts, capped to avoid
        # oversubscribing memory on the larger Pokec graphs.
        cpu_slots = min(len(ids) or 1, max(1, min(4, os.cpu_count() or 1)))
        return [-1] * cpu_slots
    return ids or [-1]


def worker_command(args, dataset: str, seed: int, cuda_id: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--cuda",
        str(cuda_id),
    ]
    scalar_options = {
        "hidden": args.hidden,
        "dropout": args.dropout,
        "clip_e": args.clip_e,
        "clip_c": args.clip_c,
        "source_epochs": args.source_epochs,
        "discriminator_steps": args.discriminator_steps,
        "encoder_lr": args.encoder_lr,
        "classifier_lr": args.classifier_lr,
        "discriminator_lr": args.discriminator_lr,
        "weight_decay": args.weight_decay,
        "source_fair_weight": args.source_fair_weight,
        "print_every": args.print_every,
    }
    for name, value in scalar_options.items():
        command.extend((f"--{name}", str(value)))
    if args.verbose:
        command.append("--verbose")
    if args.save_visualization_embeddings and seed == args.embedding_seed:
        command.append("--save_visualization_embeddings")
    else:
        command.append("--no_save_visualization_embeddings")
    return command


def launch_parallel(args) -> Path:
    """Launch one process per dataset/seed with at most one process per GPU."""
    tasks = [(dataset, seed) for dataset in args.datasets for seed in args.seeds]
    gpu_ids = parse_gpu_ids(args.gpus)
    free_gpus = list(gpu_ids)
    running = {}

    while tasks or running:
        while tasks and free_gpus:
            dataset, seed = tasks.pop(0)
            cuda_id = free_gpus.pop(0)
            command = worker_command(args, dataset, seed, cuda_id)
            process = subprocess.Popen(
                command,
                cwd=str(SCRIPT_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            running[process] = (dataset, seed, cuda_id)
            print(
                f"[launch] {dataset} seed={seed} -> "
                f"{'CPU' if cuda_id < 0 else f'GPU {cuda_id}'}; "
                f"log={log_path(dataset, seed)}"
            )

        time.sleep(1)
        for process in list(running):
            return_code = process.poll()
            if return_code is None:
                continue
            dataset, seed, cuda_id = running.pop(process)
            free_gpus.append(cuda_id)
            if return_code != 0:
                for remaining in running:
                    remaining.terminate()
                raise RuntimeError(
                    f"{dataset} seed={seed} failed; see {log_path(dataset, seed)}"
                )
            print(f"[done] {dataset} seed={seed}")

    summary = aggregate_all(args.datasets, args.seeds)
    print(f"[summary] {summary}")
    return summary


def get_args():
    parser = argparse.ArgumentParser(
        description="FatraGNN source-only direct target evaluation runner"
    )
    parser.add_argument("--worker", action="store_true", help="run one dataset/seed")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="run all tasks in parallel (this is already the default mode)",
    )
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument(
        "--dataset", choices=list(DATASET_ID_MAP), default=None
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_ID_MAP),
        default=["bailA", "germanA", "pokec", "syn"],
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1111, 2222, 3333, 4444, 5555]
    )
    parser.add_argument("--cuda", "--gpu", dest="cuda", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--save_visualization_embeddings", action="store_true", default=True
    )
    parser.add_argument("--save_embeddings", dest="save_visualization_embeddings", action="store_true")
    parser.add_argument(
        "--no_save_visualization_embeddings",
        dest="save_visualization_embeddings",
        action="store_false",
    )
    parser.add_argument("--embedding_seed", type=int, default=1111)

    # Original FatraGNN defaults where applicable.
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip_e", type=float, default=1.0)
    parser.add_argument("--clip_c", type=float, default=1.0)
    parser.add_argument("--source_epochs", type=int, default=400)
    parser.add_argument("--discriminator_steps", type=int, default=2)
    parser.add_argument("--encoder_lr", type=float, default=0.005)
    parser.add_argument("--classifier_lr", type=float, default=0.005)
    parser.add_argument("--discriminator_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--source_fair_weight", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    if args.aggregate_only:
        summary = aggregate_all(args.datasets, args.seeds)
        print(f"[summary] {summary}")
    elif args.worker:
        if args.dataset is None:
            raise ValueError("--worker requires --dataset")
        run_single_with_log(args)
    else:
        # Default invocation: four datasets x five seeds via the device pool.
        launch_parallel(args)


if __name__ == "__main__":
    main()
