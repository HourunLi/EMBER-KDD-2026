"""Run SOGA on the four EMBER/SFDA benchmark domain pairs.

SOGA is a source-free graph domain-adaptation method, but the upstream demo
mixes source pretraining and target adaptation in one procedure and uses target
labels after every adaptation epoch to select a checkpoint.  This benchmark
runner keeps SOGA's model and optimization objectives unchanged while enforcing
the source-free experiment protocol:

1. Pretrain only on the source graph and save a model-only checkpoint.
2. Release the source graph, create a fresh model, load the checkpoint, and
   adapt using only target features/edges with SOGA's IM + two NCE losses.
3. Read target labels only once, after the final target epoch, for all-valid
   ACC/AUC/DP/EO evaluation and optional first-run embedding export.

Running ``python run.py`` launches 4 datasets x 5 seeds.  Jobs are scheduled
one per detected GPU (or across CPU workers when CUDA is unavailable), every
run has a separate log under ``logs/``, and the final aggregate is written to
``SOGA_5runs_summary.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODES_DIR = SCRIPT_DIR / "codes"

# The upstream SOGA code uses absolute imports such as ``model.NCE_utilies``.
# Put its codes directory first so those imports resolve to this checkout.
for import_path in (REPO_ROOT, CODES_DIR):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from EMBER.dataset import get_dataset  # noqa: E402
from EMBER.utils import fair_metric  # noqa: E402
from model.NCE_utilies import Negative_Sampler, RandomWalker  # noqa: E402
from model.model import Model  # noqa: E402
from visualization.export_utils import save_visualization_embeddings  # noqa: E402


METHOD = "SOGA"
DATASET_PAIRS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}
DEFAULT_DATASETS = list(DATASET_PAIRS)
DEFAULT_SEEDS = [1, 3, 5, 7, 9]
METRIC_NAMES = ("ACC", "AUC", "DP", "EO")

LOG_DIR = SCRIPT_DIR / "logs"
RESULT_DIR = SCRIPT_DIR / "results"
RUN_DIR = RESULT_DIR / "runs"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
SUMMARY_FILE = SCRIPT_DIR / "SOGA_5runs_summary.md"


def seed_everything(seed: int) -> None:
    """Seed the RNGs used by EMBER loading, SOGA sampling, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(cuda_id: int) -> torch.device:
    if cuda_id >= 0 and torch.cuda.is_available():
        if cuda_id >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {cuda_id} is unavailable; "
                f"visible device count is {torch.cuda.device_count()}."
            )
        torch.cuda.set_device(cuda_id)
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def make_logger(dataset: str, run_idx: int) -> logging.Logger:
    """Log to stdout; the parent process redirects each worker to its own file."""
    logger = logging.getLogger(f"soga.{dataset}.run{run_idx}.{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def load_graph(dataset: str, domain_id: str) -> Data:
    """Reuse EMBER/dataset.py without changing paths, normalization, or masks."""
    return get_dataset(SimpleNamespace(dataset=dataset), domain_id)


def valid_label_mask(data: Data) -> torch.Tensor:
    """Binary valid labels; in particular, exclude Pokec's y=-1 nodes."""
    return (data.y == 0) | (data.y == 1)


def all_valid_mask(data: Data) -> torch.Tensor:
    """EMBER target-all split: union(train, val, test), restricted to valid y."""
    return (
        data.train_mask | data.val_mask | data.test_mask
    ) & valid_label_mask(data)


def infer_num_classes(data: Data) -> int:
    labels = data.y[valid_label_mask(data)]
    if labels.numel() == 0:
        raise ValueError("The graph contains no valid binary labels.")
    num_classes = int(labels.max().item()) + 1
    if num_classes != 2:
        raise ValueError(
            f"This fair benchmark expects two classes, found {num_classes}."
        )
    return num_classes


def layer_units(num_features: int, num_classes: int, num_layers: int) -> list[int]:
    """Match SOGA codes/model/model_utilies.py:init_layer exactly."""
    hidden_by_layers = {
        2: [100],
        3: [256, 128],
        4: [256, 128, 64],
        5: [32, 32, 32, 32],
        7: [256, 128, 64, 32, 32, 16],
        9: [256, 128, 64, 64, 32, 32, 16, 16],
    }
    if num_layers not in hidden_by_layers:
        raise ValueError(
            f"Unsupported --num-layers={num_layers}; choose one of "
            f"{sorted(hidden_by_layers)}."
        )
    return [num_features, *hidden_by_layers[num_layers], num_classes]


def make_model_args(
    num_features: int,
    num_classes: int,
    num_nodes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_unit_count_list=layer_units(
            num_features, num_classes, args.num_layers
        ),
        gnn_model=args.gnn_model,
        head=args.head,
        num_label=num_classes,
        num_target_nodes=num_nodes,
        num_negative_samples=args.num_negative_samples,
        num_positive_samples=args.num_positive_samples,
        struct_lambda=args.struct_lambda,
        neigh_lambda=args.neigh_lambda,
        metric="macro",
        device=device,
    )


def build_model(
    num_features: int,
    num_classes: int,
    num_nodes: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[Model, SimpleNamespace]:
    model_args = make_model_args(
        num_features, num_classes, num_nodes, args, device
    )
    model = Model(model_args, logger).to(device)
    # These are fixed to 5 and 2 in the upstream constructor.  Assigning the
    # CLI-configured values preserves the original defaults while making the
    # benchmark configuration explicit in one file.
    model.num_negative_samples = args.num_negative_samples
    model.num_positive_samples = args.num_positive_samples
    return model, model_args


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def checkpoint_path(dataset: str, run_idx: int) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"SOGA_{dataset}_run{run_idx}_source_model.pt"


def run_result_path(dataset: str, run_idx: int) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"SOGA_{dataset}_run{run_idx}.json"


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()
    return float(f1_score(truth, pred, average="macro", zero_division=0))


def stage1_pretrain_source(
    dataset: str,
    source_id: str,
    run_idx: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> Path:
    """Source-only supervised pretraining; save weights, never source samples."""
    logger.info(
        "stage1/source start | dataset=%s domain=%s device=%s",
        dataset,
        source_id,
        device,
    )
    source_data = load_graph(dataset, source_id).to(device)
    num_classes = infer_num_classes(source_data)
    num_features = int(source_data.x.shape[1])
    num_nodes = int(source_data.num_nodes)
    model, model_args = build_model(
        num_features, num_classes, num_nodes, args, device, logger
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.source_lr,
        weight_decay=args.weight_decay,
    )

    train_mask = source_data.train_mask & valid_label_mask(source_data)
    val_mask = source_data.val_mask & valid_label_mask(source_data)
    if not bool(train_mask.any()):
        raise ValueError(f"{dataset}{source_id} has an empty source train mask.")
    if not bool(val_mask.any()):
        logger.warning("source val mask is empty; falling back to source train mask")
        val_mask = train_mask

    best_val_f1 = -float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(args.source_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(source_data)
        # SOGA's custom CrossEntropy is standard CE with epsilon=0.1 label
        # smoothing.  The functional form avoids its hard-coded .cuda() call.
        loss = F.cross_entropy(
            logits[train_mask],
            source_data.y[train_mask],
            label_smoothing=0.1,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(source_data)[val_mask]
            val_f1 = macro_f1(val_logits, source_data.y[val_mask])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = cpu_state_dict(model)

        if (
            epoch == 0
            or (epoch + 1) % args.print_every == 0
            or epoch + 1 == args.source_epochs
        ):
            logger.info(
                "stage1/source epoch=%03d/%03d loss=%.6f val_macro_f1=%.6f",
                epoch + 1,
                args.source_epochs,
                loss.item(),
                val_f1,
            )

    if best_state is None:
        raise RuntimeError("Source pretraining did not produce a checkpoint.")

    path = checkpoint_path(dataset, run_idx)
    torch.save(
        {
            "method": METHOD,
            "artifact": "source_model_only",
            "dataset": dataset,
            "source_id": source_id,
            "run_idx": run_idx,
            "seed": seed,
            "best_source_epoch": best_epoch + 1,
            "best_source_val_macro_f1": best_val_f1,
            "num_features": num_features,
            "num_classes": num_classes,
            "layer_unit_count_list": model_args.layer_unit_count_list,
            "model_state": best_state,
        },
        path,
    )
    logger.info(
        "stage1/source complete | best_epoch=%d best_val_macro_f1=%.6f "
        "checkpoint=%s",
        best_epoch + 1,
        best_val_f1,
        path,
    )

    # The target stage receives only the checkpoint path.  Explicit deletion
    # makes the source-free boundary visible and prevents accidental reuse.
    del logits, val_logits, optimizer, model, source_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def make_walk_samples(
    graph,
    num_nodes: int,
    num_positive_samples: int,
    num_negative_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce SOGA's node2vec positive and degree-based negative samples."""
    if num_positive_samples != 2:
        raise ValueError(
            "The upstream SOGA NCE implementation expects exactly two nodes "
            "per walk; keep --num-positive-samples=2."
        )

    positive_sampler = RandomWalker(
        graph,
        p=0.25,
        q=2,
        use_rejection_sampling=1,
    )
    positive_sampler.preprocess_transition_probs()
    walks = positive_sampler.simulate_walks(
        num_walks=1,
        walk_length=num_positive_samples,
        workers=1,
        verbose=0,
    )
    if len(walks) != num_nodes:
        raise RuntimeError(
            f"Expected one positive walk per node ({num_nodes}), got {len(walks)}."
        )

    pairs = np.empty((num_nodes, 2), dtype=np.int64)
    for index, walk in enumerate(walks):
        if not walk:
            raise RuntimeError("SOGA random walk unexpectedly returned an empty walk.")
        pairs[index, 0] = walk[0]
        pairs[index, 1] = walk[1] if len(walk) > 1 else walk[0]

    negative_sampler = Negative_Sampler(graph)
    negative_count = num_nodes * num_negative_samples
    negatives = np.fromiter(
        (negative_sampler.sample() for _ in range(negative_count)),
        dtype=np.int64,
        count=negative_count,
    ).reshape(num_nodes, num_negative_samples)

    sample_tensor = torch.from_numpy(pairs).to(device=device, dtype=torch.long)
    center_nodes = sample_tensor[:, :1]
    positive_nodes = sample_tensor[:, 1:2]
    negative_nodes = torch.from_numpy(negatives).to(
        device=device, dtype=torch.long
    )
    return center_nodes, positive_nodes, negative_nodes


def initialize_target_samplers(
    model: Model,
    target_graph_cpu: Data,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Initialize SOGA's two independently sampled NCE views.

    The upstream checkout effectively sends the same target ``edge_index`` to
    both samplers: it writes a precomputed graph into ``edge_idx`` (not
    ``edge_index``), so PyG's ``to_networkx`` never observes that replacement.
    EMBER also provides no separate SOGA structure graph.  Using the same target
    adjacency twice therefore preserves the effective upstream computation,
    while the random walks/negatives remain independently drawn for each term.
    """
    graph_only = Data(
        edge_index=target_graph_cpu.edge_index,
        num_nodes=target_graph_cpu.num_nodes,
    )
    graph = to_networkx(graph_only)
    num_nodes = int(target_graph_cpu.num_nodes)

    logger.info(
        "target sampling start | nodes=%d edges=%d negatives_per_node=%d",
        num_nodes,
        int(target_graph_cpu.edge_index.shape[1]),
        args.num_negative_samples,
    )
    (
        model.center_nodes_struct,
        model.positive_samples_struct,
        model.negative_samples_struct,
    ) = make_walk_samples(
        graph,
        num_nodes,
        args.num_positive_samples,
        args.num_negative_samples,
        device,
    )
    (
        model.center_nodes_neigh,
        model.positive_samples_neigh,
        model.negative_samples_neigh,
    ) = make_walk_samples(
        graph,
        num_nodes,
        args.num_positive_samples,
        args.num_negative_samples,
        device,
    )
    logger.info("target sampling complete")


def build_target_inputs(data: Data) -> Data:
    """Strip a loaded target graph down to the tensors allowed in adaptation."""
    return Data(
        x=data.x,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
    )


def build_evaluator(data: Data) -> dict[str, torch.Tensor]:
    """Build evaluation-only tensors after target adaptation has finished."""
    return {
        "y": data.y.detach().cpu().clone(),
        "sens": data.sens_labels.detach().cpu().clone(),
        "all_valid_mask": all_valid_mask(data).detach().cpu().clone(),
    }


def evaluate_target_all_valid(
    model: Model,
    target_inputs: Data,
    evaluator: dict[str, torch.Tensor],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate only after adaptation on target all(valid), in percent."""
    model.eval()
    with torch.no_grad():
        features = model.Extractor(target_inputs.x, target_inputs.edge_index)
        logits = model.Classifier(features)
        mask_cpu = evaluator["all_valid_mask"]
        mask_device = mask_cpu.to(target_inputs.x.device)
        probs = F.softmax(logits, dim=1)[:, 1][mask_device].cpu().numpy()
        pred = logits.argmax(dim=1)[mask_device].cpu().numpy()
        representations = features[mask_device].cpu().numpy()

    labels = evaluator["y"][mask_cpu].numpy()
    sens = evaluator["sens"][mask_cpu].numpy()
    if labels.size == 0:
        raise ValueError("Target all(valid) evaluation mask is empty.")

    acc = accuracy_score(labels, pred) * 100.0
    auc = (
        roc_auc_score(labels, probs) * 100.0
        if np.unique(labels).size == 2
        else 50.0
    )
    valid_sensitive = (sens == 0) | (sens == 1)
    dp, eo = fair_metric(
        pred[valid_sensitive],
        labels[valid_sensitive],
        sens[valid_sensitive],
    )
    metrics = {
        "ACC": float(acc),
        "AUC": float(auc),
        "DP": float(dp * 100.0),
        "EO": float(eo * 100.0),
    }
    return metrics, representations, labels, sens


def stage2_adapt_target(
    dataset: str,
    target_id: str,
    run_idx: int,
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, float]:
    """Target-only SOGA adaptation followed by one final all-valid evaluation."""
    logger.info(
        "stage2/target start | dataset=%s domain=%s device=%s",
        dataset,
        target_id,
        device,
    )
    target_data = load_graph(dataset, target_id)
    target_inputs_cpu = build_target_inputs(target_data)
    num_features = int(target_inputs_cpu.x.shape[1])
    num_nodes = int(target_inputs_cpu.num_nodes)
    # Do not retain a Data object exposing y/sensitive attributes/masks during
    # adaptation.  Evaluation reloads those tensors only after the final epoch.
    del target_data

    artifact = torch.load(checkpoint, map_location="cpu")
    num_classes = int(artifact["num_classes"])
    if num_features != int(artifact["num_features"]):
        raise ValueError(
            f"Source/target feature mismatch for {dataset}: "
            f"{artifact['num_features']} vs {num_features}."
        )

    model, model_args = build_model(
        num_features, num_classes, num_nodes, args, device, logger
    )
    if model_args.layer_unit_count_list != artifact["layer_unit_count_list"]:
        raise ValueError("Current model settings do not match the source checkpoint.")
    model.load_state_dict(artifact["model_state"])

    # From here until evaluate_target_all_valid(), neither source samples nor
    # target y/sensitive attributes/masks are passed to the adaptation code.
    initialize_target_samplers(model, target_inputs_cpu, args, device, logger)
    target_inputs = target_inputs_cpu.to(device)
    del target_inputs_cpu, artifact
    gc.collect()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.target_lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.target_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(target_inputs)
        probs = F.softmax(logits, dim=-1)
        struct_nce = model.NCE_loss(
            probs,
            model.center_nodes_struct,
            model.positive_samples_struct,
            model.negative_samples_struct,
        )
        neigh_nce = model.NCE_loss(
            probs,
            model.center_nodes_neigh,
            model.positive_samples_neigh,
            model.negative_samples_neigh,
        )
        entropy = model.ent(probs)
        diversity = model.div(probs)
        information_maximization = entropy - diversity
        loss = (
            information_maximization
            + args.struct_lambda * struct_nce
            + args.neigh_lambda * neigh_nce
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite target loss at epoch {epoch + 1}: {loss.item()}"
            )
        loss.backward()
        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % args.print_every == 0
            or epoch + 1 == args.target_epochs
        ):
            logger.info(
                "stage2/target epoch=%03d/%03d total=%.6f im=%.6f "
                "nce_struct=%.6f nce_neigh=%.6f",
                epoch + 1,
                args.target_epochs,
                loss.item(),
                information_maximization.item(),
                struct_nce.item(),
                neigh_nce.item(),
            )

    target_eval_data = load_graph(dataset, target_id)
    if int(target_eval_data.num_nodes) != num_nodes:
        raise ValueError("Reloaded target evaluation graph has a different size.")
    evaluator = build_evaluator(target_eval_data)
    del target_eval_data
    metrics, representations, labels, sens = evaluate_target_all_valid(
        model, target_inputs, evaluator
    )
    logger.info(
        "target all(valid) | ACC=%.4f AUC=%.4f DP=%.4f EO=%.4f nodes=%d",
        metrics["ACC"],
        metrics["AUC"],
        metrics["DP"],
        metrics["EO"],
        labels.shape[0],
    )

    if args.save_embeddings and run_idx == 0:
        feat_path, labels_path = save_visualization_embeddings(
            REPO_ROOT / "visualization" / "embeddings",
            METHOD,
            dataset,
            representations,
            y=labels,
            sens=sens,
        )
        logger.info("saved first-run embeddings | %s", feat_path)
        logger.info("saved first-run labels | %s", labels_path)

    del optimizer, model, target_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def write_run_result(
    dataset: str,
    source_id: str,
    target_id: str,
    run_idx: int,
    seed: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> Path:
    path = run_result_path(dataset, run_idx)
    payload = {
        "method": METHOD,
        "protocol": "source-model-only target adaptation; final target all(valid)",
        "dataset": dataset,
        "source_id": source_id,
        "target_id": target_id,
        "run_idx": run_idx,
        "seed": seed,
        "metrics": metrics,
        "config": {
            "source_epochs": args.source_epochs,
            "target_epochs": args.target_epochs,
            "source_lr": args.source_lr,
            "target_lr": args.target_lr,
            "weight_decay": args.weight_decay,
            "num_layers": args.num_layers,
            "gnn_model": args.gnn_model,
            "head": args.head,
            "struct_lambda": args.struct_lambda,
            "neigh_lambda": args.neigh_lambda,
            "num_positive_samples": args.num_positive_samples,
            "num_negative_samples": args.num_negative_samples,
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_single(args: argparse.Namespace) -> None:
    source_id, target_id = DATASET_PAIRS[args.dataset]
    seed_everything(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    device = get_device(args.cuda)
    logger = make_logger(args.dataset, args.run_idx)
    logger.info(
        "worker start | method=%s dataset=%s run=%d seed=%d "
        "source=%s target=%s device=%s",
        METHOD,
        args.dataset,
        args.run_idx,
        args.seed,
        source_id,
        target_id,
        device,
    )
    logger.info(
        "config | source_epochs=%d target_epochs=%d source_lr=%g "
        "target_lr=%g weight_decay=%g layers=%d gnn=%s "
        "struct_lambda=%g neigh_lambda=%g positive=%d negative=%d",
        args.source_epochs,
        args.target_epochs,
        args.source_lr,
        args.target_lr,
        args.weight_decay,
        args.num_layers,
        args.gnn_model,
        args.struct_lambda,
        args.neigh_lambda,
        args.num_positive_samples,
        args.num_negative_samples,
    )

    checkpoint = stage1_pretrain_source(
        args.dataset,
        source_id,
        args.run_idx,
        args.seed,
        args,
        device,
        logger,
    )
    metrics = stage2_adapt_target(
        args.dataset,
        target_id,
        args.run_idx,
        checkpoint,
        args,
        device,
        logger,
    )
    result_path = write_run_result(
        args.dataset,
        source_id,
        target_id,
        args.run_idx,
        args.seed,
        metrics,
        args,
    )
    logger.info("worker complete | result=%s", result_path)
    print("RESULT_JSON=" + json.dumps(metrics, sort_keys=True), flush=True)


def load_results(
    datasets: list[str], seeds: list[int]
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    missing = []
    for dataset in datasets:
        rows = []
        for run_idx, expected_seed in enumerate(seeds):
            path = run_result_path(dataset, run_idx)
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload["seed"]) != int(expected_seed):
                raise ValueError(
                    f"Seed mismatch in {path}: expected {expected_seed}, "
                    f"found {payload['seed']}."
                )
            rows.append(payload)
        results[dataset] = rows
    if missing:
        raise FileNotFoundError(
            "Cannot aggregate because result files are missing:\n" + "\n".join(missing)
        )
    return results


def metric_statistics(rows: list[dict], metric: str) -> dict[str, float]:
    values = np.asarray(
        [row["metrics"][metric] for row in rows], dtype=np.float64
    )
    return {
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=0)),
        "std": float(np.std(values, ddof=0)),
    }


def aggregate_all(datasets: list[str], seeds: list[int]) -> Path:
    """Write one Markdown containing mean, population variance, std, and runs."""
    results = load_results(datasets, seeds)
    lines = [
        "# SOGA SFDA Benchmark Results",
        "",
        f"- Datasets: {', '.join(datasets)}",
        f"- Runs per dataset: {len(seeds)} (seeds: {', '.join(map(str, seeds))})",
        "- Evaluation: target `all = train | val | test`, filtered to valid binary labels",
        "- Pokec: nodes with `y = -1` are excluded from metrics and embedding export",
        "- Variance: population variance over the runs (`ddof=0`)",
        "- Target checkpoint: final adaptation epoch; target labels are not used for selection",
        "",
        "## Mean and variance",
        "",
        "| Dataset | Metric | Mean | Variance | Std |",
        "|---|---:|---:|---:|---:|",
    ]

    stats_by_dataset: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in datasets:
        stats_by_dataset[dataset] = {}
        for metric in METRIC_NAMES:
            stats = metric_statistics(results[dataset], metric)
            stats_by_dataset[dataset][metric] = stats
            lines.append(
                f"| {dataset} | {metric} | {stats['mean']:.4f} | "
                f"{stats['variance']:.6f} | {stats['std']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Baseline-table format (mean ± std)",
            "",
            "| Dataset | ACC | AUC | DP | EO |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in datasets:
        values = []
        for metric in METRIC_NAMES:
            stats = stats_by_dataset[dataset][metric]
            values.append(f"{stats['mean']:.2f} ± {stats['std']:.2f}")
        lines.append(f"| {dataset} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Raw runs",
            "",
            "| Dataset | Run | Seed | ACC | AUC | DP | EO |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in datasets:
        for row in results[dataset]:
            metrics = row["metrics"]
            lines.append(
                f"| {dataset} | {int(row['run_idx']) + 1} | {row['seed']} | "
                f"{metrics['ACC']:.4f} | {metrics['AUC']:.4f} | "
                f"{metrics['DP']:.4f} | {metrics['EO']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Per-run logs: `logs/{dataset}_run{1..5}.log`",
            "- Per-run JSON: `results/runs/SOGA_{dataset}_run{0..4}.json`",
            "- Source-model checkpoints: `results/checkpoints/`",
            "- First-run embeddings: `../../visualization/embeddings/SOGA/{dataset}/`",
            "",
        ]
    )
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    return SUMMARY_FILE


def parse_gpu_ids(spec: str) -> list[int]:
    normalized = spec.strip().lower()
    if normalized in {"", "auto"}:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    if normalized in {"cpu", "-1", "none"}:
        return []
    ids = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("--gpus contains duplicate device IDs.")
    if not torch.cuda.is_available():
        raise RuntimeError("--gpus was specified, but CUDA is unavailable.")
    unavailable = [item for item in ids if item < 0 or item >= torch.cuda.device_count()]
    if unavailable:
        raise ValueError(
            f"Unavailable CUDA IDs {unavailable}; visible count is "
            f"{torch.cuda.device_count()}."
        )
    return ids


def worker_command(
    args: argparse.Namespace,
    dataset: str,
    run_idx: int,
    seed: int,
    cuda_id: int,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        dataset,
        "--run-idx",
        str(run_idx),
        "--seed",
        str(seed),
        "--cuda",
        str(cuda_id),
        "--source-epochs",
        str(args.source_epochs),
        "--target-epochs",
        str(args.target_epochs),
        "--source-lr",
        str(args.source_lr),
        "--target-lr",
        str(args.target_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--num-layers",
        str(args.num_layers),
        "--gnn-model",
        args.gnn_model,
        "--head",
        str(args.head),
        "--struct-lambda",
        str(args.struct_lambda),
        "--neigh-lambda",
        str(args.neigh_lambda),
        "--num-positive-samples",
        str(args.num_positive_samples),
        "--num-negative-samples",
        str(args.num_negative_samples),
        "--print-every",
        str(args.print_every),
        "--cpu-threads",
        str(args.cpu_threads),
    ]
    if not args.save_embeddings:
        command.append("--no-save-embeddings")
    return command


def terminate_workers(running: dict[subprocess.Popen, dict]) -> None:
    for process in running:
        if process.poll() is None:
            process.terminate()
    for process, info in list(running.items()):
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        info["log_handle"].close()


def launch_parallel(args: argparse.Namespace) -> None:
    """Schedule all dataset/run pairs across GPUs or a CPU worker pool."""
    if len(args.seeds) != 5:
        raise ValueError(
            f"This benchmark is configured for exactly 5 runs; got {len(args.seeds)} seeds."
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_gpu_ids(args.gpus)
    if gpu_ids:
        resources = [
            gpu_id
            for gpu_id in gpu_ids
            for _ in range(args.workers_per_gpu)
        ]
        resource_description = ",".join(map(str, gpu_ids))
    else:
        resources = [-1] * args.cpu_workers
        resource_description = f"CPU x {args.cpu_workers}"
    if not resources:
        raise ValueError("No execution resources are configured.")

    tasks = [
        (dataset, run_idx, seed)
        for run_idx, seed in enumerate(args.seeds)
        for dataset in args.datasets
    ]
    queue = list(tasks)
    free_resources = list(resources)
    running: dict[subprocess.Popen, dict] = {}
    completed = 0

    print(
        f"[SOGA] launching {len(tasks)} jobs on {resource_description}; "
        f"logs: {LOG_DIR}",
        flush=True,
    )

    try:
        while queue or running:
            while queue and free_resources:
                dataset, run_idx, seed = queue.pop(0)
                cuda_id = free_resources.pop(0)
                log_path = LOG_DIR / f"{dataset}_run{run_idx + 1}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                command = worker_command(
                    args, dataset, run_idx, seed, cuda_id
                )
                environment = os.environ.copy()
                environment.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
                environment.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
                process = subprocess.Popen(
                    command,
                    cwd=SCRIPT_DIR,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                running[process] = {
                    "dataset": dataset,
                    "run_idx": run_idx,
                    "seed": seed,
                    "resource": cuda_id,
                    "log_path": log_path,
                    "log_handle": log_handle,
                }
                device_name = f"cuda:{cuda_id}" if cuda_id >= 0 else "cpu"
                print(
                    f"[launch {completed + len(running):02d}/{len(tasks)}] "
                    f"{dataset} run{run_idx + 1} seed={seed} -> {device_name} | "
                    f"{log_path.name}",
                    flush=True,
                )

            time.sleep(1.0)
            for process in list(running):
                return_code = process.poll()
                if return_code is None:
                    continue
                info = running.pop(process)
                info["log_handle"].close()
                free_resources.append(info["resource"])
                if return_code != 0:
                    terminate_workers(running)
                    raise RuntimeError(
                        f"{info['dataset']} run{info['run_idx'] + 1} failed "
                        f"with exit code {return_code}; see {info['log_path']}"
                    )
                completed += 1
                print(
                    f"[done {completed:02d}/{len(tasks)}] "
                    f"{info['dataset']} run{info['run_idx'] + 1}",
                    flush=True,
                )
    except BaseException:
        terminate_workers(running)
        raise

    summary = aggregate_all(args.datasets, args.seeds)
    print(f"[SOGA] all jobs complete; summary: {summary}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SOGA source-free runner for the EMBER fair graph benchmark"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--dataset",
        choices=DEFAULT_DATASETS,
        default="bailA",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=DEFAULT_DATASETS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--run-idx", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0], help=argparse.SUPPRESS)
    parser.add_argument("--cuda", type=int, default=-1, help=argparse.SUPPRESS)

    # Direct ``python run.py`` uses every visible GPU, one worker per GPU.
    # On a CPU-only host it still executes in parallel with four workers.
    parser.add_argument("--gpus", type=str, default="auto")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=1)

    # Original SOGA defaults, made explicit for the four-dataset benchmark.
    parser.add_argument("--source-epochs", type=int, default=101)
    parser.add_argument("--target-epochs", type=int, default=101)
    parser.add_argument("--source-lr", type=float, default=0.01)
    parser.add_argument("--target-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--num-layers", type=int, choices=[2, 3, 4, 5, 7, 9], default=3
    )
    parser.add_argument(
        "--gnn-model", choices=["GCN", "SAGE", "GAT"], default="GCN"
    )
    parser.add_argument("--head", type=int, default=1)
    parser.add_argument("--struct-lambda", type=float, default=1.0)
    parser.add_argument("--neigh-lambda", type=float, default=1.0)
    parser.add_argument("--num-positive-samples", type=int, default=2)
    parser.add_argument("--num-negative-samples", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=10)

    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        "--save-embeddings", dest="save_embeddings", action="store_true"
    )
    embedding_group.add_argument(
        "--no-save-embeddings", dest="save_embeddings", action="store_false"
    )
    parser.set_defaults(save_embeddings=True)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be at least 1")
    if args.cpu_workers < 1:
        parser.error("--cpu-workers must be at least 1")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be at least 1")
    if args.source_epochs < 1 or args.target_epochs < 1:
        parser.error("source and target epochs must be positive")
    if args.source_lr <= 0 or args.target_lr <= 0:
        parser.error("source and target learning rates must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if args.struct_lambda < 0 or args.neigh_lambda < 0:
        parser.error("NCE lambda values must be non-negative")
    if args.num_positive_samples != 2:
        parser.error("upstream SOGA requires --num-positive-samples=2")
    if args.num_negative_samples < 1:
        parser.error("--num-negative-samples must be positive")
    if args.print_every < 1:
        parser.error("--print-every must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    if args.aggregate_only:
        path = aggregate_all(args.datasets, args.seeds)
        print(f"[SOGA] summary regenerated: {path}")
    elif args.worker:
        run_single(args)
    else:
        launch_parallel(args)


if __name__ == "__main__":
    main()
