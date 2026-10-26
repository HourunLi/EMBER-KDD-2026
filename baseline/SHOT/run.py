"""SHOT-GCN on EMBER's four pairs: edit DEFAULTS / DATASET_PARAMS, then python run.py.

Only final ACC/AUC/DP/EO are reported. The object-branch losses/updates and EMBER
loaders are reused; the backbone is a two-layer GCN with exact graph neighborhoods.
The source-label protocols and Pokec handling are configured in this file.
"""

from __future__ import annotations

import argparse
import ast
import gc
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIRS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

DEFAULTS = {
    "datasets": ["bailA", "germanA", "pokec", "syn"],
    "runs": 5,
    "gpus": [0],
    "seed": 1111,
    "seed_step": 1111,
    "target_seed_offset": 100000,  # target: 101111, 102222, ...
    "source_protocol": "full",
    "source_only": False,
    "eval_batch_size": 192,
    "workers": 0,
    "cpu_threads": 4,
    "output_dir": str(HERE / "results"),
}






DATASET_PARAMS = {
    "bailA": {
        "source_epochs": 20,
        "target_epochs": 15,
        "batch_size": 64,
        "hidden_dim": 256,
        "bottleneck": 256,
        "source_lr": 0.01,
        "source_lr_decay": 0.3,
        "target_lr": 0.01,
        "lr_decay1": 0.1,
        "lr_decay2": 1.0,
        "smooth": 0.1,
        "cls_par": 0.3,
        "ent_par": 1.0,
        "ent": True, "gent": True,
        "epsilon": 1e-5, "threshold": 0,
        "distance": "cosine", "classifier": "bn", "layer": "wn",
        "interval": 15,
    },
    "germanA": {
        "source_epochs": 20,
        "target_epochs": 15,
        "batch_size": 64,
        "hidden_dim": 256,
        "bottleneck": 256,
        "source_lr": 0.01,
        "source_lr_decay": 0.1,
        "target_lr": 0.01,
        "lr_decay1": 0.1,
        "lr_decay2": 1.0,
        "smooth": 0.1,
        "cls_par": 0.3,
        "ent_par": 1.0,
        "ent": True, "gent": True,
        "epsilon": 1e-5, "threshold": 0,
        "distance": "cosine", "classifier": "bn", "layer": "wn",
        "interval": 15,
    },
    "pokec": {
        "source_epochs": 20,
        "target_epochs": 15,
        "batch_size": 64,
        "hidden_dim": 256,
        "bottleneck": 256,
        "source_lr": 0.01,
        "source_lr_decay": 1.0,
        "target_lr": 0.01,
        "lr_decay1": 0.1,
        "lr_decay2": 1.0,
        "smooth": 0.1,
        "cls_par": 0.3,
        "ent_par": 1.0,
        "ent": True, "gent": True,
        "epsilon": 1e-5, "threshold": 0,
        "distance": "cosine", "classifier": "bn", "layer": "wn",
        "interval": 15,
    },
    "syn": {
        "source_epochs": 20,
        "target_epochs": 15,
        "batch_size": 64,
        "hidden_dim": 256,
        "bottleneck": 256,
        "source_lr": 0.01,
        "source_lr_decay": 0.1,
        "target_lr": 0.01,
        "lr_decay1": 0.1,
        "lr_decay2": 1.0,
        "smooth": 0.1,
        "cls_par": 0.3,
        "ent_par": 1.0,
        "ent": True, "gent": True,
        "epsilon": 1e-5, "threshold": 0,
        "distance": "cosine", "classifier": "bn", "layer": "wn",
        "interval": 15,
    },
}

# ==============================================================


def runtime():
    """Keep the launcher / --help usable without importing the ML stack."""
    global np, torch, F, reference, get_dataset, fair_metric, accuracy_score, roc_auc_score, cdist
    global csr_matrix, gcn_norm
    if "reference" in globals():
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.spatial.distance import cdist
    from scipy.sparse import csr_matrix
    from sklearn.metrics import accuracy_score, roc_auc_score
    from torch_geometric.nn.conv.gcn_conv import gcn_norm

    sys.path.insert(0, str(ROOT))
    from EMBER.dataset import get_dataset
    from EMBER.utils import fair_metric

    # Reuse upstream networks/losses/helpers without importing image-only IO.
    modules = {}
    for name in ("network", "loss"):
        spec = importlib.util.spec_from_file_location(f"_shot_{name}", HERE / "object" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    helper_names = {"op_copy", "lr_scheduler"}
    path = HERE / "object" / "image_target.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    if {node.name for node in definitions} != helper_names:
        raise ImportError(f"Missing SHOT optimizer helpers in {path}")
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), modules)
    reference = SimpleNamespace(**modules)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def all_valid_mask(data):
    """EMBER's all split; do not turn Pokec's missing y=-1 into class 0."""
    return (data.train_mask | data.val_mask | data.test_mask) & (
        (data.y == 0) | (data.y == 1)
    )


def source_masks(data, cfg):
    """Native full or held-out source selection, with EMBER's fixed splits."""
    valid = (data.y == 0) | (data.y == 1)
    train = valid if cfg.source_protocol == "full" else data.train_mask.bool() & valid
    selection = data.val_mask.bool() & valid
    if int(train.sum()) < 2 or not bool(selection.any()):
        raise ValueError("Source training needs >=2 valid nodes and a nonempty selection split")
    return train, selection


def prepare_data(dataset, source_seed, target_seed, source_only=False):
    """Prepare both domains offline; keep target annotations outside its graph."""
    source_id, target_id = PAIRS[dataset]
    seed_all(source_seed)
    source = get_dataset(SimpleNamespace(dataset=dataset), source_id)
    if not bool(torch.isfinite(source.x).all()):
        raise ValueError(f"{dataset}: source features contain NaN/Inf; check the raw data")
    if source_only:
        return source, None, None
    seed_all(target_seed)
    target = get_dataset(SimpleNamespace(dataset=dataset), target_id)
    if not bool(torch.isfinite(target.x).all()):
        raise ValueError(f"{dataset}: target features contain NaN/Inf; check the raw data")
    if source.x.shape[1] != target.x.shape[1]:
        raise ValueError(f"{dataset}: source/target feature dimensions differ")
    annotations = SimpleNamespace(**{
        name: getattr(target, name)
        for name in ("y", "sens_labels", "train_mask", "val_mask", "test_mask")
    })
    return source, Graph(target.x, target.edge_index), annotations


class Graph:
    """Label-free graph with global GCN normalization and fixed A_norm @ X.

    Each minibatch retains all neighbors. Caching only the first linear graph
    aggregation is exact: A_norm @ (X @ W) == (A_norm @ X) @ W.
    """

    def __init__(self, x, edge_index):
        edges, weights = gcn_norm(edge_index.cpu(), num_nodes=len(x), dtype=x.dtype)
        self.adj = csr_matrix(
            (weights.numpy(), (edges[1].numpy(), edges[0].numpy())),
            shape=(len(x), len(x)),
        )
        self.ax = torch.from_numpy(self.adj @ x.cpu().numpy())

    def batch(self, row_ids, device):
        rows = self.adj[row_ids.numpy()]
        neighbors, columns = np.unique(rows.indices, return_inverse=True)
        indices = np.stack((np.repeat(np.arange(len(row_ids)), np.diff(rows.indptr)), columns))
        adj = torch.sparse_coo_tensor(
            torch.from_numpy(indices), torch.from_numpy(rows.data),
            size=(len(row_ids), len(neighbors)), device=device,
        ).coalesce()
        return self.ax[neighbors].to(device), adj


def build_model(input_dim, cfg, device):
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Linear(input_dim, cfg.hidden_dim)
            self.conv2 = torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        def forward(self, batch):
            ax, adj = batch
            hidden = F.relu(self.conv1(ax))
            return F.relu(self.conv2(torch.sparse.mm(adj, hidden)))

    net_f = GCN()
    net_f.apply(reference.network.init_weights)
    net_b = reference.network.feat_bottleneck(
        feature_dim=cfg.hidden_dim, bottleneck_dim=cfg.bottleneck, type=cfg.classifier
    )
    net_c = reference.network.feat_classifier(
        class_num=2, bottleneck_dim=cfg.bottleneck, type=cfg.layer
    )
    return tuple(net.to(device) for net in (net_f, net_b, net_c))


def loader_for(graph, cfg, row_ids=None, shuffle=False):
    # Batches contain node IDs only; graph context never contains annotations.
    row_ids = torch.arange(len(graph.ax)) if row_ids is None else row_ids
    return torch.utils.data.DataLoader(
        row_ids,
        batch_size=cfg.batch_size if shuffle else cfg.eval_batch_size,
        shuffle=shuffle,
        num_workers=cfg.workers,
        drop_last=False,
    )


def training_batches(loader):
    """Upstream skips singleton batches because bottleneck uses BatchNorm."""
    if len(loader.dataset) < 2:
        raise ValueError("SHOT training requires at least two examples")
    while True:
        for batch in loader:
            if len(batch) > 1:
                yield batch


def cpu_state(nets):
    return [
        {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        for net in nets
    ]


def set_mode(nets, train):
    for net in nets:
        net.train(train)


def predict(nets, graph, loader, device):
    net_f, net_b, net_c = nets
    set_mode(nets, False)
    features, outputs = [], []
    with torch.no_grad():
        for row_ids in loader:
            feature = net_b(net_f(graph.batch(row_ids, device)))
            features.append(feature.float().cpu())
            outputs.append(net_c(feature).float().cpu())
    return torch.cat(features), torch.cat(outputs)


def optimizer_for(nets, learning_rate, scales):
    groups = []
    for net, scale in zip(nets, scales):
        for parameter in net.parameters():
            parameter.requires_grad_(scale > 0)
            if scale > 0:
                groups.append({"params": parameter, "lr": learning_rate * scale})
    return reference.op_copy(torch.optim.SGD(groups))


def train_source(nets, data, graph, cfg, device):
    """SHOT source training; loss only on selected nodes, context on source graph."""
    train_mask, val_mask = source_masks(data, cfg)
    train_loader = loader_for(graph, cfg, train_mask.nonzero().flatten(), shuffle=True)
    val_loader = loader_for(graph, cfg, val_mask.nonzero().flatten())
    optimizer = optimizer_for(nets, cfg.source_lr, (cfg.source_lr_decay, 1.0, 1.0))
    criterion = reference.loss.CrossEntropyLabelSmooth(
        num_classes=2, epsilon=cfg.smooth, use_gpu=device.type == "cuda"
    )
    max_iter = cfg.source_epochs * len(train_loader)
    interval_iter = max(1, max_iter // 10)
    batches = training_batches(train_loader)
    best_accuracy, best_state, best_iter = -1.0, None, 0
    set_mode(nets, True)
    net_f, net_b, net_c = nets
    for iteration in range(1, max_iter + 1):
        row_ids = next(batches)
        reference.lr_scheduler(optimizer, iteration, max_iter)
        outputs = net_c(net_b(net_f(graph.batch(row_ids, device))))
        loss = criterion(outputs, data.y[row_ids].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % interval_iter == 0 or iteration == max_iter:
            _, logits = predict(nets, graph, val_loader, device)
            accuracy = float((logits.argmax(1) == data.y[val_mask]).float().mean())
            if accuracy >= best_accuracy:
                best_accuracy, best_state, best_iter = accuracy, cpu_state(nets), iteration
            set_mode(nets, True)
    # Upstream saves the best source-selection states, then reloads them.
    for net, state in zip(nets, best_state):
        net.load_state_dict(state)
    return {
        "selection_is_holdout": cfg.source_protocol == "train_val",
        "training_nodes": int(train_mask.sum()), "selection_nodes": int(val_mask.sum()),
        "training_selection_overlap": int((train_mask & val_mask).sum()),
        "total_iterations": max_iter, "best_iteration": best_iter,
        "selection_ACC": best_accuracy * 100,
    }


def obtain_label(nets, graph, loader, cfg, device):
    """Original two-pass prototype clustering, without true target labels."""
    all_fea, logits = predict(nets, graph, loader, device)
    all_output = torch.nn.Softmax(dim=1)(logits)
    _, predict_label = torch.max(all_output, 1)
    if cfg.distance == "cosine":
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()
    k = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(k)[predict_label].sum(axis=0)
        labelset = np.where(cls_count > cfg.threshold)[0]
        if labelset.size == 0:
            raise ValueError("SHOT threshold excludes every predicted class; lower threshold")
        dd = cdist(all_fea, initc[labelset], cfg.distance)
        pred_label = dd.argmin(axis=1)
        predict_label = labelset[pred_label]
        aff = np.eye(k)[predict_label]
    return torch.from_numpy(predict_label.astype("int")).to(device)


def adapt_target(nets, graph, cfg, device):
    """SHOT adaptation receives only the target graph, never its annotations."""
    net_f, net_b, net_c = nets
    net_c.eval()
    for parameter in net_c.parameters():
        parameter.requires_grad_(False)
    optimizer = optimizer_for((net_f, net_b), cfg.target_lr, (cfg.lr_decay1, cfg.lr_decay2))
    train_loader = loader_for(graph, cfg, shuffle=True)
    test_loader = loader_for(graph, cfg)
    max_iter = cfg.target_epochs * len(train_loader)
    interval_iter = max(1, max_iter // cfg.interval)
    batches = training_batches(train_loader)
    set_mode((net_f, net_b), True)
    for iter_num in range(max_iter):
        # Fetch the batch before refreshing pseudo-labels, as in upstream SHOT.
        row_ids = next(batches)
        if iter_num % interval_iter == 0 and cfg.cls_par > 0:
            mem_label = obtain_label(nets, graph, test_loader, cfg, device)
            set_mode((net_f, net_b), True)
        reference.lr_scheduler(optimizer, iter_num + 1, max_iter)
        outputs = net_c(net_b(net_f(graph.batch(row_ids, device))))
        if cfg.cls_par > 0:
            classifier_loss = F.cross_entropy(outputs, mem_label[row_ids.to(device)]) * cfg.cls_par
        else:
            classifier_loss = outputs.new_tensor(0.0)
        if cfg.ent:
            softmax_out = torch.nn.Softmax(dim=1)(outputs)
            entropy_loss = torch.mean(reference.loss.Entropy(softmax_out))
            if cfg.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + cfg.epsilon))
            classifier_loss += entropy_loss * cfg.ent_par
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        if (iter_num + 1) % interval_iter == 0 or iter_num + 1 == max_iter:
            # Keep this upstream pass: iterating the loader consumes RNG and
            # therefore affects subsequent shuffled batches, even without logs.
            predict(nets, graph, test_loader, device)
            set_mode((net_f, net_b), True)
    # Upstream SHOT uses the final target iterate, with no best-target selection.


def metrics(logits, labels, sensitive):
    """Unchanged EMBER class-mean ACC/AUC and DP/EO, in percent."""
    prob = torch.nn.Softmax(dim=1)(logits)[:, 1].numpy()
    pred = (prob > 0.5).astype(int)
    y = labels.cpu().numpy()
    sens = sensitive.cpu().numpy()
    if not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Evaluation requires a nonempty valid-binary label subset")
    classes = np.unique(y)
    acc = np.mean([accuracy_score(y[y == cls], pred[y == cls]) for cls in classes])
    auc = np.mean([
        roc_auc_score((y == cls).astype(int), prob if cls == 1 else 1 - prob)
        for cls in classes
    ]) if len(classes) == 2 else float("nan")
    dp, eo = fair_metric(pred, y, sens)
    return {"ACC": float(acc * 100), "AUC": float(auc * 100),
            "DP": float(dp * 100), "EO": float(eo * 100)}


def run_one(cfg, dataset, run_idx, gpu):
    runtime()
    torch.set_num_threads(cfg.cpu_threads)
    if gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; set DEFAULTS['gpus'] = [-1] to run on CPU")
        torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    source_seed = cfg.seed + run_idx * cfg.seed_step
    target_seed = source_seed + cfg.target_seed_offset
    source, target_graph, annotations = prepare_data(dataset, source_seed, target_seed, cfg.source_only)
    source_graph = Graph(source.x, source.edge_index)
    input_dim = source.x.shape[1]
    seed_all(source_seed)
    nets = build_model(input_dim, cfg, device)
    source_info = train_source(nets, source, source_graph, cfg, device)
    result = {"dataset": dataset, "run": run_idx + 1, "source_seed": source_seed,
              "source_selection": source_info}
    if cfg.source_only:
        _, mask = source_masks(source, cfg)
        _, logits = predict(nets, source_graph, loader_for(source_graph, cfg, mask.nonzero().flatten()), device)
        result["metrics"] = metrics(logits, source.y[mask], source.sens_labels[mask])
        return result
    # Only learned model state crosses into target adaptation.
    for net in nets:
        for parameter in net.parameters():
            parameter.grad = None
    source_head = cpu_state((nets[2],))[0]
    del source, source_graph
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seed_all(target_seed)
    adapt_target(nets, target_graph, cfg, device)
    _, logits = predict(nets, target_graph, loader_for(target_graph, cfg), device)
    target_head = cpu_state((nets[2],))[0]
    if any(not torch.equal(source_head[key], target_head[key]) for key in source_head):
        raise RuntimeError("SHOT's frozen classifier changed during adaptation")
    # No target label is scored until all adaptation updates have completed.
    eval_mask = all_valid_mask(annotations)
    if not bool(eval_mask.any()):
        raise ValueError("Target has no valid binary labels for final evaluation")
    result.update(target_seed=target_seed,
                  metrics=metrics(logits[eval_mask], annotations.y[eval_mask], annotations.sens_labels[eval_mask]))
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def metric_summary(values):
    # EMBER's population statistics: undefined runs propagate, never get dropped.
    if any(value is None or not math.isfinite(value) for value in values):
        mean = variance = std = float("nan")
    else:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
    return {"mean": mean, "variance": variance, "std": std, "runs": len(values)}


def aggregate(rows, cfg):
    """Only the final paper table; retain per-seed values/config in one JSON."""
    lines = [
        f"# SHOT-GCN ({cfg.runs} runs)", "",
        ("Source held-out validation only. " if cfg.source_only else "Final target all(valid). ") +
        f"source_protocol={cfg.source_protocol}; %, mean +/- std (ddof=0).", "",
        "|Dataset|ACC|AUC|DP|EO|", "|---|---|---|---|---|",
    ]
    summary = {}
    for dataset in cfg.datasets:
        summary[dataset] = {
            key: metric_summary([row["metrics"][key] for row in rows if row["dataset"] == dataset])
            for key in ("ACC", "AUC", "DP", "EO")
        }
        cells = [f"{stat['mean']:.2f}+/-{stat['std']:.2f}" for stat in summary[dataset].values()]
        lines.append("|" + "|".join([dataset, *cells]) + "|")
    output = Path(cfg.output_dir)
    write_json(output / "results.json", {"config": vars(cfg), "summary": summary, "runs": rows})
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Results: {output / 'summary.md'}", flush=True)


def launch(cfg):
    # Every invocation gets a fresh directory. Never overwrite downloaded
    # results/checkpoints, or mix full and train_val experiments in one summary.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(cfg.output_dir).resolve() / f"{stamp}_{cfg.source_protocol}"
    output.mkdir(parents=True, exist_ok=False)
    cfg.output_dir = str(output)
    config_path = output / "results.json"
    write_json(config_path, {"config": vars(cfg), "runs": [], "summary": None})
    slots = Queue()
    for gpu in cfg.gpus:
        slots.put(gpu)

    def task(dataset, run_idx):
        gpu = slots.get()
        try:
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--worker-config", str(config_path), "--worker-dataset", dataset,
                       "--worker-run", str(run_idx), "--worker-gpu", str(gpu)]
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = str(cfg.seed + run_idx * cfg.seed_step)
            environment["PYTHONIOENCODING"] = "utf-8"
            print(f"Start {dataset} run={run_idx + 1}, gpu={gpu}", flush=True)
            process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                     cwd=str(HERE), env=environment, check=False)
            try:
                if process.returncode:
                    raise RuntimeError(f"worker exited with {process.returncode}")
                result = json.loads(process.stdout.rsplit("SHOT_RESULT=", 1)[1])
            except (RuntimeError, ValueError, IndexError) as error:
                log_path = output / f"{dataset}_run_{run_idx + 1}.log"
                log_path.write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
                raise RuntimeError(f"{dataset} run={run_idx + 1} failed; see {log_path}") from error
            print(f"Done {dataset} run={run_idx + 1}: {result['metrics']}", flush=True)
            return result
        finally:
            slots.put(gpu)

    jobs = [(dataset, index) for dataset in cfg.datasets for index in range(cfg.runs)]
    with ThreadPoolExecutor(max_workers=len(cfg.gpus)) as pool:
        futures = [pool.submit(task, *job) for job in jobs]
        rows = [future.result() for future in futures]
    aggregate(rows, cfg)


def dataset_config(cfg, dataset):
    """Resolve one dataset's saved parameters for validation and its worker."""
    settings = vars(cfg).copy()
    dataset_params = settings.pop("dataset_params")
    return SimpleNamespace(**{**settings, **dataset_params[dataset]})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    method_defaults = DATASET_PARAMS["bailA"]
    for key, default in {**DEFAULTS, **method_defaults}.items():
        option = "--" + key.replace("_", "-")
        # Absent CLI options must not overwrite a dataset's own parameters.
        argument_default = default if key in DEFAULTS else argparse.SUPPRESS
        if isinstance(default, bool):
            parser.add_argument(option, default=argument_default, action=argparse.BooleanOptionalAction)
        elif isinstance(default, list):
            parser.add_argument(option, nargs="+", type=type(default[0]), default=argument_default)
        else:
            parser.add_argument(option, type=type(default), default=argument_default)
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dataset", choices=PAIRS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-gpu", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_config:
        return args
    cfg = SimpleNamespace(**{key: getattr(args, key) for key in DEFAULTS})
    if not cfg.datasets or len(set(cfg.datasets)) != len(cfg.datasets) or set(cfg.datasets) - PAIRS.keys():
        parser.error(f"datasets must be a nonempty unique selection from {list(PAIRS)}")
    overrides = {key: getattr(args, key) for key in method_defaults if hasattr(args, key)}
    cfg.dataset_params = {}
    for dataset in cfg.datasets:
        if set(DATASET_PARAMS[dataset]) != set(method_defaults):
            parser.error(f"{dataset}: parameter names must match bailA's parameter names")
        cfg.dataset_params[dataset] = {**DATASET_PARAMS[dataset], **overrides}
        validate_config(dataset_config(cfg, dataset), parser)
    return cfg


def validate_config(cfg, parser):
    """Apply the original configuration checks to each resolved dataset."""
    if not cfg.gpus or len(set(cfg.gpus)) != len(cfg.gpus) or any(gpu < -1 for gpu in cfg.gpus):
        parser.error("gpus must contain unique nonnegative GPU IDs, or [-1] for CPU")
    if -1 in cfg.gpus and cfg.gpus != [-1]:
        parser.error("use [-1] alone for CPU")
    for key in ("runs", "seed_step", "source_epochs", "target_epochs", "eval_batch_size",
                "cpu_threads", "hidden_dim", "bottleneck", "interval"):
        if getattr(cfg, key) < 1:
            parser.error(f"{key} must be positive")
    if cfg.batch_size < 2 or cfg.workers < 0:
        parser.error("batch_size must be >= 2 and workers must be >= 0")
    if cfg.source_protocol not in ("full", "train_val"):
        parser.error("source_protocol must be full or train_val")
    if cfg.source_only and cfg.source_protocol != "train_val":
        parser.error("source_only requires source_protocol=train_val for independent source validation")
    if cfg.source_lr_decay <= 0:
        parser.error("source_lr_decay must be positive")
    if cfg.lr_decay1 <= 0 and cfg.lr_decay2 <= 0:
        parser.error("at least one target learning-rate multiplier must be positive")
    if cfg.cls_par < 0 or cfg.ent_par < 0 or cfg.threshold < 0:
        parser.error("cls_par, ent_par and threshold must be nonnegative")
    if cfg.cls_par == 0 and (not cfg.ent or cfg.ent_par == 0):
        parser.error("at least one of pseudo-label loss and entropy loss must be active")
    if cfg.distance not in ("cosine", "euclidean") or cfg.classifier not in ("bn", "ori") or cfg.layer not in ("wn", "linear"):
        parser.error("invalid distance, classifier or layer")
    if not 0 <= cfg.smooth <= 1 or cfg.epsilon <= 0 or min(cfg.source_lr, cfg.target_lr) <= 0:
        parser.error("require smooth in [0,1], epsilon > 0, and positive learning rates")
    for key in ("source_lr", "source_lr_decay", "target_lr", "lr_decay1", "lr_decay2",
                "smooth", "cls_par", "ent_par", "epsilon"):
        if not math.isfinite(getattr(cfg, key)):
            parser.error(f"{key} must be finite")
    if cfg.seed < 0 or cfg.seed + (cfg.runs - 1) * cfg.seed_step + cfg.target_seed_offset >= 2**32:
        parser.error("source/target seeds must fit NumPy's [0, 2**32) range")
    if cfg.target_seed_offset < 0:
        parser.error("target_seed_offset must be nonnegative")


def main():
    cfg = parse_args()
    if getattr(cfg, "worker_config", None):
        settings = json.loads(Path(cfg.worker_config).read_text(encoding="utf-8"))["config"]
        worker_cfg = dataset_config(SimpleNamespace(**settings), cfg.worker_dataset)
        result = run_one(worker_cfg, cfg.worker_dataset, cfg.worker_run, cfg.worker_gpu)
        print("SHOT_RESULT=" + json.dumps(json_safe(result), allow_nan=False), flush=True)
    else:
        launch(cfg)


if __name__ == "__main__":
    main()
