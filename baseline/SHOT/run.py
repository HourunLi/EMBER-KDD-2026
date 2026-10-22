"""SHOT (object/image_{source,target}.py) on EMBER's four domain pairs.

Run ``python run.py`` after editing DEFAULTS below. Requires the EMBER Python
environment plus torchvision (imported by SHOT's original network module).
Only the image input backbone is replaced with a node-feature MLP. SHOT's
losses, optimizer, scheduler, bottleneck, classifier and prototype updates are
unchanged. Edges are loaded by EMBER but are not inputs to this non-graph method.
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

# ===================== 可直接修改的默认参数 =====================
DEFAULTS = {
    "datasets": ["bailA", "germanA", "pokec", "syn"],
    "runs": 5,
    "gpus": [0],                 # 可用 GPU 的逻辑编号，例如 [0, 2, 3]；[-1] 为 CPU
    "seed": 1111,               # source: 1111, 2222, 3333, ...，与当前 EMBER 一致
    "seed_step": 1111,
    "target_seed_offset": 100000,  # target: 101111, 102222, ...
    "source_epochs": 20,        # SHOT object/image_source.py 的默认值
    "target_epochs": 15,        # SHOT object/image_target.py 的默认值
    "batch_size": 64,
    "eval_batch_size": 192,
    "workers": 0,
    "cpu_threads": 4,
    "hidden_dim": 256,          # 节点特征 MLP 的宽度（图像 backbone 的输入适配）
    "bottleneck": 256,
    "source_lr": 0.01,
    "target_lr": 0.01,
    "lr_decay1": 0.1,
    "lr_decay2": 1.0,
    "smooth": 0.1,
    "cls_par": 0.3,             # SHOT；设为 0 即原方法的 SHOT-IM
    "ent_par": 1.0,
    "ent": True,
    "gent": True,
    "epsilon": 1e-5,
    "threshold": 0,
    "distance": "cosine",
    "classifier": "bn",
    "layer": "wn",
    "interval": 15,             # 原方法：target 总迭代数 // interval 更新伪标签
    "save_checkpoints": True,
    "output_dir": str(HERE / "results"),
}
# 数据路径直接沿用 EMBER/dataset.py；不在这里另写一套预处理或归一化。
# ==============================================================


def runtime():
    """Keep the launcher / --help usable without importing the ML stack."""
    global np, torch, F, reference, get_dataset, fair_metric, accuracy_score, roc_auc_score, cdist
    if "reference" in globals():
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.spatial.distance import cdist
    from sklearn.metrics import accuracy_score, roc_auc_score

    sys.path.insert(0, str(ROOT))
    from EMBER.dataset import get_dataset
    from EMBER.utils import fair_metric

    # Import the original network/loss under private names. Importing the whole
    # image_target module would also require its unused OpenCV image reader.
    modules = {}
    for name in ("network", "loss"):
        spec = importlib.util.spec_from_file_location(f"_shot_{name}", HERE / "object" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    # Compile just the two self-contained upstream helper definitions, without
    # changing their bodies or importing image-only data loading dependencies.
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


def build_model(input_dim, cfg, device):
    # Node vectors cannot enter ResNet/VGG. This is the only backbone adapter;
    # no graph propagation, feature renormalization or augmentation is added.
    net_f = torch.nn.Sequential(
        torch.nn.Linear(input_dim, cfg.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        torch.nn.ReLU(),
    )
    net_f.in_features = cfg.hidden_dim  # SHOT backbone API: output feature width
    net_f.apply(reference.network.init_weights)
    net_b = reference.network.feat_bottleneck(
        feature_dim=cfg.hidden_dim, bottleneck_dim=cfg.bottleneck, type=cfg.classifier
    )
    net_c = reference.network.feat_classifier(
        class_num=2, bottleneck_dim=cfg.bottleneck, type=cfg.layer
    )
    return tuple(net.to(device) for net in (net_f, net_b, net_c))


def loader_for(x, cfg, y=None, shuffle=False):
    # Target datasets contain features and row IDs only, never true labels,
    # sensitive attributes, or masks derived from target labels.
    row_ids = torch.arange(x.shape[0])
    tensors = (x, row_ids) if y is None else (x, y)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(
        dataset,
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
            if batch[0].shape[0] > 1:
                yield batch


def cpu_state(nets):
    return [
        {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        for net in nets
    ]


def load_state(nets, states):
    for net, state in zip(nets, states):
        net.load_state_dict(state)


def set_mode(nets, train):
    for net in nets:
        net.train(train)


def predict(nets, loader, device):
    net_f, net_b, net_c = nets
    set_mode(nets, False)
    features, outputs = [], []
    with torch.no_grad():
        for x, _ in loader:
            feature = net_b(net_f(x.to(device)))
            features.append(feature.float().cpu())
            outputs.append(net_c(feature).float().cpu())
    return torch.cat(features), torch.cat(outputs)


def train_source(nets, data, cfg, device):
    """Upstream source label smoothing / SGD / source-val ACC selection."""
    valid = (data.y == 0) | (data.y == 1)
    train_mask, val_mask = data.train_mask & valid, data.val_mask & valid
    if not bool(train_mask.any()) or not bool(val_mask.any()):
        raise ValueError("Source train/validation split has no valid binary labels")
    train_loader = loader_for(data.x[train_mask], cfg, data.y[train_mask], shuffle=True)
    val_loader = loader_for(data.x[val_mask], cfg, data.y[val_mask])
    param_groups = [
        {"params": parameter, "lr": cfg.source_lr * (0.1 if index == 0 else 1.0)}
        for index, net in enumerate(nets)
        for parameter in net.parameters()
    ]
    optimizer = reference.op_copy(torch.optim.SGD(param_groups))
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
        x, y = next(batches)
        reference.lr_scheduler(optimizer, iteration, max_iter)
        outputs = net_c(net_b(net_f(x.to(device))))
        loss = criterion(outputs, y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % interval_iter == 0 or iteration == max_iter:
            _, logits = predict(nets, val_loader, device)
            # Keep SHOT's own source checkpoint criterion (overall val ACC).
            accuracy = float((logits.argmax(1) == data.y[val_mask]).float().mean())
            if accuracy >= best_accuracy:
                best_accuracy, best_state, best_iter = accuracy, cpu_state(nets), iteration
            print(
                f"source iter={iteration}/{max_iter} loss={loss.item():.6f} "
                f"val_acc={accuracy * 100:.4f}", flush=True
            )
            set_mode(nets, True)
    # Upstream saves the best source-val states and reloads them for adaptation.
    load_state(nets, best_state)
    return {"best_iteration": best_iter, "val_ACC": best_accuracy * 100}


def obtain_label(nets, loader, cfg, device):
    """Exact two-pass clustering from object/image_target.py:obtain_label.

    The upstream label-dependent accuracy printout is omitted. No true target
    labels are provided to this function; clustering uses predictions only.
    """
    all_fea, logits = predict(nets, loader, device)
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


def adapt_target(nets, target_x, cfg, device):
    """SHOT target stage: accepts only target X, never a labeled Data object."""
    net_f, net_b, net_c = nets
    net_c.eval()
    for parameter in net_c.parameters():
        parameter.requires_grad_(False)
    param_groups = []
    for net, decay in ((net_f, cfg.lr_decay1), (net_b, cfg.lr_decay2)):
        for parameter in net.parameters():
            parameter.requires_grad_(decay > 0)
            if decay > 0:
                param_groups.append({"params": parameter, "lr": cfg.target_lr * decay})
    optimizer = reference.op_copy(torch.optim.SGD(param_groups))
    train_loader = loader_for(target_x, cfg, shuffle=True)
    test_loader = loader_for(target_x, cfg)
    max_iter = cfg.target_epochs * len(train_loader)
    interval_iter = max(1, max_iter // cfg.interval)
    batches = training_batches(train_loader)
    net_f.train()
    net_b.train()
    for iter_num in range(max_iter):
        # Fetch the batch before refreshing pseudo-labels, as in upstream SHOT.
        x, row_ids = next(batches)
        if iter_num % interval_iter == 0 and cfg.cls_par > 0:
            mem_label = obtain_label(nets, test_loader, cfg, device)
            net_f.train()
            net_b.train()
        reference.lr_scheduler(optimizer, iter_num + 1, max_iter)
        outputs = net_c(net_b(net_f(x.to(device))))
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
            # Preserve upstream's periodic eval pass (including loader/RNG
            # cadence), but report only label-free entropy rather than ACC.
            _, eval_logits = predict(nets, test_loader, device)
            mean_entropy = reference.loss.Entropy(F.softmax(eval_logits, dim=1)).mean().item()
            print(
                f"target iter={iter_num + 1}/{max_iter} "
                f"loss={classifier_loss.item():.6f} entropy={mean_entropy:.6f}", flush=True
            )
            net_f.train()
            net_b.train()
    # Upstream SHOT uses the final target iterate, with no best-target selection.


def evaluate(nets, target, cfg, device):
    """Current EMBER learn.py: all(valid), per-class ACC/AUC, shared DP/EO."""
    mask = all_valid_mask(target)
    if not bool(mask.any()):
        raise ValueError("Target has no valid binary labels for final evaluation")
    _, logits = predict(nets, loader_for(target.x[mask], cfg), device)
    prob = torch.nn.Softmax(dim=1)(logits)[:, 1].numpy()
    pred = (prob > 0.5).astype(int)
    y = target.y[mask].cpu().numpy()
    sens = target.sens_labels[mask].cpu().numpy()
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
    source_id, target_id = PAIRS[dataset]
    print(f"SHOT {dataset}{source_id} -> {dataset}{target_id}, run={run_idx + 1}, "
          f"source_seed={source_seed}, target_seed={target_seed}, device={device}", flush=True)

    seed_all(source_seed)
    source = get_dataset(SimpleNamespace(dataset=dataset), source_id)
    input_dim = source.x.shape[1]
    seed_all(source_seed)
    nets = build_model(input_dim, cfg, device)
    source_info = train_source(nets, source, cfg, device)
    run_dir = Path(cfg.output_dir) / dataset / f"run_{run_idx + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_checkpoints:
        torch.save({"states": cpu_state(nets), "input_dim": input_dim,
                    "source_seed": source_seed, "config": vars(cfg)}, run_dir / "source.pt")
    # Only learned parameters cross the stage boundary. No source data/replay,
    # source features, source prototypes, or source label statistics are used.
    del source
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seed_all(target_seed)
    target = get_dataset(SimpleNamespace(dataset=dataset), target_id)
    if target.x.shape[1] != input_dim:
        raise ValueError(f"{dataset}: source/target feature dimensions differ")
    seed_all(target_seed)
    # Includes unlabeled Pokec nodes. Labels and sensitive attributes remain
    # outside the adaptation interface and are accessed only by evaluate().
    adapt_target(nets, target.x, cfg, device)
    metrics = evaluate(nets, target, cfg, device)
    if cfg.save_checkpoints:
        torch.save({"states": cpu_state(nets), "input_dim": input_dim,
                    "target_seed": target_seed, "config": vars(cfg)}, run_dir / "target.pt")
    result = {"dataset": dataset, "source_id": source_id, "target_id": target_id,
              "run": run_idx + 1, "source_seed": source_seed, "target_seed": target_seed,
              "source_selection": source_info, "metrics": metrics,
              "evaluation": "target all(valid), per-class ACC/AUC, EMBER fair_metric",
              "target_checkpoint": "final iteration", "config": vars(cfg)}
    write_json(run_dir / "metrics.json", result)
    print(json.dumps(metrics, sort_keys=True), flush=True)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def aggregate(rows, cfg):
    """Aggregate only results returned by this invocation; population std."""
    lines = [
        f"# SHOT ({cfg.runs} runs, mean +/- std, final target iteration)", "",
        "Target all(valid); ACC is the mean of per-class accuracies. "
        "AUC uses class probabilities; DP/EO use EMBER.utils.fair_metric. "
        "All metrics are percentages; variance/std use ddof=0.", "",
        "|Dataset|ACC|AUC|DP|EO|", "|---|---|---|---|---|",
    ]
    summary = {}
    for dataset in cfg.datasets:
        summary[dataset] = {}
        cells = []
        for metric in ("ACC", "AUC", "DP", "EO"):
            values = [row["metrics"][metric] for row in rows if row["dataset"] == dataset]
            # As in EMBER main.py, undefined per-run metrics propagate to the
            # summary instead of silently dropping runs or filling with zero.
            if any(value is None for value in values):
                mean = variance = std = float("nan")
            else:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                std = math.sqrt(variance)
            summary[dataset][metric] = {"mean": mean, "variance": variance,
                                        "std": std, "runs": len(values)}
            cells.append(f"{mean:.2f}+/-{std:.2f}")
        lines.append("|" + "|".join([dataset, *cells]) + "|")
    lines.extend(["", "|Dataset|Run|Source seed|Target seed|ACC|AUC|DP|EO|",
                  "|---|---|---|---|---|---|---|---|"])
    for row in rows:
        cells = ["nan" if row["metrics"][key] is None else f"{row['metrics'][key]:.4f}"
                 for key in ("ACC", "AUC", "DP", "EO")]
        lines.append("|" + "|".join(map(str, [row["dataset"], row["run"],
                     row["source_seed"], row["target_seed"], *cells])) + "|")
    output = Path(cfg.output_dir)
    write_json(output / "summary.json", {"config": vars(cfg), "summary": summary, "runs": rows})
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:len(cfg.datasets) + 6]), flush=True)
    print(f"Results: {output / 'summary.md'}", flush=True)


def launch(cfg):
    output = Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(output)
    config_path = output / "config.json"
    write_json(config_path, vars(cfg))
    slots = Queue()
    for gpu in cfg.gpus:
        slots.put(gpu)

    def task(dataset, run_idx):
        gpu = slots.get()
        try:
            run_dir = output / dataset / f"run_{run_idx + 1}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "run.log"
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--worker-config", str(config_path), "--worker-dataset", dataset,
                       "--worker-run", str(run_idx), "--worker-gpu", str(gpu)]
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = str(cfg.seed + run_idx * cfg.seed_step)
            print(f"Start {dataset} run={run_idx + 1}, gpu={gpu}, log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                         cwd=str(HERE), env=environment, check=False)
            if process.returncode:
                raise RuntimeError(f"{dataset} run={run_idx + 1} failed; see {log_path}")
            result = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            print(f"Done {dataset} run={run_idx + 1}: {result['metrics']}", flush=True)
            return result
        finally:
            slots.put(gpu)

    jobs = [(dataset, index) for dataset in cfg.datasets for index in range(cfg.runs)]
    with ThreadPoolExecutor(max_workers=len(cfg.gpus)) as pool:
        futures = [pool.submit(task, *job) for job in jobs]
        rows = [future.result() for future in futures]
    aggregate(rows, cfg)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULTS.items():
        option = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, default=default, action=argparse.BooleanOptionalAction)
        elif isinstance(default, list):
            parser.add_argument(option, nargs="+", type=type(default[0]), default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
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
    return cfg


def main():
    cfg = parse_args()
    if getattr(cfg, "worker_config", None):
        worker_cfg = SimpleNamespace(**json.loads(Path(cfg.worker_config).read_text(encoding="utf-8")))
        run_one(worker_cfg, cfg.worker_dataset, cfg.worker_run, cfg.worker_gpu)
    else:
        launch(cfg)


if __name__ == "__main__":
    main()
