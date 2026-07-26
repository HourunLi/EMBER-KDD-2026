"""Strict source-domain training and target-domain evaluation for FairVGNN.

Default protocol:
    source domain: syn-2
    target domain: syn-1

The target graph is not loaded until every source run has finished.  During
target evaluation no optimizer exists, all modules are frozen/eval-only, and a
full state-dict comparison verifies that inference changed neither parameters
nor buffers.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from tqdm import tqdm

from dataset import (load_bailA_domain, load_germanA_domain,
                     load_pokec_domain, load_syn_domain)
from model import (GCN_encoder_scatter, GCN_encoder_spmm, GIN_encoder,
                   MLP_classifier, MLP_discriminator, MLP_encoder,
                   SAGE_encoder, channel_masker)
from utils import fair_metric, seed_everything


MODEL_NAMES = ("generator", "discriminator", "classifier", "encoder")


def build_models(args):
    """Create the four FairVGNN modules without creating an optimizer."""
    generator = channel_masker(args).to(args.device)
    discriminator = MLP_discriminator(args).to(args.device)
    classifier = MLP_classifier(args).to(args.device)

    if args.encoder == "MLP":
        encoder = MLP_encoder(args).to(args.device)
    elif args.encoder == "GCN":
        if args.prop == "scatter":
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
    elif args.encoder == "GIN":
        encoder = GIN_encoder(args).to(args.device)
    elif args.encoder == "SAGE":
        encoder = SAGE_encoder(args).to(args.device)
    else:
        raise ValueError(f"unsupported encoder: {args.encoder}")

    return {
        "generator": generator,
        "discriminator": discriminator,
        "classifier": classifier,
        "encoder": encoder,
    }


def build_optimizers(models, args):
    """Build optimizers only for the source-training phase."""
    generator = models["generator"]
    discriminator = models["discriminator"]
    classifier = models["classifier"]
    encoder = models["encoder"]

    optimizer_g = torch.optim.Adam(
        [dict(params=generator.weights, weight_decay=args.g_wd)],
        lr=args.g_lr)
    optimizer_d = torch.optim.Adam(
        [dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)],
        lr=args.d_lr)
    optimizer_c = torch.optim.Adam(
        [dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)],
        lr=args.c_lr)

    if args.encoder == "MLP":
        encoder_parameters = [
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)]
    elif args.encoder == "GCN":
        encoder_parameters = [
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.bias, weight_decay=args.e_wd),
        ]
    elif args.encoder == "GIN":
        encoder_parameters = [
            dict(params=encoder.conv.parameters(), weight_decay=args.e_wd)]
    else:
        encoder_parameters = [
            dict(params=encoder.conv1.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.conv2.parameters(), weight_decay=args.e_wd),
        ]
    optimizer_e = torch.optim.Adam(encoder_parameters, lr=args.e_lr)

    return {
        "generator": optimizer_g,
        "discriminator": optimizer_d,
        "classifier": optimizer_c,
        "encoder": optimizer_e,
    }


def copy_state_dict_to_cpu(models):
    return {
        model_name: {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        for model_name, model in models.items()
    }


def load_checkpoint(models, checkpoint):
    for model_name in MODEL_NAMES:
        models[model_name].load_state_dict(checkpoint[model_name])


def set_eval(models):
    for model in models.values():
        model.eval()


def set_requires_grad(model, requires_grad):
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad)


def predict_logits_and_representations(data, models, args):
    """Predict and return the corresponding encoder representations."""
    generator = models["generator"]
    encoder = models["encoder"]
    classifier = models["classifier"]

    if args.f_mask == "yes":
        feature_weights = generator()
        outputs = []
        representations = []
        for _ in range(args.K):
            mask = F.gumbel_softmax(
                feature_weights, tau=args.mask_temperature, hard=True)[:, 0]
            h = encoder(data.x * mask, data.edge_index, data.adj_norm_sp)
            outputs.append(classifier(h))
            representations.append(h)
        return (
            torch.stack(outputs).mean(dim=0),
            torch.stack(representations).mean(dim=0),
        )

    h = encoder(data.x, data.edge_index, data.adj_norm_sp)
    return classifier(h), h


def predict_logits(data, models, args):
    """Predict with the learned FairVGNN view distribution."""
    logits, _ = predict_logits_and_representations(data, models, args)
    return logits


def metrics_from_logits(logits, data, mask):
    if mask.sum().item() == 0:
        raise ValueError("cannot evaluate an empty mask")

    labels = data.y[mask].detach().cpu().numpy()
    scores = logits[mask].view(-1).detach().cpu().numpy()
    predictions = (scores > 0).astype(np.float32)
    sensitive = data.sens[mask].detach().cpu().numpy()

    if len(np.unique(labels)) < 2:
        raise ValueError("AUC requires both label classes in the evaluation set")
    parity, equality = fair_metric(predictions, labels, sensitive)
    return {
        "acc": float((predictions == labels).mean()),
        "auc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "parity": float(parity),
        "equality": float(equality),
    }


def validation_metrics(source_data, models, args):
    set_eval(models)
    with torch.no_grad():
        logits = predict_logits(source_data, models, args)
    return metrics_from_logits(logits, source_data, source_data.val_mask)


def train_one_source_run(source_data, args, run_index):
    """Train one run using source data only and keep the best source-val state."""
    run_seed = args.seed + run_index
    seed_everything(run_seed)
    models = build_models(args)
    # Match the initialization path used by the original training script.
    for model in models.values():
        model.reset_parameters()
    optimizers = build_optimizers(models, args)
    criterion_d = nn.BCELoss()
    source_sens_mask = getattr(
        source_data, "sens_mask",
        torch.ones_like(source_data.sens, dtype=torch.bool))
    if source_sens_mask.sum().item() == 0:
        raise ValueError("source domain has no observed sensitive attributes")

    best_score = -float("inf")
    best_checkpoint = None
    best_metrics = None
    best_epoch = -1

    epoch_iterator = tqdm(
        range(args.epochs),
        desc=f"source run {run_index + 1}/{args.runs}",
        disable=args.quiet)
    for epoch in epoch_iterator:
        generator = models["generator"]
        discriminator = models["discriminator"]
        classifier = models["classifier"]
        encoder = models["encoder"]

        if args.f_mask == "yes":
            generator.eval()
            feature_weights = generator()
            masks = [
                F.gumbel_softmax(
                    feature_weights, tau=args.mask_temperature, hard=False)[:, 0]
                for _ in range(args.K)
            ]

        # 1) Train the sensitive discriminator on the source domain.
        discriminator.train()
        encoder.train()
        set_requires_grad(discriminator, True)
        for _ in range(args.d_epochs):
            optimizers["discriminator"].zero_grad()
            optimizers["encoder"].zero_grad()
            if args.f_mask == "yes":
                loss_d = 0
                for mask in masks:
                    h = encoder(
                        source_data.x * mask.detach(),
                        source_data.edge_index,
                        source_data.adj_norm_sp)
                    loss_d = loss_d + criterion_d(
                        discriminator(h).view(-1)[source_sens_mask],
                        source_data.sens[source_sens_mask].float())
                loss_d = loss_d / args.K
            else:
                h = encoder(source_data.x, source_data.edge_index,
                            source_data.adj_norm_sp)
                loss_d = criterion_d(
                    discriminator(h).view(-1)[source_sens_mask],
                    source_data.sens[source_sens_mask].float())
            loss_d.backward()
            optimizers["discriminator"].step()
            optimizers["encoder"].step()

        # 2) Train the task classifier using source labels only.
        classifier.train()
        encoder.train()
        for _ in range(args.c_epochs):
            optimizers["classifier"].zero_grad()
            optimizers["encoder"].zero_grad()
            if args.f_mask == "yes":
                loss_c = 0
                for mask in masks:
                    h = encoder(
                        source_data.x * mask.detach(),
                        source_data.edge_index,
                        source_data.adj_norm_sp)
                    logits = classifier(h)
                    loss_c = loss_c + F.binary_cross_entropy_with_logits(
                        logits[source_data.train_mask],
                        source_data.y[source_data.train_mask].unsqueeze(1))
                loss_c = loss_c / args.K
            else:
                h = encoder(source_data.x, source_data.edge_index,
                            source_data.adj_norm_sp)
                logits = classifier(h)
                loss_c = F.binary_cross_entropy_with_logits(
                    logits[source_data.train_mask],
                    source_data.y[source_data.train_mask].unsqueeze(1))
            loss_c.backward()
            optimizers["encoder"].step()
            optimizers["classifier"].step()

        # 3) Train the generator/encoder to hide source sensitive attributes.
        generator.train()
        discriminator.eval()
        encoder.train()
        set_requires_grad(discriminator, False)
        for _ in range(args.g_epochs):
            optimizers["generator"].zero_grad()
            optimizers["encoder"].zero_grad()
            if args.f_mask == "yes":
                feature_weights = generator()
                loss_g = 0
                for _ in range(args.K):
                    mask = F.gumbel_softmax(
                        feature_weights,
                        tau=args.mask_temperature,
                        hard=False)[:, 0]
                    h = encoder(source_data.x * mask, source_data.edge_index,
                                source_data.adj_norm_sp)
                    sensitive_prediction = discriminator(h).view(-1)
                    loss_g = loss_g + F.mse_loss(
                        sensitive_prediction,
                        torch.full_like(sensitive_prediction, 0.5))
                    loss_g = loss_g + args.ratio * F.mse_loss(
                        mask, torch.ones_like(mask))
                loss_g = loss_g / args.K
            else:
                h = encoder(source_data.x, source_data.edge_index,
                            source_data.adj_norm_sp)
                sensitive_prediction = discriminator(h).view(-1)
                loss_g = F.mse_loss(
                    sensitive_prediction,
                    torch.full_like(sensitive_prediction, 0.5))
            loss_g.backward()
            optimizers["generator"].step()
            optimizers["encoder"].step()
        set_requires_grad(discriminator, True)

        # 4) Clamp source-trained encoder weights using learned keep rates.
        if args.weight_clip == "yes":
            if args.f_mask == "yes":
                channel_weights = torch.stack(masks).mean(dim=0).detach()
            else:
                channel_weights = torch.ones_like(source_data.x[0])
            encoder.clip_parameters(channel_weights)

        current_metrics = validation_metrics(source_data, models, args)
        current_score = (
            current_metrics["auc"] + current_metrics["f1"]
            + current_metrics["acc"]
            - args.alpha
            * (current_metrics["parity"] + current_metrics["equality"])
        )
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_metrics = current_metrics
            best_checkpoint = copy_state_dict_to_cpu(models)

        if not args.quiet and (
                epoch == 0 or (epoch + 1) % args.log_every == 0
                or epoch + 1 == args.epochs):
            epoch_iterator.write(
                "source-val "
                f"epoch={epoch + 1} auc={current_metrics['auc']:.4f} "
                f"acc={current_metrics['acc']:.4f} f1={current_metrics['f1']:.4f} "
                f"parity={current_metrics['parity']:.4f} "
                f"equality={current_metrics['equality']:.4f}")

    if best_checkpoint is None:
        raise RuntimeError("source training did not produce a checkpoint")

    del optimizers, models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "run_index": run_index,
        "seed": run_seed,
        "best_epoch": best_epoch,
        "source_val_metrics": best_metrics,
        "state_dict": best_checkpoint,
    }


def train_source_runs(source_data, args):
    """Complete every source run before the target graph is loaded."""
    checkpoints = []
    for run_index in range(args.runs):
        result = train_one_source_run(source_data, args, run_index)
        checkpoints.append(result)
        metrics = result["source_val_metrics"]
        print(
            f"source run {run_index + 1}: best_epoch={result['best_epoch'] + 1} "
            f"auc={metrics['auc']:.4f} acc={metrics['acc']:.4f} "
            f"f1={metrics['f1']:.4f} parity={metrics['parity']:.4f} "
            f"equality={metrics['equality']:.4f}")
    return checkpoints


def assert_state_unchanged(before, models):
    """Verify target inference changed no parameter or registered buffer."""
    after = copy_state_dict_to_cpu(models)
    for model_name in MODEL_NAMES:
        for key, before_value in before[model_name].items():
            if not torch.equal(before_value, after[model_name][key]):
                raise RuntimeError(
                    "target inference mutated model state: "
                    f"{model_name}.{key}")


def export_target_visualization(target_data, logits, representations,
                                checkpoint_info):
    """Export target representations and predicted-Y/S joint groups locally."""
    # Match HourunLi/AAAI-2026 zyt/visualization/export_utils.py: the
    # visualization node range is the union of target train/val/test masks.
    valid_mask = (
        target_data.train_mask.bool()
        | target_data.val_mask.bool()
        | target_data.test_mask.bool()
    )
    target_label_mask = getattr(
        target_data, "label_mask", target_data.y >= 0)
    target_sens_mask = getattr(
        target_data, "sens_mask",
        (target_data.sens == 0) | (target_data.sens == 1))
    valid_mask = valid_mask & target_label_mask.bool() & target_sens_mask.bool()
    if valid_mask.sum().item() == 0:
        raise RuntimeError("target domain has no valid nodes for visualization export")

    predicted_y = (logits.view(-1) > 0).long()
    sensitive = target_data.sens.long()
    # Reference encoding:
    #   0 -> Y=1,S=0; 1 -> Y=1,S=1;
    #   2 -> Y=0,S=0; 3 -> Y=0,S=1.
    joint_labels = torch.full_like(predicted_y, -1)
    joint_labels[(predicted_y == 1) & (sensitive == 0)] = 0
    joint_labels[(predicted_y == 1) & (sensitive == 1)] = 1
    joint_labels[(predicted_y == 0) & (sensitive == 0)] = 2
    joint_labels[(predicted_y == 0) & (sensitive == 1)] = 3

    representation_array = (
        representations[valid_mask].detach().float().cpu().numpy())
    label_array = joint_labels[valid_mask].detach().cpu().numpy().astype(np.int64)

    if representation_array.ndim != 2:
        raise RuntimeError(
            "exported representations must be rank-2, got "
            f"shape={representation_array.shape}")
    if representation_array.shape[0] != label_array.shape[0]:
        raise RuntimeError(
            "representation/label row counts differ: "
            f"{representation_array.shape[0]} vs {label_array.shape[0]}")
    if not np.isfinite(representation_array).all():
        raise RuntimeError("target representations contain NaN or Inf")
    if not np.isin(label_array, np.array([0, 1, 2, 3], dtype=np.int64)).all():
        raise RuntimeError(
            "joint target labels must be in {0, 1, 2, 3}, got "
            f"{np.unique(label_array).tolist()}")

    output_dir = Path.cwd()
    feat_path = output_dir / "feat.npz"
    labels_path = output_dir / "labels.npz"
    np.savez_compressed(feat_path, representations=representation_array)
    np.savez_compressed(labels_path, labels=label_array)
    print(
        f"target visualization exported for seed {checkpoint_info['seed']}: "
        f"{feat_path} shape={representation_array.shape}; "
        f"{labels_path} shape={label_array.shape}")


def test_target_runs(target_data, checkpoints, args):
    """Evaluate frozen checkpoints on the complete target domain."""
    results = []
    target_data = target_data.to(args.device)

    for checkpoint_info in checkpoints:
        # Model construction is followed only by loading; no target optimizer is
        # created anywhere in this phase.
        seed_everything(checkpoint_info["seed"] + 100000)
        models = build_models(args)
        load_checkpoint(models, checkpoint_info["state_dict"])
        set_eval(models)
        for model in models.values():
            set_requires_grad(model, False)

        state_before = copy_state_dict_to_cpu(models)
        with torch.inference_mode():
            logits, representations = predict_logits_and_representations(
                target_data, models, args)
            metrics = metrics_from_logits(
                logits, target_data, target_data.test_mask)
        assert_state_unchanged(state_before, models)
        export_target_visualization(
            target_data, logits, representations, checkpoint_info)

        results.append(metrics)
        print(
            f"target run {checkpoint_info['run_index'] + 1}: "
            f"auc={metrics['auc']:.4f} acc={metrics['acc']:.4f} "
            f"f1={metrics['f1']:.4f} parity={metrics['parity']:.4f} "
            f"equality={metrics['equality']:.4f} "
            "[frozen-state check: passed]")
        del models, state_before, logits, representations

    return results


def print_summary(results, source_name, target_name, encoder_name):
    print(f"======{source_name} -> {target_name} ({encoder_name})======")
    for key in ("auc", "acc", "f1", "parity", "equality"):
        values = np.array([result[key] for result in results]) * 100
        print(f"{key}: {values.mean():.2f} +/- {values.std():.2f}")


def resolve_device(gpu_index):
    """Resolve a user-selected CUDA device; use -1 to force CPU."""
    if gpu_index < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GPU {gpu_index} was requested, but CUDA is not available. "
            "Use --gpu=-1 to run on CPU.")

    gpu_count = torch.cuda.device_count()
    if gpu_index >= gpu_count:
        raise ValueError(
            f"GPU index {gpu_index} is out of range; "
            f"the server exposes {gpu_count} CUDA device(s), indexed "
            f"from 0 to {gpu_count - 1}.")

    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}")


def resolve_domain_loader(source_name, target_name):
    """Select a loader and reject cross-family source/target combinations."""
    domain_families = {
        "bailA": ("bailA_", load_bailA_domain),
        "germanA": ("germanA_", load_germanA_domain),
        "pokec": ("pokec_", load_pokec_domain),
        "syn": ("syn-", load_syn_domain),
    }
    for family_name, (name_prefix, loader) in domain_families.items():
        source_matches = source_name.startswith(name_prefix)
        target_matches = target_name.startswith(name_prefix)
        if source_matches and target_matches:
            return family_name, loader
        if source_matches != target_matches:
            raise ValueError(
                "source and target must belong to the same dataset family; "
                f"got source={source_name!r}, target={target_name!r}")
    raise ValueError(
        "unsupported domain datasets. Expected matching bailA_*, germanA_*, "
        "pokec_*, or syn-* "
        f"domains, got source={source_name!r}, target={target_name!r}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train FairVGNN on a source graph and test a frozen model on a "
            "same-family target graph"))
    parser.add_argument("--source_dataset", type=str, default="syn-2")
    parser.add_argument("--target_dataset", type=str, default="syn-1")
    parser.add_argument("--source_val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="CUDA device index, for example --gpu=2; use -1 for CPU")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--d_epochs", type=int, default=5)
    parser.add_argument("--g_epochs", type=int, default=5)
    parser.add_argument("--c_epochs", type=int, default=5)
    parser.add_argument("--g_lr", type=float, default=0.001)
    parser.add_argument("--g_wd", type=float, default=0)
    parser.add_argument("--d_lr", type=float, default=0.001)
    parser.add_argument("--d_wd", type=float, default=0)
    parser.add_argument("--c_lr", type=float, default=0.001)
    parser.add_argument("--c_wd", type=float, default=0)
    parser.add_argument("--e_lr", type=float, default=0.001)
    parser.add_argument("--e_wd", type=float, default=0)
    parser.add_argument("--prop", choices=("scatter", "spmm"), default="scatter")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--encoder", choices=("MLP", "GCN", "GIN", "SAGE"), default="GCN")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--mask_temperature", type=float, default=1.0)
    parser.add_argument("--clip_e", type=float, default=1)
    parser.add_argument("--f_mask", choices=("yes", "no"), default="yes")
    parser.add_argument("--weight_clip", choices=("yes", "no"), default="yes")
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = resolve_device(args.gpu)
    domain_family, domain_loader = resolve_domain_loader(
        args.source_dataset, args.target_dataset)
    print(f"device: {args.device}")
    print(
        f"protocol: family={domain_family}, source={args.source_dataset}, "
        f"target={args.target_dataset}; "
        "target is not loaded during source training")

    source_data, args.sens_idx, preprocessing_stats = domain_loader(
        args.source_dataset,
        role="source",
        source_val_ratio=args.source_val_ratio,
        split_seed=args.seed)
    args.num_features = source_data.x.shape[1]
    args.num_classes = 1
    print(
        f"source graph: nodes={source_data.num_nodes}, "
        f"features={args.num_features}, "
        f"train={source_data.train_mask.sum().item()}, "
        f"val={source_data.val_mask.sum().item()}")

    source_data = source_data.to(args.device)
    checkpoints = train_source_runs(source_data, args)

    # Strict phase boundary: release every raw source tensor before touching the
    # target files.  Only frozen checkpoints and source-fitted min/max values
    # remain; those are part of the learned inference artifact.
    del source_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "source phase complete: raw source tensors released; loading target "
        "for frozen inference only")

    target_data, target_sens_idx, _ = domain_loader(
        args.target_dataset,
        role="target",
        preprocessing_stats=preprocessing_stats)
    if target_sens_idx != args.sens_idx:
        raise RuntimeError("source and target sensitive-attribute indices differ")
    if target_data.x.shape[1] != args.num_features:
        raise RuntimeError("source and target feature dimensions differ")
    print(
        f"target graph: nodes={target_data.num_nodes}, "
        f"features={target_data.x.shape[1]}, "
        f"test={target_data.test_mask.sum().item()}")

    results = test_target_runs(target_data, checkpoints, args)
    print_summary(
        results, args.source_dataset, args.target_dataset, args.encoder)


if __name__ == "__main__":
    main()
