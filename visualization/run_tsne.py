from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_io import (  # noqa: E402
    build_context,
    dataset_lookup,
    enabled_names,
    load_method_dataset,
    load_yaml_config,
    method_lookup,
    resolve_path,
    sample_points,
    slugify,
    split_csv_arg,
    write_coordinates_csv,
)
from plotting import compute_tsne, plot_panels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CELL-style t-SNE visualizations for SFFGNN and baseline embeddings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CURRENT_DIR / "config.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names. Defaults to enabled datasets in config.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method names. Defaults to enabled methods in config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override config output_dir.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Override sampling.max_points. Use 0 to disable sampling.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="Override tsne.perplexity.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=None,
        help="Comma-separated output formats, e.g. png,pdf.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a selected method/dataset is missing instead of skipping it.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured datasets and methods, then exit.",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check Python package availability, then exit.",
    )
    return parser.parse_args()


def check_deps() -> int:
    deps = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "scikit-learn": "sklearn",
        "PyYAML": "yaml",
    }
    missing = []
    for display_name, module_name in deps.items():
        ok = importlib.util.find_spec(module_name) is not None
        print(f"{display_name}: {'OK' if ok else 'missing'}")
        if not ok:
            missing.append(display_name)
    if missing:
        print("Install missing packages with: pip install matplotlib scikit-learn pyyaml")
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.check_deps:
        return check_deps()

    config_path = args.config.resolve()
    config = load_yaml_config(config_path)

    datasets_by_name = dataset_lookup(config)
    methods_by_name = method_lookup(config)

    selected_datasets = enabled_names(config.get("datasets", []), split_csv_arg(args.datasets))
    selected_methods = enabled_names(config.get("methods", []), split_csv_arg(args.methods))

    if args.list:
        print("Datasets:")
        for name, cfg in datasets_by_name.items():
            print(f"  - {name} ({cfg.get('title', name)})")
        print("Methods:")
        for name, cfg in methods_by_name.items():
            enabled = cfg.get("enabled", True)
            print(f"  - {name} enabled={enabled}")
        return 0

    missing_datasets = [name for name in selected_datasets if name not in datasets_by_name]
    missing_methods = [name for name in selected_methods if name not in methods_by_name]
    if missing_datasets or missing_methods:
        if missing_datasets:
            print(f"Unknown datasets: {', '.join(missing_datasets)}", file=sys.stderr)
        if missing_methods:
            print(f"Unknown methods: {', '.join(missing_methods)}", file=sys.stderr)
        return 2

    output_dir = resolve_output_dir(config, config_path, args.output_dir)

    tsne_cfg = dict(config.get("tsne") or {})
    if args.perplexity is not None:
        tsne_cfg["perplexity"] = args.perplexity

    sampling_cfg = dict(config.get("sampling") or {})
    if args.max_points is not None:
        sampling_cfg["max_points"] = args.max_points

    plot_cfg = dict(config.get("plot") or {})
    output_formats = split_csv_arg(args.formats) or list(plot_cfg.get("formats", ["png"]))

    any_plotted = False
    for dataset in selected_datasets:
        dataset_cfg = datasets_by_name[dataset]
        dataset_title = str(dataset_cfg.get("title", dataset))

        for method in selected_methods:
            method_cfg = methods_by_name[method]
            try:
                loaded = load_method_dataset(config, config_path, method_cfg, dataset)
                max_points = int(sampling_cfg.get("max_points", 5000))
                emb, labels, source_indices = sample_points(
                    loaded.embeddings,
                    loaded.labels,
                    max_points,
                    stratify=bool(sampling_cfg.get("stratify", True)),
                    random_state=int(sampling_cfg.get("random_state", tsne_cfg.get("random_state", 42))),
                )
                print(
                    f"[{method}/{dataset}] loaded {loaded.embeddings.shape[0]} points "
                    f"from {loaded.embedding_path}; plotting {labels.shape[0]} points."
                )

                coords = compute_tsne(emb, tsne_cfg)
                result_dir = output_dir / f"{slugify(method)}_{slugify(dataset)}"
                write_coordinates_csv(
                    result_dir / "coordinates.csv",
                    method,
                    dataset,
                    coords,
                    labels,
                    source_indices,
                    loaded.group_names,
                )

                panel = {"method": method, "dataset": dataset, "coords": coords, "labels": labels}
                for fmt in output_formats:
                    fmt = fmt.lstrip(".")
                    out_path = result_dir / f"tsne.{fmt}"
                    plot_panels(out_path, dataset_title, [panel], loaded.group_names, plot_cfg)
                    print(f"[saved] {out_path}")
                any_plotted = True
            except Exception as exc:
                if args.strict:
                    raise
                print(f"[skip] {method}/{dataset}: {exc}", file=sys.stderr)

    return 0 if any_plotted else 1


def resolve_output_dir(config: dict, config_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()

    config_dir = config_path.resolve().parent
    context = build_context(config, config_path, dataset="", method_name="")
    template = str(config.get("output_dir", "{visualization_root}/results"))
    return resolve_path(template, context, config_dir)


if __name__ == "__main__":
    raise SystemExit(main())
