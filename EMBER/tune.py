"""Tune EMBER twice per hyperparameter combination, in parallel across GPUs.

The search space follows the method in Sections 1-3 of the EMBER paper:

* source utility/fairness coordination (beta, alpha, gamma and MMD scale),
* matched-pair squared-cosine dual-head disentanglement,
* confidence-filtered residual prototype evolution, and
* smoothed, discounted Bayesian class-prior correction.

This deterministic search combines preserved elite anchors, local searches
around every elite, focused multi-parameter mutations, and a smaller globally
balanced sample from dataset-specific ranges.
It deliberately runs exactly two training runs for every sampled combination
(``--runs_override 2``), then ranks combinations by the mean
ACC + AUC - DP - EO score.
Different trials are dispatched in parallel with one worker per GPU, so a GPU
never hosts two EMBER trials from this script at the same time.  Before every
Pokec launch, the worker also waits for a configurable amount of free VRAM and
low GPU utilization.

Examples
--------
Inspect the sampled trials without launching training::

    python tune.py --dry-run --trials 8

Use six GPUs with the round-4 dataset-specific default budgets::

    python tune.py --gpus 0 1 2 3 4 5

Tune only Pokec and resume already completed trials automatically::

    python tune.py --datasets pokec --gpus 0 1 2 3 4 5 --trials 320
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = SCRIPT_DIR / "main.py"
CONFIG_SCRIPT = SCRIPT_DIR / "config.py"
CONFIG_PATH = SCRIPT_DIR / "config" / "config.yaml"
SEARCH_ROUND = 4
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "tune_results_round4"
RUNS_PER_COMBINATION = 2

# summary3.csv durations already include both runs.  These defaults consume
# about 37.3 GPU-hours in total: roughly 6.2 ideal wall-clock hours on six
# GPUs, or about 7.8 hours after reserving 25% for Pokec admission waits,
# heterogeneous trial lengths, and the final scheduling tail.
DEFAULT_TRIALS_PER_DATASET: Dict[str, int] = {
    "bailA": 192,
    "germanA": 192,
    "pokec": 320,
    "syn": 192,
}
REFERENCE_DURATION_SECONDS: Dict[str, float] = {
    "bailA": 142.5,
    "germanA": 47.9,
    "pokec": 247.6,
    "syn": 95.6,
}

DATASET_DOMAINS: Dict[str, Tuple[str, str]] = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

# These shared values intentionally live in config.py instead of being
# repeated for every dataset in config/config.yaml.  They are listed here so
# every anchor can be resolved to a complete search parameter dictionary and
# so selected shared method parameters can still be searched explicitly.
SHARED_DEFAULTS: Dict[str, Any] = {
    "lambda_coord": 1.0,
    "group_pseudocount": 1.0,
    "prior_discount": 0.9,
}

# Fourth-round ranges follow paper/tune/summary3.csv and retain only values or
# boundary extensions supported by the two-run averages.  Every elite mode is
# explored locally; the remaining budget mixes elite-neighborhood mutations
# with a smaller globally balanced sample.
SEARCH_SPACES: Dict[str, Dict[str, Tuple[Any, ...]]] = {
    "bailA": {
        "hidden_dim": (64,),
        "n_layers": (1,),
        "lr": (0.004, 0.0045, 0.005, 0.0055),
        "dropout": (0.3, 0.35, 0.375, 0.4, 0.425, 0.45),
        "lambda_fair": (3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        "meta_lr": (0.025, 0.03, 0.035),
        "lambda_coord": (0.5, 0.625, 0.75, 0.875, 1.0),
        "source_mmd_bandwidth": (0.25, 0.5, 0.75, 1.0, 1.25),
        "adapt_epochs": (50, 75, 100, 125),
        "adapt_lr": (0.00035, 0.0004, 0.0005, 0.0006),
        "residual_inner_steps": (20, 30, 40),
        "tau_c": (0.7, 0.725, 0.75),
        "prior_confidence_threshold": (0.55, 0.6, 0.65, 0.7, 0.75),
        "proto_temp": (0.4, 0.5, 0.625, 0.75, 1.0),
        "lambda_pi": (0.0, 0.0025, 0.005, 0.01),
        "lambda_residual_l2": (0.001, 0.0015, 0.002, 0.0025, 0.003),
        "group_pseudocount": (0.25, 0.5, 0.75),
        "prior_pseudocount": (2.5, 5.0, 7.5, 10.0, 15.0),
        "prior_discount": (0.25, 0.375, 0.5, 0.625, 0.75),
    },
    "germanA": {
        "hidden_dim": (128,),
        "n_layers": (1, 2),
        "lr": (0.0075, 0.009, 0.01, 0.011, 0.012, 0.015),
        "dropout": (0.28, 0.3, 0.35, 0.4),
        "lambda_fair": (3.0, 4.0, 5.0, 6.0, 8.0),
        "meta_lr": (0.01, 0.015, 0.02, 0.025),
        "lambda_coord": (0.5, 0.625, 0.75, 1.0),
        "source_mmd_bandwidth": (0.05, 0.1, 0.15, 0.25, 0.5, 0.75),
        "adapt_epochs": (5, 10, 15, 20, 25, 30),
        "adapt_lr": (0.0001, 0.00015, 0.0002, 0.000275, 0.00035, 0.0005),
        "residual_inner_steps": (10, 20, 30, 40),
        "tau_c": (0.55, 0.6, 0.65, 0.7, 0.75),
        "prior_confidence_threshold": (0.6, 0.65, 0.7, 0.75),
        "proto_temp": (0.2, 0.25, 0.4, 0.5, 0.6, 0.75),
        "lambda_pi": (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        "lambda_residual_l2": (0.003, 0.006, 0.01, 0.02, 0.03, 0.05),
        "group_pseudocount": (0.5, 1.0, 2.0, 3.0),
        "prior_pseudocount": (0.5, 1.0, 2.0, 5.0),
        "prior_discount": (0.5, 0.625, 0.75, 0.85, 0.9, 0.95),
    },
    "pokec": {
        "hidden_dim": (64, 128),
        "n_layers": (2, 3, 4),
        "lr": (0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003,
               0.0035, 0.004, 0.0045, 0.005),
        "dropout": (0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
        "lambda_fair": (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        "meta_lr": (0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02),
        "lambda_coord": (0.5, 0.75, 1.0),
        "source_mmd_bandwidth": (0.25, 0.5, 0.75, 1.0, 1.5, 1.75, 2.0,
                                 2.25, 2.5, 3.0),
        "adapt_epochs": (10, 15, 20, 25, 35, 50, 75, 100, 125),
        "adapt_lr": (0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.00075,
                     0.001, 0.00125, 0.0015),
        "residual_inner_steps": (5, 10, 15, 20),
        "tau_c": (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
        "prior_confidence_threshold": (0.2, 0.25, 0.35, 0.4, 0.45, 0.5,
                                       0.55, 0.6),
        "proto_temp": (0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5),
        "lambda_pi": (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4),
        "lambda_residual_l2": (0.00003, 0.00005, 0.0001, 0.0002, 0.0003,
                               0.001, 0.002, 0.003, 0.005, 0.01),
        "group_pseudocount": (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
        "prior_pseudocount": (50.0, 75.0, 100.0, 125.0, 150.0, 200.0),
        "prior_discount": (0.0, 0.25, 0.5, 0.625, 0.75),
    },
    "syn": {
        # n_layers is intentionally absent: the MLP backbone ignores it.
        "hidden_dim": (256,),
        "lr": (0.002, 0.0025, 0.003, 0.0035, 0.004),
        "dropout": (0.25, 0.3, 0.35, 0.4, 0.45),
        "lambda_fair": (2.0, 4.0, 6.0, 8.0),
        "meta_lr": (0.004, 0.005, 0.006, 0.0075, 0.009),
        "lambda_coord": (0.4, 0.5, 0.6, 0.75),
        "source_mmd_bandwidth": (0.5, 0.625, 0.75, 0.875, 1.0),
        "adapt_epochs": (75, 100, 125, 150),
        "adapt_lr": (0.0001, 0.0002, 0.00035, 0.0005, 0.00075),
        "residual_inner_steps": (20, 30, 40),
        "tau_c": (0.4, 0.45, 0.5, 0.55),
        "prior_confidence_threshold": (0.4, 0.45, 0.5, 0.55, 0.6),
        "proto_temp": (0.05, 0.075, 0.1, 0.125, 0.15, 0.2),
        "lambda_pi": (0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
        "lambda_residual_l2": (0.0000003, 0.000001, 0.000003, 0.00001,
                               0.00003),
        "group_pseudocount": (0.5, 1.0, 1.5, 2.0, 3.0),
        "prior_pseudocount": (25.0, 50.0, 75.0, 100.0),
        "prior_discount": (0.85, 0.9, 0.95),
    },
}

# Primary anchors are the best two-run configurations in summary3.csv.
SEARCH_ANCHORS: Dict[str, Dict[str, Any]] = {
    "bailA": {
        "hidden_dim": 64, "n_layers": 1, "lr": 0.0045, "dropout": 0.4,
        "lambda_fair": 6.0, "meta_lr": 0.03, "lambda_coord": 0.75,
        "source_mmd_bandwidth": 1.0,
        "adapt_epochs": 75, "adapt_lr": 0.0005,
        "residual_inner_steps": 20, "tau_c": 0.7,
        "prior_confidence_threshold": 0.7, "proto_temp": 0.75,
        "lambda_pi": 0.0, "lambda_residual_l2": 0.002,
        "group_pseudocount": 0.5, "prior_pseudocount": 5.0,
        "prior_discount": 0.5,
    },
    "germanA": {
        "hidden_dim": 128, "n_layers": 1, "lr": 0.01, "dropout": 0.3,
        "lambda_fair": 4.0, "meta_lr": 0.015, "lambda_coord": 0.75,
        "source_mmd_bandwidth": 0.1,
        "adapt_epochs": 25, "adapt_lr": 0.00035,
        "residual_inner_steps": 30, "tau_c": 0.6,
        "prior_confidence_threshold": 0.7, "proto_temp": 0.6,
        "lambda_pi": 0.8, "lambda_residual_l2": 0.03,
        "group_pseudocount": 1.0, "prior_pseudocount": 2.0,
        "prior_discount": 0.75,
    },
    "pokec": {
        "hidden_dim": 64, "n_layers": 2, "lr": 0.001, "dropout": 0.2,
        "lambda_fair": 4.0, "meta_lr": 0.005, "lambda_coord": 1.0,
        "source_mmd_bandwidth": 2.0,
        "adapt_epochs": 75, "adapt_lr": 0.001,
        "residual_inner_steps": 10, "tau_c": 0.25,
        "prior_confidence_threshold": 0.45, "proto_temp": 0.75,
        "lambda_pi": 0.1, "lambda_residual_l2": 0.003,
        "group_pseudocount": 0.5, "prior_pseudocount": 100.0,
        "prior_discount": 0.5,
    },
    "syn": {
        "hidden_dim": 256, "lr": 0.003, "dropout": 0.3,
        "lambda_fair": 6.0, "meta_lr": 0.005, "lambda_coord": 0.5,
        "source_mmd_bandwidth": 0.75,
        "adapt_epochs": 100, "adapt_lr": 0.0001,
        "residual_inner_steps": 30, "tau_c": 0.45,
        "prior_confidence_threshold": 0.55, "proto_temp": 0.15,
        "lambda_pi": 1.0, "lambda_residual_l2": 0.000001,
        "group_pseudocount": 1.0, "prior_pseudocount": 50.0,
        "prior_discount": 0.9,
    },
}

# Preserve structurally distinct two-run modes.  The final Pokec entry is a
# targeted hybrid of the positive t5/t12/t13/t14 local directions.
SEARCH_EXTRA_ANCHORS: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "bailA": (
        {
            "hidden_dim": 64, "n_layers": 1, "lr": 0.0045, "dropout": 0.4,
            "lambda_fair": 6.0, "meta_lr": 0.03, "lambda_coord": 0.75,
            "source_mmd_bandwidth": 0.5,
            "adapt_epochs": 75, "adapt_lr": 0.0005,
            "residual_inner_steps": 20, "tau_c": 0.7,
            "prior_confidence_threshold": 0.7, "proto_temp": 0.75,
            "lambda_pi": 0.0, "lambda_residual_l2": 0.003,
            "group_pseudocount": 0.5, "prior_pseudocount": 5.0,
            "prior_discount": 0.5,
        },
        {
            "hidden_dim": 64, "n_layers": 1, "lr": 0.0045, "dropout": 0.4,
            "lambda_fair": 6.0, "meta_lr": 0.03, "lambda_coord": 0.75,
            "source_mmd_bandwidth": 1.0,
            "adapt_epochs": 75, "adapt_lr": 0.0005,
            "residual_inner_steps": 20, "tau_c": 0.7,
            "prior_confidence_threshold": 0.7, "proto_temp": 0.75,
            "lambda_pi": 0.005, "lambda_residual_l2": 0.003,
            "group_pseudocount": 0.5, "prior_pseudocount": 5.0,
            "prior_discount": 0.5,
        },
        {
            "hidden_dim": 64, "n_layers": 1, "lr": 0.0055, "dropout": 0.35,
            "lambda_fair": 3.0, "meta_lr": 0.03, "lambda_coord": 0.75,
            "source_mmd_bandwidth": 1.0,
            "adapt_epochs": 75, "adapt_lr": 0.0004,
            "residual_inner_steps": 30, "tau_c": 0.75,
            "prior_confidence_threshold": 0.55, "proto_temp": 1.0,
            "lambda_pi": 0.005, "lambda_residual_l2": 0.001,
            "group_pseudocount": 0.5, "prior_pseudocount": 10.0,
            "prior_discount": 0.75,
        },
    ),
    "germanA": (
        {
            "hidden_dim": 128, "n_layers": 2, "lr": 0.01, "dropout": 0.4,
            "lambda_fair": 4.0, "meta_lr": 0.02, "lambda_coord": 0.5,
            "source_mmd_bandwidth": 0.25,
            "adapt_epochs": 25, "adapt_lr": 0.0002,
            "residual_inner_steps": 40, "tau_c": 0.65,
            "prior_confidence_threshold": 0.7, "proto_temp": 0.5,
            "lambda_pi": 0.5, "lambda_residual_l2": 0.01,
            "group_pseudocount": 1.0, "prior_pseudocount": 1.0,
            "prior_discount": 0.9,
        },
        {
            "hidden_dim": 128, "n_layers": 2, "lr": 0.0075, "dropout": 0.3,
            "lambda_fair": 8.0, "meta_lr": 0.02, "lambda_coord": 0.75,
            "source_mmd_bandwidth": 0.1,
            "adapt_epochs": 25, "adapt_lr": 0.0002,
            "residual_inner_steps": 20, "tau_c": 0.75,
            "prior_confidence_threshold": 0.75, "proto_temp": 0.5,
            "lambda_pi": 0.7, "lambda_residual_l2": 0.01,
            "group_pseudocount": 3.0, "prior_pseudocount": 5.0,
            "prior_discount": 0.85,
        },
        {
            "hidden_dim": 128, "n_layers": 1, "lr": 0.012, "dropout": 0.4,
            "lambda_fair": 4.0, "meta_lr": 0.025, "lambda_coord": 0.5,
            "source_mmd_bandwidth": 0.75,
            "adapt_epochs": 5, "adapt_lr": 0.0001,
            "residual_inner_steps": 20, "tau_c": 0.75,
            "prior_confidence_threshold": 0.65, "proto_temp": 0.25,
            "lambda_pi": 0.6, "lambda_residual_l2": 0.01,
            "group_pseudocount": 1.0, "prior_pseudocount": 1.0,
            "prior_discount": 0.75,
        },
    ),
    "pokec": (
        {
            "hidden_dim": 128, "n_layers": 4, "lr": 0.004, "dropout": 0.1,
            "lambda_fair": 4.0, "meta_lr": 0.015, "lambda_coord": 0.75,
            "source_mmd_bandwidth": 0.5,
            "adapt_epochs": 25, "adapt_lr": 0.0002,
            "residual_inner_steps": 10, "tau_c": 0.45,
            "prior_confidence_threshold": 0.5, "proto_temp": 1.25,
            "lambda_pi": 0.3, "lambda_residual_l2": 0.0001,
            "group_pseudocount": 1.0, "prior_pseudocount": 75.0,
            "prior_discount": 0.5,
        },
        {
            "hidden_dim": 128, "n_layers": 4, "lr": 0.0025, "dropout": 0.1,
            "lambda_fair": 2.0, "meta_lr": 0.01, "lambda_coord": 0.75,
            "source_mmd_bandwidth": 2.5,
            "adapt_epochs": 15, "adapt_lr": 0.0002,
            "residual_inner_steps": 10, "tau_c": 0.35,
            "prior_confidence_threshold": 0.25, "proto_temp": 0.75,
            "lambda_pi": 0.4, "lambda_residual_l2": 0.0001,
            "group_pseudocount": 0.5, "prior_pseudocount": 150.0,
            "prior_discount": 0.0,
        },
        {
            "hidden_dim": 64, "n_layers": 3, "lr": 0.004, "dropout": 0.2,
            "lambda_fair": 0.5, "meta_lr": 0.01, "lambda_coord": 0.75,
            "source_mmd_bandwidth": 2.0,
            "adapt_epochs": 20, "adapt_lr": 0.0003,
            "residual_inner_steps": 20, "tau_c": 0.45,
            "prior_confidence_threshold": 0.45, "proto_temp": 1.0,
            "lambda_pi": 0.3, "lambda_residual_l2": 0.0001,
            "group_pseudocount": 0.5, "prior_pseudocount": 100.0,
            "prior_discount": 0.0,
        },
        {
            "hidden_dim": 64, "n_layers": 3, "lr": 0.004, "dropout": 0.2,
            "lambda_fair": 0.5, "meta_lr": 0.01, "lambda_coord": 0.75,
            "source_mmd_bandwidth": 2.0,
            "adapt_epochs": 20, "adapt_lr": 0.0003,
            "residual_inner_steps": 20, "tau_c": 0.45,
            "prior_confidence_threshold": 0.45, "proto_temp": 1.0,
            "lambda_pi": 0.2, "lambda_residual_l2": 0.0001,
            "group_pseudocount": 0.5, "prior_pseudocount": 75.0,
            "prior_discount": 0.5,
        },
    ),
    "syn": (
        {
            "hidden_dim": 256, "lr": 0.003, "dropout": 0.4,
            "lambda_fair": 2.0, "meta_lr": 0.0075, "lambda_coord": 0.5,
            "source_mmd_bandwidth": 0.75,
            "adapt_epochs": 125, "adapt_lr": 0.0005,
            "residual_inner_steps": 30, "tau_c": 0.5,
            "prior_confidence_threshold": 0.4, "proto_temp": 0.05,
            "lambda_pi": 0.1, "lambda_residual_l2": 0.00001,
            "group_pseudocount": 3.0, "prior_pseudocount": 75.0,
            "prior_discount": 0.95,
        },
        {
            "hidden_dim": 256, "lr": 0.002, "dropout": 0.3,
            "lambda_fair": 2.0, "meta_lr": 0.005, "lambda_coord": 0.5,
            "source_mmd_bandwidth": 1.0,
            "adapt_epochs": 75, "adapt_lr": 0.0002,
            "residual_inner_steps": 40, "tau_c": 0.5,
            "prior_confidence_threshold": 0.5, "proto_temp": 0.1,
            "lambda_pi": 0.75, "lambda_residual_l2": 0.00001,
            "group_pseudocount": 2.0, "prior_pseudocount": 50.0,
            "prior_discount": 0.9,
        },
        {
            "hidden_dim": 256, "lr": 0.002, "dropout": 0.4,
            "lambda_fair": 2.0, "meta_lr": 0.005, "lambda_coord": 0.75,
            "source_mmd_bandwidth": 0.75,
            "adapt_epochs": 75, "adapt_lr": 0.0005,
            "residual_inner_steps": 40, "tau_c": 0.4,
            "prior_confidence_threshold": 0.45, "proto_temp": 0.15,
            "lambda_pi": 0.1, "lambda_residual_l2": 0.000003,
            "group_pseudocount": 2.0, "prior_pseudocount": 75.0,
            "prior_discount": 0.9,
        },
    ),
}

# Local candidates are generated around every elite anchor.  The order matters
# when a per-anchor quota is smaller than the number of searchable factors.
LOCAL_SEARCH_KEYS: Dict[str, Tuple[str, ...]] = {
    "bailA": (
        "lambda_residual_l2", "source_mmd_bandwidth", "lambda_pi",
        "proto_temp", "prior_confidence_threshold", "lr", "dropout",
        "lambda_fair", "meta_lr", "lambda_coord",
        "adapt_epochs", "adapt_lr", "residual_inner_steps", "tau_c",
        "prior_discount",
    ),
    "germanA": (
        "lambda_pi", "lambda_fair", "adapt_lr", "proto_temp",
        "prior_discount", "lambda_coord", "source_mmd_bandwidth",
        "residual_inner_steps", "adapt_epochs", "dropout", "n_layers", "lr",
        "meta_lr", "tau_c",
        "prior_confidence_threshold", "lambda_residual_l2",
        "group_pseudocount",
    ),
    "pokec": (
        "adapt_lr", "lambda_pi", "prior_pseudocount", "prior_discount",
        "dropout", "lambda_fair", "source_mmd_bandwidth", "adapt_epochs",
        "lr", "meta_lr", "lambda_coord",
        "residual_inner_steps", "tau_c", "prior_confidence_threshold",
        "lambda_residual_l2", "proto_temp", "group_pseudocount",
    ),
    "syn": (
        "lambda_coord", "lr", "meta_lr", "dropout", "proto_temp",
        "lambda_fair", "source_mmd_bandwidth",
        "residual_inner_steps", "adapt_epochs", "adapt_lr", "tau_c",
        "prior_confidence_threshold", "lambda_pi", "lambda_residual_l2",
        "group_pseudocount", "prior_pseudocount", "prior_discount",
    ),
}
LOCAL_TRIALS_PER_ANCHOR: Dict[str, int] = {
    "bailA": 16,
    "germanA": 18,
    "pokec": 16,
    "syn": 18,
}
ELITE_RANDOM_FRACTION: Dict[str, float] = {
    "bailA": 0.75,
    "germanA": 0.75,
    "pokec": 0.85,
    "syn": 0.75,
}
ELITE_MUTATION_BOUNDS: Dict[str, Tuple[int, int]] = {
    "bailA": (2, 5),
    "germanA": (2, 6),
    "pokec": (2, 7),
    "syn": (2, 6),
}
ELITE_NEIGHBOR_DEPTH: Dict[str, int] = {
    "bailA": 3,
    "germanA": 3,
    "pokec": 3,
    "syn": 3,
}
ELITE_FIXED_KEYS: Dict[str, Tuple[str, ...]] = {
    # Keep the discovered Pokec architecture modes intact during focused
    # mutations; the global segment still explores all architecture pairs.
    "pokec": ("hidden_dim", "n_layers"),
}


PRINT_LOCK = threading.Lock()
FAILURE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Trial:
    dataset: str
    trial_id: int
    parameters: Dict[str, Any]
    strategy: str
    anchor_index: int | None = None

    @property
    def name(self) -> str:
        return f"trial_{self.trial_id:04d}"


def print_status(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def load_base_config() -> Dict[str, Dict[str, Any]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dataset mapping in {CONFIG_PATH}")
    return payload


def _stable_dataset_seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(dataset.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big")


def _signature(parameters: Mapping[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _resolved_search_anchors(
    dataset: str,
    space: Mapping[str, Sequence[Any]],
    base_config: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve and validate the primary and secondary anchors for a dataset."""
    effective_base = dict(SHARED_DEFAULTS)
    effective_base.update(base_config[dataset])
    anchor_overrides = (
        SEARCH_ANCHORS[dataset],
        *SEARCH_EXTRA_ANCHORS.get(dataset, ()),
    )

    anchors: List[Dict[str, Any]] = []
    seen = set()
    for anchor_index, overrides in enumerate(anchor_overrides):
        unknown = sorted(set(overrides) - set(space))
        if unknown:
            raise KeyError(
                f"Anchor {anchor_index} for {dataset} contains parameters outside "
                f"its search space: {', '.join(unknown)}"
            )

        resolved = dict(effective_base)
        resolved.update(overrides)
        missing = [key for key in space if key not in resolved]
        if missing:
            raise KeyError(
                f"No anchor/default value for {dataset}: "
                f"{', '.join(sorted(missing))}"
            )
        parameters: Dict[str, Any] = {}
        outside: Dict[str, Any] = {}
        for key, choices in space.items():
            value = resolved[key]
            matching_choices = [choice for choice in choices if choice == value]
            if not matching_choices:
                outside[key] = value
                continue
            # Normalize 1 versus 1.0 and similar numeric aliases to the exact
            # object stored in the search space before computing signatures.
            parameters[key] = matching_choices[0]
        if outside:
            rendered = ", ".join(
                f"{key}={value!r}" for key, value in sorted(outside.items())
            )
            raise ValueError(
                f"Anchor {anchor_index} for {dataset} lies outside the "
                f"fourth-round ranges: {rendered}"
            )

        signature = _signature(parameters)
        if signature in seen:
            raise ValueError(f"Duplicate search anchor for {dataset}: {anchor_index}")
        seen.add(signature)
        anchors.append(parameters)
    return anchors


def _balanced_parameter_columns(
    space: Mapping[str, Sequence[Any]],
    count: int,
    rng: random.Random,
) -> Dict[str, List[Any]]:
    """Give every categorical value near-equal representation in the sample."""
    columns: Dict[str, List[Any]] = {}
    for key, choices in space.items():
        if not choices:
            raise ValueError(f"Search range for {key} is empty")
        repeats = math.ceil(count / len(choices)) if count else 0
        values = list(choices) * repeats
        rng.shuffle(values)
        columns[key] = values[:count]
    return columns


def _local_anchor_candidates(
    dataset: str,
    space: Mapping[str, Sequence[Any]],
    anchor: Mapping[str, Any],
) -> Iterable[Dict[str, Any]]:
    """Yield one-factor perturbations in nearest-to-anchor order."""
    alternatives: Dict[str, List[Any]] = {}
    for key in LOCAL_SEARCH_KEYS[dataset]:
        alternatives[key] = _ordered_alternatives(space[key], anchor[key])

    max_depth = max((len(values) for values in alternatives.values()), default=0)
    for depth in range(max_depth):
        for key in LOCAL_SEARCH_KEYS[dataset]:
            values = alternatives[key]
            if depth >= len(values):
                continue
            candidate = dict(anchor)
            candidate[key] = values[depth]
            yield candidate


def _ordered_alternatives(
    choices: Sequence[Any],
    anchor_value: Any,
) -> List[Any]:
    """Return non-anchor choices from nearest to farthest."""
    values = [value for value in choices if value != anchor_value]
    try:
        values.sort(key=lambda value: abs(float(value) - float(anchor_value)))
    except (TypeError, ValueError):
        values.sort(key=repr)
    return values


def _elite_mutation_candidate(
    dataset: str,
    space: Mapping[str, Sequence[Any]],
    anchors: Sequence[Mapping[str, Any]],
    rng: random.Random,
) -> Tuple[Dict[str, Any], int]:
    """Jointly mutate a few nearby values while preserving an elite mode."""
    weights = [2.0] + [1.0] * (len(anchors) - 1)
    anchor_index = rng.choices(range(len(anchors)), weights=weights, k=1)[0]
    anchor = anchors[anchor_index]
    fixed_keys = set(ELITE_FIXED_KEYS.get(dataset, ()))
    mutable_keys = [
        key
        for key, choices in space.items()
        if len(choices) > 1 and key not in fixed_keys
    ]
    lower, upper = ELITE_MUTATION_BOUNDS[dataset]
    lower = min(lower, len(mutable_keys))
    upper = min(upper, len(mutable_keys))
    mutation_count = rng.randint(lower, upper)
    selected_keys = rng.sample(mutable_keys, mutation_count)

    parameters = dict(anchor)
    neighbor_depth = ELITE_NEIGHBOR_DEPTH[dataset]
    for key in selected_keys:
        alternatives = _ordered_alternatives(space[key], anchor[key])
        parameters[key] = rng.choice(alternatives[:neighbor_depth])
    return parameters, anchor_index


def make_trials(
    dataset: str,
    count: int,
    sampler_seed: int,
    base_config: Mapping[str, Mapping[str, Any]],
) -> List[Trial]:
    """Preserve elites, search every elite locally, then sample interactions."""
    space = SEARCH_SPACES[dataset]
    anchors = _resolved_search_anchors(dataset, space, base_config)
    trials: List[Trial] = []
    seen = set()
    for anchor_index, anchor in enumerate(anchors):
        if len(trials) >= count:
            return trials
        seen.add(_signature(anchor))
        trials.append(
            Trial(
                dataset=dataset,
                trial_id=len(trials),
                parameters=anchor,
                strategy=("primary_anchor" if anchor_index == 0 else "secondary_anchor"),
                anchor_index=anchor_index,
            )
        )

    local_quota = LOCAL_TRIALS_PER_ANCHOR[dataset]
    local_generators = [
        iter(_local_anchor_candidates(dataset, space, anchor))
        for anchor in anchors
    ]
    local_counts = [0] * len(anchors)
    active = [True] * len(anchors)
    while len(trials) < count and any(active):
        progressed = False
        for anchor_index, generator in enumerate(local_generators):
            if len(trials) >= count:
                break
            if not active[anchor_index] or local_counts[anchor_index] >= local_quota:
                active[anchor_index] = False
                continue
            while True:
                try:
                    candidate = next(generator)
                except StopIteration:
                    active[anchor_index] = False
                    break
                signature = _signature(candidate)
                if signature in seen:
                    continue
                seen.add(signature)
                trials.append(
                    Trial(
                        dataset=dataset,
                        trial_id=len(trials),
                        parameters=candidate,
                        strategy="local",
                        anchor_index=anchor_index,
                    )
                )
                local_counts[anchor_index] += 1
                progressed = True
                break
        if not progressed:
            break

    remaining = count - len(trials)
    if remaining == 0:
        return trials

    rng = random.Random(_stable_dataset_seed(sampler_seed, dataset))
    elite_target = round(remaining * ELITE_RANDOM_FRACTION[dataset])
    attempts = 0
    elite_added = 0
    while elite_added < elite_target and len(trials) < count:
        parameters, anchor_index = _elite_mutation_candidate(
            dataset, space, anchors, rng
        )
        signature = _signature(parameters)
        attempts += 1
        if signature in seen:
            if attempts > max(count * 1000, 10000):
                raise RuntimeError(
                    f"Could not sample {elite_target} elite mutations for {dataset}"
                )
            continue
        seen.add(signature)
        trials.append(
            Trial(
                dataset=dataset,
                trial_id=len(trials),
                parameters=parameters,
                strategy="elite_random",
                anchor_index=anchor_index,
            )
        )
        elite_added += 1

    global_target = count - len(trials)
    columns = _balanced_parameter_columns(space, global_target, rng)
    attempts = 0
    index = 0
    while len(trials) < count:
        if index < global_target:
            parameters = {key: columns[key][index] for key in space}
            index += 1
        else:
            parameters = {key: rng.choice(choices) for key, choices in space.items()}
        signature = _signature(parameters)
        attempts += 1
        if signature in seen:
            if attempts > max(count * 1000, 10000):
                raise RuntimeError(
                    f"Could not sample {count} unique trials for {dataset}"
                )
            continue
        seen.add(signature)
        trials.append(
            Trial(
                dataset=dataset,
                trial_id=len(trials),
                parameters=parameters,
                strategy="global_random",
            )
        )
    return trials


def raw_result_path(results_dir: Path, trial: Trial) -> Path:
    return results_dir / "raw" / trial.dataset / f"{trial.name}.json"


def log_path(results_dir: Path, trial: Trial) -> Path:
    return results_dir / "logs" / trial.dataset / f"{trial.name}.log"


def stderr_path(results_dir: Path, trial: Trial) -> Path:
    return results_dir / "logs" / trial.dataset / f"{trial.name}.stderr.log"


def is_complete_result(
    path: Path,
    expected_parameters: Mapping[str, Any] | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as result_file:
            payload = json.load(result_file)
        target = payload["metrics"]["target_after"]
        for key in ("acc", "auc", "dp", "eo"):
            values = target.get(key)
            if (
                not isinstance(values, list)
                or len(values) != RUNS_PER_COMBINATION
            ):
                return False
            if not all(math.isfinite(float(value)) for value in values):
                return False
        if expected_parameters is not None:
            actual_parameters = payload.get("tuning", {}).get("parameters")
            if not isinstance(actual_parameters, dict):
                return False
            if _signature(actual_parameters) != _signature(expected_parameters):
                return False
            actual_runs = payload.get("tuning", {}).get("runs_per_combination")
            if actual_runs != RUNS_PER_COMBINATION:
                return False
        return True
    except (OSError, OverflowError, ValueError, KeyError, TypeError):
        return False


def _override_text(key: str, value: Any) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    return f"{key}={rendered}"


@lru_cache(maxsize=1)
def _main_cli_options() -> frozenset[str]:
    """Read the options actually registered by the colocated config.py.

    Some deployed EMBER copies predate the checkpoint-disable arguments.  A
    static AST inspection keeps tune.py compatible with both parser versions
    without importing config.py, which would immediately parse tune.py's own
    command line and initialize the training runtime.
    """
    source = CONFIG_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CONFIG_SCRIPT))
    options = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }
    return frozenset(options)


def _checkpoint_cli_arguments(options: Iterable[str]) -> List[str]:
    """Return only checkpoint-safety switches supported by main.py's parser."""
    supported = set(options)
    arguments: List[str] = []
    for aliases in (
        ("--no-checkpoint", "--no_checkpoint"),
        ("--disable-checkpoint-save", "--disable_checkpoint_save"),
    ):
        selected = next((option for option in aliases if option in supported), None)
        if selected is not None:
            arguments.append(selected)
    return arguments


def query_gpu_status() -> Dict[int, Dict[str, int]]:
    """Return current free/total memory and utilization for every GPU.

    The check intentionally uses ``nvidia-smi`` rather than importing torch:
    it runs before the child process initializes CUDA and therefore observes
    memory held by other users or stale jobs.
    """
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "nvidia-smi is required for Pokec GPU admission checks; "
            "use --gpus -1 for CPU or load the NVIDIA driver environment"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"nvidia-smi failed while checking GPU memory: {detail}") from error

    statuses: Dict[int, Dict[str, int]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            index, free_mb, total_mb, utilization = (int(field) for field in fields)
        except ValueError as error:
            raise RuntimeError(f"Unexpected nvidia-smi output: {line!r}") from error
        statuses[index] = {
            "free_mb": free_mb,
            "total_mb": total_mb,
            "utilization": utilization,
        }
    if not statuses:
        raise RuntimeError("nvidia-smi returned no GPU status rows")
    return statuses


def wait_for_pokec_gpu(
    args: argparse.Namespace,
    device_id: int,
) -> Dict[str, int] | None:
    """Wait until a Pokec worker has a safely idle GPU before launching.

    A 44-GB card with only a few hundred MB free can pass a superficial
    device-id check but still fail during the first backward pass.  Requiring
    a configurable free-memory reserve and low utilization prevents that
    situation and also avoids competing with another active process.
    """
    if device_id < 0:
        return None

    deadline = (
        time.monotonic() + args.gpu_wait_timeout
        if args.gpu_wait_timeout > 0
        else None
    )
    last_report = 0.0
    while True:
        status = query_gpu_status().get(device_id)
        if status is None:
            raise RuntimeError(f"GPU {device_id} was not reported by nvidia-smi")
        if status["total_mb"] < args.pokec_min_free_memory_mb:
            raise RuntimeError(
                f"GPU {device_id} has only {status['total_mb']} MiB total VRAM, "
                f"below the Pokec minimum-free threshold "
                f"{args.pokec_min_free_memory_mb} MiB"
            )
        if (
            status["free_mb"] >= args.pokec_min_free_memory_mb
            and status["utilization"] <= args.pokec_max_gpu_utilization
        ):
            print_status(
                f"[GPU {device_id}] Pokec admission granted: "
                f"free={status['free_mb']} MiB, util={status['utilization']}%"
            )
            return status

        now = time.monotonic()
        if now - last_report >= max(args.gpu_poll_seconds, 1):
            print_status(
                f"[GPU {device_id}] waiting for Pokec: "
                f"free={status['free_mb']} MiB (need >= "
                f"{args.pokec_min_free_memory_mb}), "
                f"util={status['utilization']}% (need <= "
                f"{args.pokec_max_gpu_utilization}%)"
            )
            last_report = now
        if deadline is not None and now >= deadline:
            raise TimeoutError(
                f"GPU {device_id} did not reach the Pokec memory threshold "
                f"within {args.gpu_wait_timeout:.0f}s"
            )
        time.sleep(args.gpu_poll_seconds)


def build_command(
    args: argparse.Namespace,
    device_id: int,
    trial: Trial,
    result_path: Path,
    trial_log_path: Path,
) -> List[str]:
    inid, outid = DATASET_DOMAINS[trial.dataset]
    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--dataset", trial.dataset,
        "--inid", inid,
        "--outid", outid,
        "--device_id", str(device_id),
        "--seed", str(args.seed),
        "--runs_override", str(RUNS_PER_COMBINATION),
        "--disable_embedding_export",
        "--log_path", str(trial_log_path),
        "--result_path", str(result_path),
    ]
    # With no explicit target seed, runner.py uses seed + run_idx * 1111, so
    # the two target-adaptation runs no longer reset to the same random seed.
    if args.target_seed_offset is not None:
        command.extend(
            ("--target_seed", str(args.seed + args.target_seed_offset))
        )
    command.extend(_checkpoint_cli_arguments(_main_cli_options()))
    for key, value in trial.parameters.items():
        command.extend(("--override", _override_text(key, value)))
    return command


def _enrich_result(
    path: Path,
    trial: Trial,
    duration_seconds: float,
    device_id: int,
    gpu_status: Mapping[str, int] | None = None,
) -> None:
    with path.open("r", encoding="utf-8") as result_file:
        payload = json.load(result_file)
    payload["tuning"] = {
        "trial_id": trial.trial_id,
        "parameters": trial.parameters,
        "search_strategy": trial.strategy,
        "anchor_index": trial.anchor_index,
        "duration_seconds": duration_seconds,
        "device_id": device_id,
        "runs_per_combination": RUNS_PER_COMBINATION,
    }
    if gpu_status is not None:
        payload["tuning"]["gpu_admission"] = dict(gpu_status)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    temporary.replace(path)


def run_trial(
    args: argparse.Namespace,
    device_id: int,
    trial: Trial,
) -> None:
    result = raw_result_path(args.results_dir, trial)
    trial_log = log_path(args.results_dir, trial)
    trial_stderr = stderr_path(args.results_dir, trial)
    result.parent.mkdir(parents=True, exist_ok=True)
    trial_log.parent.mkdir(parents=True, exist_ok=True)

    if not args.rerun and is_complete_result(result, trial.parameters):
        print_status(f"[resume] {trial.dataset} {trial.name} already complete")
        return
    if args.rerun and result.exists():
        result.unlink()

    gpu_status = wait_for_pokec_gpu(args, device_id) if trial.dataset == "pokec" else None
    command = build_command(args, device_id, trial, result, trial_log)
    print_status(
        f"[device {device_id}] start {trial.dataset} {trial.name} "
        f"[{trial.strategy}]"
    )
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    duration = time.monotonic() - started
    if completed.stderr:
        trial_stderr.write_text(completed.stderr, encoding="utf-8")
    elif trial_stderr.exists():
        trial_stderr.unlink()

    if completed.returncode != 0:
        raise RuntimeError(
            f"{trial.dataset} {trial.name} exited with {completed.returncode}; "
            f"see {trial_log} and {trial_stderr}"
        )
    if not is_complete_result(result):
        raise RuntimeError(f"{trial.dataset} {trial.name} did not produce {result}")
    _enrich_result(result, trial, duration, device_id, gpu_status)
    if not is_complete_result(result, trial.parameters):
        raise RuntimeError(f"{trial.dataset} {trial.name} result metadata mismatch")
    print_status(
        f"[device {device_id}] done  {trial.dataset} {trial.name} "
        f"({duration / 60.0:.1f} min)"
    )


def worker(
    args: argparse.Namespace,
    device_id: int,
    tasks: queue.Queue,
    failures: List[str],
    stop_event: threading.Event,
) -> None:
    while True:
        trial = tasks.get()
        try:
            if trial is None:
                return
            if stop_event.is_set():
                continue
            run_trial(args, device_id, trial)
        except Exception as error:  # keep other trials resumable by default
            message = str(error)
            with FAILURE_LOCK:
                failures.append(message)
            print_status(f"[device {device_id}] ERROR: {message}")
            if args.fail_fast:
                stop_event.set()
        finally:
            tasks.task_done()


def detect_gpus() -> List[int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    detected = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            detected.append(int(line))
    return detected


def resolve_devices(tokens: Sequence[str]) -> List[int]:
    if "auto" in tokens:
        if len(tokens) != 1:
            raise ValueError("--gpus auto cannot be combined with explicit GPU ids")
        devices = detect_gpus()
        return devices or [-1]

    devices = [int(token) for token in tokens]
    if not devices:
        raise ValueError("At least one GPU id is required; use -1 for CPU")
    if len(set(devices)) != len(devices):
        raise ValueError("--gpus must not contain duplicate ids")
    if any(device < -1 for device in devices):
        raise ValueError("GPU ids must be non-negative, or -1 for CPU")
    if -1 in devices and len(devices) != 1:
        raise ValueError("CPU id -1 cannot be combined with GPU ids")
    return devices


def filter_devices_for_pokec(
    args: argparse.Namespace,
    devices: Sequence[int],
) -> List[int]:
    """Exclude GPUs whose total VRAM cannot satisfy the Pokec reserve.

    A mixed cluster may contain a small GPU and a 44-GB GPU.  Keeping the
    small device in the worker pool would let it repeatedly pick Pokec jobs
    and fail before the larger device can consume them.  We therefore remove
    such devices for a run that includes Pokec; other datasets can still be
    tuned separately on the small device.
    """
    if "pokec" not in args.datasets or all(device_id < 0 for device_id in devices):
        return list(devices)

    statuses = query_gpu_status()
    eligible = [
        device_id
        for device_id in devices
        if device_id in statuses
        and statuses[device_id]["total_mb"] >= args.pokec_min_free_memory_mb
    ]
    excluded = [device_id for device_id in devices if device_id not in eligible]
    if excluded:
        print_status(
            "Excluding GPUs from this Pokec run because total VRAM is below "
            f"{args.pokec_min_free_memory_mb} MiB: {excluded}"
        )
    if not eligible:
        raise RuntimeError(
            "No selected GPU has enough total VRAM for Pokec; "
            f"need at least {args.pokec_min_free_memory_mb} MiB"
        )
    return eligible


def resolve_trial_counts(args: argparse.Namespace) -> Dict[str, int]:
    """Resolve global or dataset-specific CLI budgets against round-4 defaults."""
    counts = {
        dataset: DEFAULT_TRIALS_PER_DATASET[dataset]
        for dataset in args.datasets
    }
    if args.trials is not None:
        if args.trials < 1:
            raise ValueError("--trials must be at least 1")
        return {dataset: args.trials for dataset in args.datasets}

    seen = set()
    for item in args.trials_per_dataset or ():
        dataset, separator, raw_count = item.partition("=")
        dataset = dataset.strip()
        if not separator or not dataset or not raw_count.strip():
            raise ValueError(
                f"Invalid --trials-per-dataset value {item!r}; "
                "expected DATASET=COUNT"
            )
        if dataset not in DATASET_DOMAINS:
            raise ValueError(f"Unknown dataset in trial budget: {dataset}")
        if dataset not in args.datasets:
            raise ValueError(
                f"Trial budget supplied for unselected dataset: {dataset}"
            )
        if dataset in seen:
            raise ValueError(f"Duplicate trial budget for dataset: {dataset}")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(f"Invalid trial count for {dataset}: {raw_count}") from error
        if count < 1:
            raise ValueError(f"Trial count for {dataset} must be at least 1")
        seen.add(dataset)
        counts[dataset] = count
    return counts


def estimated_gpu_hours(trial_counts: Mapping[str, int]) -> float:
    """Estimate GPU-hours from summary3 two-run mean durations."""
    total_seconds = math.fsum(
        trial_counts[dataset] * REFERENCE_DURATION_SECONDS[dataset]
        for dataset in trial_counts
    )
    return total_seconds / 3600.0


def write_manifest(
    args: argparse.Namespace,
    trials: Sequence[Trial],
    devices: Sequence[int],
) -> Path:
    path = args.results_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    strategies = (
        "primary_anchor",
        "secondary_anchor",
        "local",
        "elite_random",
        "global_random",
    )
    strategy_counts = {
        dataset: {
            strategy: sum(
                trial.dataset == dataset and trial.strategy == strategy
                for trial in trials
            )
            for strategy in strategies
        }
        for dataset in args.datasets
    }
    anchor_counts = {
        dataset: (
            strategy_counts[dataset]["primary_anchor"]
            + strategy_counts[dataset]["secondary_anchor"]
        )
        for dataset in args.datasets
    }
    local_counts = {
        dataset: strategy_counts[dataset]["local"]
        for dataset in args.datasets
    }
    elite_random_counts = {
        dataset: strategy_counts[dataset]["elite_random"]
        for dataset in args.datasets
    }
    global_random_counts = {
        dataset: strategy_counts[dataset]["global_random"]
        for dataset in args.datasets
    }
    gpu_hours = estimated_gpu_hours(args.trial_counts)
    gpu_device_count = sum(device_id >= 0 for device_id in devices)
    payload = {
        "search_round": SEARCH_ROUND,
        "search": "multi_anchor_local_plus_elite_mutation_plus_global_balanced",
        "datasets": args.datasets,
        "trials_per_dataset": args.trial_counts,
        "anchor_trials_per_dataset": anchor_counts,
        "local_trials_per_dataset": local_counts,
        "elite_random_trials_per_dataset": elite_random_counts,
        "global_random_trials_per_dataset": global_random_counts,
        "strategy_counts_per_dataset": strategy_counts,
        "runs_per_combination": RUNS_PER_COMBINATION,
        "training_seed": args.seed,
        "sampler_seed": args.sampler_seed,
        "target_seed_offset": args.target_seed_offset,
        "target_seed_policy": (
            "runner_run_index_offset"
            if args.target_seed_offset is None
            else "fixed_offset"
        ),
        "devices": list(devices),
        "estimated_budget_from_summary3": {
            "gpu_hours": gpu_hours,
            "ideal_wall_hours": (
                gpu_hours / gpu_device_count if gpu_device_count else None
            ),
            "reference_duration_seconds": REFERENCE_DURATION_SECONDS,
        },
        "objective": {
            "formula": "ACC + AUC - DP - EO",
            "direction": "maximize",
        },
        "pokec_gpu_guard": {
            "min_free_memory_mb": args.pokec_min_free_memory_mb,
            "max_utilization_percent": args.pokec_max_gpu_utilization,
            "poll_seconds": args.gpu_poll_seconds,
            "wait_timeout_seconds": args.gpu_wait_timeout,
        },
        "search_spaces": {dataset: SEARCH_SPACES[dataset] for dataset in args.datasets},
        "trials": [
            {
                "dataset": trial.dataset,
                "trial_id": trial.trial_id,
                "strategy": trial.strategy,
                "anchor_index": trial.anchor_index,
                "parameters": trial.parameters,
            }
            for trial in trials
        ],
    }
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    return path


def _metric(payload: Mapping[str, Any], metric: str) -> float:
    values = payload["metrics"]["target_after"][metric]
    if not isinstance(values, list) or len(values) != RUNS_PER_COMBINATION:
        raise ValueError(
            f"Expected {RUNS_PER_COMBINATION} target-after values for {metric}"
        )
    return math.fsum(float(value) for value in values) / RUNS_PER_COMBINATION


def _pareto_flags(rows: Sequence[Mapping[str, Any]]) -> List[bool]:
    """Maximize ACC/AUC and minimize DP/EO simultaneously."""
    flags: List[bool] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                other["acc"] >= row["acc"]
                and other["auc"] >= row["auc"]
                and other["dp"] <= row["dp"]
                and other["eo"] <= row["eo"]
            )
            strictly_better = (
                other["acc"] > row["acc"]
                or other["auc"] > row["auc"]
                or other["dp"] < row["dp"]
                or other["eo"] < row["eo"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def aggregate(
    args: argparse.Namespace,
    trials: Sequence[Trial],
    base_config: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for trial in trials:
        path = raw_result_path(args.results_dir, trial)
        if not is_complete_result(path, trial.parameters):
            continue
        with path.open("r", encoding="utf-8") as result_file:
            payload = json.load(result_file)
        acc = _metric(payload, "acc")
        auc = _metric(payload, "auc")
        dp = _metric(payload, "dp")
        eo = _metric(payload, "eo")
        score = acc + auc - dp - eo
        tuning = payload.get("tuning", {})
        rows.append(
            {
                "dataset": trial.dataset,
                "trial_id": trial.trial_id,
                "search_strategy": trial.strategy,
                "anchor_index": trial.anchor_index,
                "score": score,
                "acc": acc,
                "auc": auc,
                "dp": dp,
                "eo": eo,
                "runs_per_combination": tuning.get("runs_per_combination"),
                "duration_seconds": tuning.get("duration_seconds"),
                "parameters": trial.parameters,
                "result_path": str(path),
            }
        )

    best_configs: Dict[str, Dict[str, Any]] = {}
    for dataset in args.datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_rows.sort(key=lambda row: (-row["score"], row["trial_id"]))
        pareto = _pareto_flags(dataset_rows)
        for rank, (row, is_pareto) in enumerate(zip(dataset_rows, pareto), 1):
            row["rank"] = rank
            row["is_pareto"] = is_pareto
        if dataset_rows:
            best = dataset_rows[0]
            merged = dict(base_config[dataset])
            merged.update(best["parameters"])
            best_configs[dataset] = merged

    rows.sort(key=lambda row: (row["dataset"], row["rank"]))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / "summary.json"
    csv_path = args.results_dir / "summary.csv"
    best_path = args.results_dir / "best_config.yaml"
    pareto_path = args.results_dir / "pareto_front.json"

    with json_path.open("w", encoding="utf-8") as output:
        json.dump(rows, output, ensure_ascii=False, indent=2)
    fieldnames = [
        "dataset", "rank", "trial_id", "search_strategy", "anchor_index",
        "score", "is_pareto",
        "acc", "auc", "dp", "eo", "runs_per_combination",
        "duration_seconds",
        "parameters", "result_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["parameters"] = json.dumps(
                row["parameters"], sort_keys=True, ensure_ascii=False
            )
            writer.writerow(csv_row)
    with best_path.open("w", encoding="utf-8") as output:
        yaml.safe_dump(best_configs, output, sort_keys=False, allow_unicode=True)
    with pareto_path.open("w", encoding="utf-8") as output:
        json.dump(
            [row for row in rows if row["is_pareto"]],
            output,
            ensure_ascii=False,
            indent=2,
        )
    return rows, best_configs


def print_dry_run(
    args: argparse.Namespace,
    trials: Sequence[Trial],
    devices: Sequence[int],
) -> None:
    gpu_hours = estimated_gpu_hours(args.trial_counts)
    gpu_count = sum(device_id >= 0 for device_id in devices)
    print_status(
        f"Dry run: {len(trials)} unique combinations, exactly "
        f"{RUNS_PER_COMBINATION} runs per combination, "
        f"{len(trials) * RUNS_PER_COMBINATION} total runs, "
        f"devices={list(devices)}"
    )
    estimate = f"estimated={gpu_hours:.1f} GPU-hours from summary3"
    if gpu_count:
        estimate += f", ideal wall={gpu_hours / gpu_count:.1f} hours"
    print_status(f"  budgets={args.trial_counts}; {estimate}")
    for trial in trials:
        print_status(
            f"  {trial.dataset} {trial.name} [{trial.strategy}]: "
            f"{trial.parameters}"
        )
    if trials:
        example = trials[0]
        command = build_command(
            args,
            devices[0],
            example,
            raw_result_path(args.results_dir, example),
            log_path(args.results_dir, example),
        )
        print_status("Example command:")
        print_status(subprocess.list2cmdline(command))


def validate_args(args: argparse.Namespace) -> None:
    if set(args.trial_counts) != set(args.datasets):
        raise ValueError("Resolved trial budgets must match the selected datasets")
    if any(count < 1 for count in args.trial_counts.values()):
        raise ValueError("Every dataset trial budget must be at least 1")
    if args.pokec_min_free_memory_mb < 1:
        raise ValueError("--pokec-min-free-memory-mb must be positive")
    if not 0 <= args.pokec_max_gpu_utilization <= 100:
        raise ValueError("--pokec-max-gpu-utilization must lie in [0, 100]")
    if args.gpu_poll_seconds < 1:
        raise ValueError("--gpu-poll-seconds must be at least 1")
    if args.gpu_wait_timeout < 0:
        raise ValueError("--gpu-wait-timeout must be non-negative")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets must not contain duplicates")
    unknown = set(args.datasets) - set(DATASET_DOMAINS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    if not MAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing EMBER entry point: {MAIN_SCRIPT}")
    if not CONFIG_SCRIPT.exists():
        raise FileNotFoundError(f"Missing EMBER argument parser: {CONFIG_SCRIPT}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing EMBER config: {CONFIG_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run two EMBER training runs per sampled hyperparameter "
            "combination and parallelize combinations across GPUs."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_DOMAINS),
        default=list(DATASET_DOMAINS),
    )
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--trials",
        type=int,
        default=None,
        help="uniform unique-combination budget for every selected dataset",
    )
    budget_group.add_argument(
        "--trials-per-dataset",
        nargs="+",
        metavar="DATASET=COUNT",
        help=(
            "override round-4 defaults per dataset, for example "
            "bailA=192 germanA=192 pokec=320 syn=192"
        ),
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["auto"],
        metavar="ID",
        help="GPU ids (for example 0 1 2 3), auto, or -1 for CPU",
    )
    parser.add_argument(
        "--pokec-min-free-memory-mb",
        type=int,
        default=20000,
        help="minimum free VRAM required before starting a Pokec trial",
    )
    parser.add_argument(
        "--pokec-max-gpu-utilization",
        type=int,
        default=20,
        help="maximum GPU utilization allowed before starting Pokec",
    )
    parser.add_argument(
        "--gpu-poll-seconds",
        type=int,
        default=30,
        help="seconds between Pokec GPU admission checks",
    )
    parser.add_argument(
        "--gpu-wait-timeout",
        type=float,
        default=0.0,
        help="maximum wait in seconds; 0 waits indefinitely",
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument(
        "--target-seed-offset",
        type=int,
        default=None,
        help=(
            "optional fixed target-seed offset; omit to let runner.py use "
            "seed + run_idx * 1111 for the two runs"
        ),
    )
    parser.add_argument("--sampler-seed", type=int, default=2030)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild summaries from completed raw results without training",
    )
    return parser


def interleave_dataset_trials(
    trials_by_dataset: Mapping[str, Sequence[Trial]],
    dataset_order: Sequence[str],
) -> List[Trial]:
    """Round-robin datasets so every mode is covered before long random tails."""
    max_count = max((len(trials) for trials in trials_by_dataset.values()), default=0)
    return [
        trials_by_dataset[dataset][trial_index]
        for trial_index in range(max_count)
        for dataset in dataset_order
        if trial_index < len(trials_by_dataset[dataset])
    ]


def main() -> None:
    args = build_parser().parse_args()
    args.results_dir = args.results_dir.resolve()
    args.trial_counts = resolve_trial_counts(args)
    validate_args(args)
    devices = resolve_devices(args.gpus)
    if not args.dry_run and not args.aggregate_only:
        devices = filter_devices_for_pokec(args, devices)
    base_config = load_base_config()
    trials_by_dataset = {
        dataset: make_trials(
            dataset,
            args.trial_counts[dataset],
            args.sampler_seed,
            base_config,
        )
        for dataset in args.datasets
    }
    trials = interleave_dataset_trials(trials_by_dataset, args.datasets)

    if args.dry_run:
        print_dry_run(args, trials, devices)
        return

    failures: List[str] = []
    if not args.aggregate_only:
        manifest = write_manifest(args, trials, devices)
        print_status(f"Saved tuning manifest to {manifest}")
        tasks: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        for trial in trials:
            if args.rerun or not is_complete_result(
                raw_result_path(args.results_dir, trial), trial.parameters
            ):
                tasks.put(trial)
        for _ in devices:
            tasks.put(None)

        workers = [
            threading.Thread(
                target=worker,
                args=(args, device_id, tasks, failures, stop_event),
                daemon=True,
            )
            for device_id in devices
        ]
        for thread in workers:
            thread.start()
        tasks.join()
        for thread in workers:
            thread.join()

    rows, best_configs = aggregate(args, trials, base_config)
    print_status(
        f"Aggregated {len(rows)}/{len(trials)} completed combinations under "
        f"{args.results_dir}"
    )
    for dataset, config in best_configs.items():
        best_row = next(
            row for row in rows if row["dataset"] == dataset and row["rank"] == 1
        )
        print_status(
            f"Best {dataset}: trial={best_row['trial_id']} "
            f"score={best_row['score']:.3f} config={config}"
        )
    if failures:
        raise RuntimeError(
            f"{len(failures)} trial(s) failed; successful trials and summaries were retained"
        )


if __name__ == "__main__":
    main()
