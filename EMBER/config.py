import argparse
import math
import os
import sys

import torch
import yaml

from adaptation import ABLATION_MODES
from utils import Logger


def read_config(args):
    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(fileNamePath, 'config/config.yaml')
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cont = f.read()
        config_dict = yaml.safe_load(cont)[args.dataset]

    print(config_dict)
    for key, value in config_dict.items():
        args.__setattr__(key, value)
    return args


_RUNTIME_OVERRIDE_KEYS = {
    'dataset', 'inid', 'outid', 'device_id', 'device',
    'log_path', 'result_path', 'runs_override', 'seed', 'target_seed',
    'ablation', 'disable_embedding_export',
    'save_visualization_embeddings', 'use_checkpoint',
    'disable_checkpoint_save', 'source_only', 'override', 'tune', 'verbose',
}


def apply_overrides(args, overrides):
    """Apply typed KEY=VALUE overrides after loading the dataset YAML."""
    for item in overrides:
        key, separator, raw_value = item.partition('=')
        key = key.strip()
        if not separator or not key:
            parser.error("invalid --override {!r}; expected KEY=VALUE".format(item))
        if key in _RUNTIME_OVERRIDE_KEYS:
            parser.error(
                "{} is a runtime argument; use its dedicated CLI option".format(key)
            )
        if not hasattr(args, key):
            parser.error("unknown override key: {}".format(key))

        current = getattr(args, key)
        action = next(
            (candidate for candidate in parser._actions if candidate.dest == key),
            None,
        )
        expected_type = action.type if action and action.type else type(current)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as error:
            parser.error("invalid value for {}: {}".format(key, error))
        if isinstance(value, (dict, list, tuple, set)):
            parser.error("{} expects a scalar value".format(key))

        if expected_type is bool:
            if not isinstance(value, bool):
                parser.error("{} expects true or false".format(key))
        elif expected_type is int:
            if isinstance(value, bool) or not isinstance(value, int):
                parser.error("{} expects an integer".format(key))
        elif expected_type is float:
            if isinstance(value, bool):
                parser.error("{} expects a float".format(key))
            try:
                value = float(value)
            except (TypeError, ValueError):
                parser.error("{} expects a float".format(key))
            if not math.isfinite(value):
                parser.error("{} expects a finite float".format(key))
        elif expected_type is str and not isinstance(value, str):
            parser.error("{} expects a string".format(key))
        if action and action.choices and value not in action.choices:
            parser.error(
                "{} must be one of {}".format(
                    key, ", ".join(str(choice) for choice in action.choices)
                )
            )

        setattr(args, key, value)
    return args


def explicit_cli_values(argument_parser, parsed_args, argv):
    """Return values for options explicitly present on the command line."""
    destinations = set()
    for token in argv:
        if not token.startswith('--'):
            continue
        option = token.split('=', 1)[0]
        action = argument_parser._option_string_actions.get(option)
        if action is not None:
            destinations.add(action.dest)
    return {
        destination: getattr(parsed_args, destination)
        for destination in destinations
    }

def mprint(*arg, **kwargs):
    if VERBOSE:
        print(*arg, **kwargs)


parser = argparse.ArgumentParser(description='Train and evaluate EMBER.')

parser.add_argument('--dataset', type=str, default='bailA')
parser.add_argument('--inid',    type=str, default='_2')
parser.add_argument('--outid',   type=str, default='_1')
#parser.add_argument('--dataset', type=str, default='germanA')
#parser.add_argument('--inid',    type=str, default='_2')
#parser.add_argument('--outid',   type=str, default='_1')
#parser.add_argument('--dataset', type=str, default='pokec')
#parser.add_argument('--inid',    type=str, default='_z')
#parser.add_argument('--outid',   type=str, default='_n')
# parser.add_argument('--dataset', type=str, default='syn')
# parser.add_argument('--inid',    type=str, default='-2')
# parser.add_argument('--outid',   type=str, default='-1')
# optimisation
parser.add_argument('--lr',           type=float, default=0.004)
parser.add_argument('--lr2_reg',      type=float, default=0.001)
parser.add_argument('--train_epochs', type=int,   default=800)
parser.add_argument('--dropout',      type=float, default=0.5)
parser.add_argument('--runs',         type=int,   default=5)

# network
parser.add_argument(
    '--cls_n_layers',
    type=int,
    default=3,
    help='number of GNN layers in the task encoder; ignored by MLP/vanilla',
)
parser.add_argument(
    '--sens_n_layers',
    type=int,
    default=3,
    help='number of GNN layers in the sensitive encoder; ignored by MLP/vanilla',
)
parser.add_argument('--cls_encoder', type=str,
                    choices=['GCN', 'GAT', 'SAGE', 'MLP', 'vanilla'],
                    default='GCN')
parser.add_argument('--sens_encoder', type=str,
                    choices=['GCN', 'GAT', 'SAGE', 'MLP', 'vanilla'],
                    default='GCN')
parser.add_argument('--hidden_dim',    type=int, default=32)
parser.add_argument('--device_id',     type=str, default='0')

# Source pretraining (paper notation: beta, alpha, gamma)
parser.add_argument('--lambda_fair',  type=float, default=1)
parser.add_argument('--meta_lr', type=float, default=0.01)
parser.add_argument(
    '--lambda_coord',
    type=float,
    default=1.0,
    help='coordination strength gamma in [0, 1]; gamma=1 is the full fairness update',
)
parser.add_argument('--lambda_sen', type=float, default=1.0)
parser.add_argument('--lambda_dec', type=float, default=1.0)
parser.add_argument('--source_mmd_bandwidth', type=float, default=1.0)

# Fixed implementation controls.  These affect memory use / stochastic
# approximation, but are not method hyperparameters to tune per dataset, so
# they intentionally stay out of config/config.yaml.
parser.add_argument('--disentangle_batch_size', type=int, default=512)
parser.add_argument('--source_mmd_min_samples', type=int, default=1)
parser.add_argument('--source_mmd_max_samples', type=int, default=1024)
parser.add_argument('--mmd_chunk_size', type=int, default=256)

# SFDA target adaptation
parser.add_argument('--adapt_epochs', type=int,   default=200)
parser.add_argument('--tau_c',        type=float, default=0.7)   # confidence threshold delta
parser.add_argument('--lambda_pi',    type=float, default=1.0)   # Bayesian prior strength eta
parser.add_argument('--adapt_lr',     type=float, default=1e-3)
parser.add_argument('--residual_inner_steps', type=int, default=5)
parser.add_argument('--lambda_residual_l2', type=float, default=1e-3)
parser.add_argument('--group_pseudocount', type=float, default=1.0)  # nu_0 in Eq. (14)
parser.add_argument('--prior_confidence_threshold', type=float, default=0.7)  # delta_p
parser.add_argument('--prior_pseudocount', type=float, default=1.0)  # n_0 in Eq. (17)
parser.add_argument('--prior_discount', type=float, default=0.9)  # omega in Eq. (17)
parser.add_argument('--proto_temp',   type=float, default=0.1)   # softmax temperature for posterior sharpening
parser.add_argument('--ablation', type=str,
                    choices=ABLATION_MODES,
                    default='full')

# misc
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--result_path', type=str, default=None)
parser.add_argument('--runs_override', type=int, default=None)
parser.add_argument(
    '--source_only', '--source-only',
    action='store_true',
    help=(
        'tuning-only mode: train source objectives on source train_mask, '
        'report source val metrics, and skip target loading/adaptation'
    ),
)
parser.add_argument(
    '--override', action='append', default=[], metavar='KEY=VALUE',
    help='override a config value after loading the dataset YAML; repeatable',
)
parser.add_argument('--seed',     type=int, default=1111)
parser.add_argument('--target_seed', type=int, default=None)
checkpoint_group = parser.add_mutually_exclusive_group()
checkpoint_group.add_argument(
    '--use_checkpoint', '--use-checkpoint',
    dest='use_checkpoint',
    action='store_true',
    help='load a matching source checkpoint when available (default)',
)
checkpoint_group.add_argument(
    '--no_checkpoint', '--no-checkpoint',
    dest='use_checkpoint',
    action='store_false',
    help='always retrain the source model',
)
parser.set_defaults(use_checkpoint=True)
parser.add_argument(
    '--disable_checkpoint_save', '--disable-checkpoint-save',
    action='store_true',
    help='do not write source checkpoints (useful for parallel one-run tuning)',
)
parser.add_argument('--save_visualization_embeddings', action='store_true',
                    help='export embeddings for root-level t-SNE visualization')
parser.add_argument('--disable_embedding_export', action='store_true')
parser.add_argument('--tune',     action='store_true')
parser.set_defaults(tune=True)
parser.add_argument('--verbose',  action='store_true')

args = parser.parse_args()
cli_values = explicit_cli_values(parser, args, sys.argv[1:])
if args.log_path is None:
    args.log_path = os.path.join('logs', '{}.log'.format(args.dataset))
sys.stdout = Logger(args.log_path)

VERBOSE = args.verbose
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device('cuda:{}'.format(args.device_id))
    mprint('using gpu:{} to train the model'.format(args.device_id))
else:
    args.device = torch.device('cpu')
    mprint('using cpu to train the model')

if args.tune:
    args = read_config(args)
for key, value in cli_values.items():
    setattr(args, key, value)
args = apply_overrides(args, args.override)


def require(condition, message):
    if not condition:
        parser.error(message)


require(args.train_epochs >= 2, "train_epochs must be at least 2")
require(args.cls_n_layers >= 1, "cls_n_layers must be at least 1")
require(args.sens_n_layers >= 1, "sens_n_layers must be at least 1")
require(args.lr > 0.0, "lr must be positive")
require(args.lr2_reg >= 0.0, "lr2_reg must be non-negative")
require(args.lambda_fair >= 0.0, "lambda_fair (beta) must be non-negative")
require(args.meta_lr > 0.0, "meta_lr (alpha) must be positive")
require(
    0.0 <= args.lambda_coord <= 1.0,
    "lambda_coord (gamma) must lie in [0, 1]",
)
require(args.lambda_sen >= 0.0, "lambda_sen must be non-negative")
require(args.lambda_dec >= 0.0, "lambda_dec must be non-negative")
require(args.disentangle_batch_size >= 1, "disentangle_batch_size must be positive")
require(args.source_mmd_bandwidth > 0.0, "source_mmd_bandwidth must be positive")
require(args.source_mmd_min_samples >= 1, "source_mmd_min_samples must be positive")
require(args.source_mmd_max_samples >= 0, "source_mmd_max_samples must be non-negative")
require(args.mmd_chunk_size >= 1, "mmd_chunk_size must be positive")
require(args.adapt_epochs >= 1, "adapt_epochs must be at least 1")
require(args.adapt_lr > 0.0, "adapt_lr must be positive")
require(args.residual_inner_steps >= 1, "residual_inner_steps must be at least 1")
require(args.lambda_residual_l2 >= 0.0, "lambda_residual_l2 must be non-negative")
require(0.0 <= args.tau_c <= 1.0, "tau_c (delta) must lie in [0, 1]")
require(
    0.0 <= args.prior_confidence_threshold <= 1.0,
    "prior_confidence_threshold (delta_p) must lie in [0, 1]",
)
require(args.proto_temp > 0.0, "proto_temp (tau) must be positive")
require(0.0 <= args.lambda_pi <= 1.0, "lambda_pi (eta) must lie in [0, 1]")
require(args.group_pseudocount > 0.0, "group_pseudocount (nu_0) must be positive")
require(args.prior_pseudocount > 0.0, "prior_pseudocount (n_0) must be positive")
require(0.0 <= args.prior_discount < 1.0, "prior_discount (omega) must lie in [0, 1)")
if args.runs_override is not None:
    args.runs = args.runs_override
print(args)
