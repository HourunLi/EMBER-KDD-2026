import argparse
import torch
import os
import yaml
from utils import *
import sys

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

def mprint(*arg, **kwargs):
    if VERBOSE:
        print(*arg, **kwargs)


parser = argparse.ArgumentParser(description='mine Arguments.')

# # dataset
# parser.add_argument('--dataset', type=str, default='syn')
# parser.add_argument('--inid',    type=str, default='-2')
# parser.add_argument('--outid',   type=str, default='-1')
# parser.add_argument('--dataset', type=str, default='pokec')
# parser.add_argument('--inid',    type=str, default='_z')
# parser.add_argument('--outid',   type=str, default='_n')
parser.add_argument('--dataset', type=str, default='bailA')
parser.add_argument('--inid',    type=str, default='_2')
parser.add_argument('--outid',   type=str, default='_1')
# parser.add_argument('--dataset', type=str, default='german')
# parser.add_argument('--inid',    type=str, default='_2')
# parser.add_argument('--outid',   type=str, default='_1')
# optimisation
parser.add_argument('--lr',           type=float, default=0.004)
parser.add_argument('--lr2_reg',      type=float, default=0.001)
parser.add_argument('--train_epochs', type=int,   default=800)
parser.add_argument('--dropout',      type=float, default=0.5)
parser.add_argument('--tau',          type=float, default=1.0)
parser.add_argument('--runs',         type=int,   default=5)

# network
parser.add_argument('--n_layers',      type=int, default=3)
parser.add_argument('--inter_encoder', type=str,
                    choices=['GCN', 'GAT', 'SAGE', 'MLP', 'vanilla'],
                    default='GCN')
parser.add_argument('--hidden_dim',    type=int, default=32)
parser.add_argument('--device_id',     type=str, default='0')

# fairness loss weights
parser.add_argument('--lambda_fair',  type=float, default=1)
parser.add_argument('--meta_lr_src',  type=float, default=0.01)  # inner-loop lr for source meta-learning

# SFDA target adaptation
parser.add_argument('--adapt_epochs', type=int,   default=200)
parser.add_argument('--tau_c',        type=float, default=0.2)   # top-tau_c fraction of nodes kept as high-confidence
parser.add_argument('--lambda_pi',    type=float, default=1.0)
parser.add_argument('--alpha_p',      type=float, default=0.90)
parser.add_argument('--alpha_r',      type=float, default=0.99)
parser.add_argument('--alpha_pi',     type=float, default=0.90)
parser.add_argument('--lambda_s',     type=float, default=1.0)
parser.add_argument('--lambda_e',     type=float, default=0.1)
parser.add_argument('--lambda_res',   type=float, default=0.01)
parser.add_argument('--meta_lr',      type=float, default=0.01)
parser.add_argument('--adapt_lr',     type=float, default=1e-3)
parser.add_argument('--tau_adjust',   type=float, default=1.0)   # Bayesian logit adjustment temperature
parser.add_argument('--mmd_chunk_size', type=int, default=1024)
parser.add_argument('--proto_temp',   type=float, default=0.1)   # softmax temperature for posterior sharpening

# misc
parser.add_argument('--log_path', type=str, default='logs/log.txt')
parser.add_argument('--seed',     type=int, default=1111)
parser.add_argument('--use_checkpoint',    dest='use_checkpoint', action='store_true',
                    help='load checkpoint and skip source training when available')
parser.set_defaults(use_checkpoint=False)
parser.add_argument('--tune',     action='store_true')
parser.set_defaults(tune=True)
parser.add_argument('--verbose',  action='store_true')

args = parser.parse_args()
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
print(args)
