import argparse
import torch
import os
import yaml

def read_config(args):
    # specify the model family

    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(fileNamePath, 'config/{}.yaml'.format(args.times))
    print(yamlPath)
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cont = f.read()
        # TODO
        config_dict = yaml.safe_load(cont)['g'][args.dataset]

    for key, value in config_dict.items():
        args.__setattr__(key, value)

    return args

def mprint(*arg, **kwargs):
    if VERBOSE:  # 仅在 arg.verbose 为 True 时输出
        print(*arg, **kwargs)
        

parser = argparse.ArgumentParser(description='mine Arguments.')

# dataset
parser.add_argument('--dataset', type=str, default='gernman')
parser.add_argument('--inid', type=str, default='_2')# --inid=_B0 
parser.add_argument('--outid', type=str, default='_1')# --outid=all 

# training config
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--start', type=int, default=50)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--dic_epochs', type=int, default=2) # discriminator epochs
parser.add_argument('--dtb_epochs', type=int, default=5) # debiasing mechanism epochs
parser.add_argument('--cla_epochs', type=int, default=10) # classifier epochs
parser.add_argument('--clo_epochs', type=int, default=2) # closure epochs
parser.add_argument('--con_epochs', type=int, default=5) # closure epochs
parser.add_argument('--a_epochs', type=int, default=5) # adversarial
parser.add_argument('--g_epochs', type=int, default=5) # generation
parser.add_argument('--g_lr', type=float, default=5e-4)
parser.add_argument('--g_wd', type=float, default=1e-4)
parser.add_argument('--d_lr', type=float, default=5e-4)
parser.add_argument('--d_wd', type=float, default=0)
parser.add_argument('--c_lr', type=float, default=5e-4)
parser.add_argument('--c_wd', type=float, default=1e-4)
parser.add_argument('--e_lr', type=float, default=5e-4)
parser.add_argument('--e_wd', type=float, default=0)
parser.add_argument('--p_lr', type=float, default=5e-4)
parser.add_argument('--p_wd', type=float, default=0)
parser.add_argument('--early_stopping', type=int, default=0)
parser.add_argument('--predictfile', type=str, default='tmp')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--clip_e', type=float, default=1)
parser.add_argument('--clip_c', type=float, default=1)
parser.add_argument('--weight_clip', type=str, default='yes')
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--ood', type=int, default=1)
parser.add_argument('--discri', type=int, default=1)
parser.add_argument('--disturb', type=int, default=1)
parser.add_argument('--modiStru', type=int, default=0)
parser.add_argument('--times', type=str, default='config')

# network
parser.add_argument('--n_layer', type=int, default=3, help='the number of layers')
parser.add_argument('--encoder', type=str, choices=['GCN', 'GAT', 'SAGE', 'MLP'], default='GCN', help='GNN bachbone')
parser.add_argument('--prop', type=str, default='scatter')
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--feature_dim', type=int, default=32)
parser.add_argument('--activation', type=str, default='sigmoid')


# mixup config
parser.add_argument('--imb_ratio', type=float, default=100, help='imbalance ratio')
parser.add_argument('--gdc', type=str, choices=['ppr', 'hk', 'none'], default='ppr', help='how to convert to weighted graph')
parser.add_argument('--warmup', type=int, default=5, help='warmup epoch')
parser.add_argument('--max_flag', action="store_true", help='synthesizing to max or mean num of training set. default is mean') 
parser.add_argument('--no_mask', action="store_true", help='whether to mask the self class in sampling neighbor classes. default is mask')

# adversarial learning config
parser.add_argument("--K", type=int, default=10, help="Number of whole adversarial phases")
parser.add_argument("--T_min", type=int, default=10, help="Number of iterations in Min-phase")
parser.add_argument("--T_max", type=int, default=15, help="Number of iterations in Max-phase")
parser.add_argument("--gamma", type=float, default=1.0, help="Higher value leads to stricter distance constraint")
parser.add_argument("--adv_learning_rate", type=float, default=1.0, help="Learning rate for adversarial training")
parser.add_argument("--flip_p", type=float, default=0.1, help="flip probability")
parser.add_argument('--f_mask', action='store_true', help = "mask for adversarial learning")
parser.add_argument('--ratio', type=float, default=1)

# contrastive learning
parser.add_argument('--close', action='store_true')
parser.add_argument("--tau", type=float, default=1, help="contrastive temperature")


# hardware
parser.add_argument('--cuda', action='store_true', help='use of cuda')
parser.add_argument("--device_id", type=str, default="0", help="device id for gpu")

# others
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--tune', action='store_true', help='if tune')
parser.add_argument('--verbose', action='store_true', help = "verbose for training")

args = parser.parse_args()
args.strlist = None

VERBOSE = args.verbose
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda:{}".format(args.device_id))
    mprint("using gpu:{} to train the model".format(args.device_id))
else:
    args.device = torch.device("cpu")
    mprint("using cpu to train the model")
    
if args.tune:
    args = read_config(args)
if args.outid == "all":
    args.outid = ""
mprint(args)