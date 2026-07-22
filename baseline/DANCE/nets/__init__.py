from .gcn import create_gcn
from .gat import create_gat
from .sage import create_sage
from .mlp import create_mlp
from model import *

def create_encoder(args):
    encoder = None
    if (args.encoder == 'GCN'):
        encoder = create_gcn(args)
    elif (args.encoder == 'GAT'):
        encoder = create_gat(args)
    elif (args.encoder == 'SAGE'):
        encoder = create_sage(args)
    elif (args.encoder == 'MLP'):
        encoder = create_mlp(args)
    else:
        print("Error: fault encoder parameter")
        exit(1)
    optimizer_e = torch.optim.Adam(params = encoder.parameters(), weight_decay=args.e_wd, lr = args.e_lr)
    return encoder.to(args.device), optimizer_e

# def create_encoder(args):
#     if (args.encoder == 'MLP'):
#         encoder = MLP_encoder(args).to(args.device)
#         optimizer_e = torch.optim.Adam([
#             dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
#     elif (args.encoder == 'GCN'):
#         if args.prop == 'scatter':
#             encoder = GCN_encoder_scatter(args).to(args.device)
#         else:
#             encoder = GCN_encoder_spmm(args).to(args.device)
#         optimizer_e = torch.optim.Adam([
#             dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
#             dict(params=encoder.bias, weight_decay=args.e_wd)], lr=args.e_lr)
#     elif (args.encoder == 'GIN'):
#         encoder = GIN_encoder(args).to(args.device)
#         optimizer_e = torch.optim.Adam([
#             dict(params=encoder.conv.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
#     elif (args.encoder == 'SAGE'):
#         encoder = SAGE_encoder(args).to(args.device)
#         optimizer_e = torch.optim.Adam([
#             dict(params=encoder.conv1.parameters(), weight_decay=args.e_wd),
#             dict(params=encoder.conv2.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
#     return encoder, optimizer_e