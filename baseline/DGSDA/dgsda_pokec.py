"""
DGSDA on Pokec (pokec_z -> pokec_n)
Data loading : dataset.py::load_pokec logic, Benchmark-GraphFairness paths
Algorithm    : DGSDA BernNet (dual Bernstein props, MMD, entropy-min, theta)
Evaluation   : Acc, AUC-ROC, Demographic Parity, Equal Opportunity
"""
import os, sys, random, argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian, from_scipy_sparse_matrix
from torch_geometric.data import Data

sys.path.insert(0, "/home/disk2/lhr/sfda/code")
from utils import seed_everything, fair_metric

BENCHMARK_PATH = "/home/disk2/lhr/sfda/fairDomainAdaption/Benchmark-GraphFairness/dataset"

# ---------------------------------------------------------------------------
# 1. DATA LOADING  (mirrors dataset.py::load_pokec)
# ---------------------------------------------------------------------------

def _feature_norm(features):
    min_v = features.min(dim=0)[0]
    max_v = features.max(dim=0)[0]
    return 2.0 * (features - min_v) / (max_v - min_v).clamp(min=1e-6) - 1.0

def _index_to_mask(n, index):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[index] = True
    return mask

def load_pokec_benchmark(dataset_name, sens_attr="region",
                         predict_attr="I_am_working_in_field"):
    """
    Loads pokec_z or pokec_n from Benchmark-GraphFairness.
    Feature space = intersection of pokec_z and pokec_n columns
    (minus user_id/sens_attr/predict_attr) — identical to dataset.py::load_pokec.
    """
    path_z = os.path.join(BENCHMARK_PATH, "pokec_z")
    path_n = os.path.join(BENCHMARK_PATH, "pokec_n")
    dpath  = os.path.join(BENCHMARK_PATH, dataset_name)

    hz = list(pd.read_csv(os.path.join(path_z, "pokec_z.csv")).columns)
    hn = list(pd.read_csv(os.path.join(path_n, "pokec_n.csv")).columns)
    header = [c for c in hz if c in hn]
    for col in ("user_id", sens_attr, predict_attr):
        if col in header:
            header.remove(col)

    df = pd.read_csv(os.path.join(dpath, f"{dataset_name}.csv"))
    features = torch.FloatTensor(np.array(df[header], dtype=np.float32))
    labels   = torch.LongTensor(df[predict_attr].values.copy())
    labels[labels > 1] = 1
    sens_labels = torch.FloatTensor(df[sens_attr].values.astype(int))

    idx     = np.array(df["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    eu = np.genfromtxt(os.path.join(dpath, f"{dataset_name}_edges.txt"), dtype=int)
    edges = np.array(list(map(idx_map.get, eu.flatten())), dtype=int).reshape(eu.shape)
    n = labels.shape[0]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
                        shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(n)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    lbl = labels.numpy()
    i0, i1 = np.where(lbl==0)[0], np.where(lbl==1)[0]
    random.shuffle(i0); random.shuffle(i1)
    tr = np.append(i0[:int(.8*len(i0))], i1[:int(.8*len(i1))])
    va = np.append(i0[int(.8*len(i0)):int(.9*len(i0))], i1[int(.8*len(i1)):int(.9*len(i1))])
    te = np.append(i0[int(.9*len(i0)):], i1[int(.9*len(i1)):])

    features = _feature_norm(features)
    return Data(x=features, edge_index=edge_index, y=labels,
                sens_labels=sens_labels,
                train_mask=_index_to_mask(n, torch.LongTensor(tr)),
                val_mask=_index_to_mask(n, torch.LongTensor(va)),
                test_mask=_index_to_mask(n, torch.LongTensor(te)))

# ---------------------------------------------------------------------------
# 2. DGSDA MODEL  (verbatim from DGSDA/models.py)
# ---------------------------------------------------------------------------

class Bern_prop(MessagePassing):
    def __init__(self, K, is_source_domain=True, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = nn.Parameter(torch.Tensor(K+1),
                                 requires_grad=is_source_domain)
        self.is_source_domain = is_source_domain
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_source_domain:
            self.temp.data.fill_(1.0)
        else:
            self.temp.data = torch.linspace(1.0, 0.0, self.K+1)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)
        ei1, n1 = get_laplacian(edge_index, edge_weight, normalization="sym",
                                dtype=x.dtype, num_nodes=x.size(self.node_dim))
        ei2, n2 = add_self_loops(ei1, -n1, fill_value=2.0,
                                 num_nodes=x.size(self.node_dim))
        tmp = [x]
        for _ in range(self.K):
            x = self.propagate(ei2, x=x, norm=n2, size=None)
            tmp.append(x)
        out = (comb(self.K,0)/(2**self.K)) * TEMP[0] * tmp[self.K]
        for i in range(self.K):
            x = tmp[self.K-i-1]
            x = self.propagate(ei1, x=x, norm=n1, size=None)
            for _ in range(i):
                x = self.propagate(ei1, x=x, norm=n1, size=None)
            out = out + (comb(self.K,i+1)/(2**self.K)) * TEMP[i+1] * x
        return out

    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j


class BernNet(nn.Module):
    """
    DGSDA BernNet with 4 Bernstein propagators:
      prop1: source hidden (learnable theta_s)
      prop2: target hidden (fixed linspace theta_t)
      prop3: post-cls source
      prop4: post-cls target
    """
    def __init__(self, in_features, hidden, num_classes,
                 dropout, dp_rate=0.0, K=8):
        super().__init__()
        self.lin1  = nn.Linear(in_features, hidden)
        self.lin2  = nn.Linear(hidden, num_classes)
        self.prop1 = Bern_prop(K, is_source_domain=True)
        self.prop2 = Bern_prop(K, is_source_domain=False)
        self.prop3 = Bern_prop(K, is_source_domain=True)
        self.prop4 = Bern_prop(K, is_source_domain=False)
        self.dropout = dropout
        self.dp_rate = dp_rate

    def _hidden(self, data, is_source):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dp_rate, training=self.training)
        return self.prop1(x, data.edge_index) if is_source \
               else self.prop2(x, data.edge_index)

    def forward(self, data, is_source=True):
        x = self._hidden(data, is_source)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dp_rate, training=self.training)
        return self.prop3(x, data.edge_index) if is_source \
               else self.prop4(x, data.edge_index)

# ---------------------------------------------------------------------------
# 3. DGSDA LOSSES  (from DGSDA/utils.py and DGSDA/main.py)
# ---------------------------------------------------------------------------

def _gauss_kernel(src, tgt, km=2.0, kn=5, fix_sigma=None):
    n = src.size(0) + tgt.size(0)
    tot = torch.cat([src, tgt], dim=0)
    t0  = tot.unsqueeze(0).expand(n, n, tot.size(1))
    t1  = tot.unsqueeze(1).expand(n, n, tot.size(1))
    L2  = ((t0-t1)**2).sum(2)
    bw  = fix_sigma if fix_sigma else (L2.data.sum()+1e-6)/(n**2-n)
    bw /= km**(kn//2)
    return sum(torch.exp(-L2/(bw*km**i)) for i in range(kn))

def _get_mmd(sf, tf):
    K  = _gauss_kernel(sf, tf)
    bs = min(sf.size(0), tf.size(0))
    return torch.mean(K[:bs,:bs] + K[bs:,bs:] - K[:bs,bs:] - K[bs:,:bs])

def mmd_loss(sf, tf, sn=1000, times=5):
    sn0, tn0 = sf.size(0), tf.size(0)
    val = sf.new_tensor(0.0)
    for _ in range(times):
        si = torch.randint(sn0, (min(sn,sn0),))
        ti = torch.randint(tn0, (min(sn,tn0),))
        val = val + _get_mmd(sf[si], tf[ti])
    return val / times

def entropy_min_loss(logits):
    """Class-balanced entropy minimisation (DGSDA/main.py)."""
    p  = F.softmax(logits, dim=1)
    lp = F.log_softmax(logits, dim=1)
    a  = p.sum(dim=0)
    return -torch.sum(p * lp / (a/a.sum()).unsqueeze(0), dim=1).mean()

def theta_loss(model):
    """L1(theta_s, theta_t) + |theta_s|_1 + |theta_t|_1."""
    ts, tt = model.prop1.temp, model.prop2.temp
    return F.l1_loss(ts, tt) + ts.abs().sum() + tt.abs().sum()

# ---------------------------------------------------------------------------
# 4. EVALUATION
# ---------------------------------------------------------------------------

def evaluate(model, data, device, is_source=False, mask_name="test_mask"):
    model.eval()
    data = data.to(device)
    mask = getattr(data, mask_name).cpu().numpy()
    with torch.no_grad():
        logits = model(data, is_source=is_source)
        probs  = F.softmax(logits, dim=1)[:,1].cpu().numpy()
        preds  = (probs > 0.5).astype(int)
    y    = data.y.cpu().numpy()[mask]
    s    = data.sens_labels.cpu().numpy()[mask].astype(int)
    pr   = probs[mask]
    pd_m = preds[mask]
    acc = accuracy_score(y, pd_m) * 100
    auc = roc_auc_score(y, pr)*100 if len(set(y))==2 else float("nan")
    dp, eo = fair_metric(pd_m, y, s)
    return {"acc": acc, "auc": auc, "dp": dp*100, "eo": eo*100}

# ---------------------------------------------------------------------------
# 5. TRAINING  (DGSDA joint source+target)
# ---------------------------------------------------------------------------

def train_dgsda(args, source_data, target_data):
    dev = args.device
    sd  = source_data.to(dev)
    td  = target_data.to(dev)

    model = BernNet(
        in_features=sd.x.size(1), hidden=args.hidden,
        num_classes=2, dropout=args.dropout_ratio,
        dp_rate=args.dp_ratio, K=args.K,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epoch+1):
        model.train()
        opt.zero_grad()

        # source classification loss
        sl = model(sd, is_source=True)
        L_cls = F.cross_entropy(sl[sd.train_mask], sd.y[sd.train_mask])

        # theta consistency loss
        L_th = theta_loss(model)

        # MMD on lin1 hidden repr
        xd = F.dropout
        sh = F.relu(model.lin1(xd(sd.x, p=model.dropout, training=True)))
        th = F.relu(model.lin1(xd(td.x, p=model.dropout, training=True)))
        L_mmd = mmd_loss(sh, th)

        # entropy minimisation on target
        tl = model(td, is_source=False)
        L_ent = entropy_min_loss(tl)

        loss = L_cls + args.alpha*L_th + args.beta*L_mmd + args.gamma*L_ent
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = model(sd, is_source=True)
                vloss = F.cross_entropy(vl[sd.val_mask], sd.y[sd.val_mask]).item()
                tp = model(td, is_source=False).argmax(1)
                tacc = tp.eq(td.y).float().mean().item()
            print(f"Ep {epoch:4d} | cls={L_cls.item():.4f} th={L_th.item():.4f}"
                  f" mmd={L_mmd.item():.4f} ent={L_ent.item():.4f}"
                  f" | val={vloss:.4f} tgt_acc={tacc:.4f}")
            if vloss < best_val:
                best_val   = vloss
                best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(dev) for k,v in best_state.items()})

    src_m = evaluate(model, sd, dev, is_source=True,  mask_name="test_mask")
    tgt_m = evaluate(model, td, dev, is_source=False, mask_name="test_mask")
    print("\n=== Source (pokec_z) ===")
    print(f"  Acc={src_m['acc']:.2f}  AUC={src_m['auc']:.2f}  DP={src_m['dp']:.2f}  EO={src_m['eo']:.2f}")
    print("=== Target (pokec_n) ===")
    print(f"  Acc={tgt_m['acc']:.2f}  AUC={tgt_m['auc']:.2f}  DP={tgt_m['dp']:.2f}  EO={tgt_m['eo']:.2f}")
    return {"source": src_m, "target": tgt_m}


# ---------------------------------------------------------------------------
# 6. MULTI-RUN WRAPPER
# ---------------------------------------------------------------------------

def run_experiments(args):
    results = {"source": [], "target": []}
    for run in range(args.runs):
        seed = args.seed + run
        seed_everything(seed)
        print(f"\n{'='*60}\n Run {run+1}/{args.runs}  seed={seed}\n{'='*60}")
        sd = load_pokec_benchmark("pokec_z")
        td = load_pokec_benchmark("pokec_n")
        print(f"  Source: {sd.x.size(0)} nodes, {sd.x.size(1)} features")
        print(f"  Target: {td.x.size(0)} nodes, {td.x.size(1)} features")
        r = train_dgsda(args, sd, td)
        results["source"].append(r["source"])
        results["target"].append(r["target"])

    print(f"\n{'='*60}\n FINAL ({args.runs} runs)\n{'='*60}")
    for dom, name in (("source", "pokec_z"), ("target", "pokec_n")):
        for m in ("acc", "auc", "dp", "eo"):
            v = np.array([r[m] for r in results[dom]])
            print(f"  [{name}] {m.upper()}: {v.mean():.2f} +/- {v.std():.2f}")
    return results


# ---------------------------------------------------------------------------
# 7. ENTRY POINT
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="DGSDA on Pokec")
    p.add_argument("--lr",            type=float, default=0.01)
    p.add_argument("--wd",            type=float, default=0.01)
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--K",             type=int,   default=8)
    p.add_argument("--dropout_ratio", type=float, default=0.3)
    p.add_argument("--dp_ratio",      type=float, default=0.3)
    p.add_argument("--epoch",         type=int,   default=100)
    p.add_argument("--alpha",         type=float, default=0.05, help="theta loss weight")
    p.add_argument("--beta",          type=float, default=0.5,  help="MMD loss weight")
    p.add_argument("--gamma",         type=float, default=0.05, help="entropy loss weight")
    p.add_argument("--cuda",          type=int,   default=0)
    p.add_argument("--runs",          type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=1111)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.cuda >= 0 and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.cuda}")
    else:
        args.device = torch.device("cpu")
    print(f"Device: {args.device}")
    run_experiments(args)
