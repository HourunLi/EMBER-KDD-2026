import sys
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import add_remaining_self_loops, degree


class Logger(object):
    """Tee stdout to a log file."""
    def __init__(self, fileN="Default.logs"):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(fileN), exist_ok=True)
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fair_metric(pred, labels, sens):
    """
    Compute Demographic Parity (DP) and Equalized Odds (EO).

    Args:
      pred  : np.array  binary predictions {0,1}
      labels: np.array  ground-truth labels {0,1}
      sens  : np.array  binary sensitive attribute {0,1}
    Returns:
      (parity, equality): scalar floats
    """
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y0 = np.bitwise_and(idx_s0, labels == 0)
    idx_s1_y0 = np.bitwise_and(idx_s1, labels == 0)
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    def safe_gap(left, right):
        if left.size == 0 or right.size == 0:
            return 0.0
        value = abs(float(left.mean()) - float(right.mean()))
        return value if np.isfinite(value) else 0.0

    parity = safe_gap(pred[idx_s0], pred[idx_s1])
    # Eq. (2): average the group gap in correct predictions for y=0 and y=1.
    y0_gap = safe_gap(
        pred[idx_s0_y0] == 0,
        pred[idx_s1_y0] == 0,
    )
    y1_gap = safe_gap(
        pred[idx_s0_y1] == 1,
        pred[idx_s1_y1] == 1,
    )
    equality = 0.5 * (y0_gap + y1_gap)
    return float(parity), float(equality)


def sens_correlation(features, sens_idx):
    """Pearson correlation between each feature and the sensitive attribute."""
    import pandas as pd
    corr = pd.DataFrame(np.array(features)).corr()
    return corr[sens_idx].to_numpy()


def conditional_samples(e):
    """
    Assign each sample to group 0 or 1 based on Pearson correlation
    with the first sample. Used for conditional negative sampling.
    """
    from scipy import stats
    anchor = e[0]
    return [0 if stats.pearsonr(anchor, e[k])[0] > 0 else 1 for k in range(e.shape[0])]


def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = True
    return mask
