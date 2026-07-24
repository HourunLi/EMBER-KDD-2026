import sys
import random
import os
import numpy as np
import torch


class Logger:
    """Tee stdout to a log file."""

    def __init__(self, fileN="Default.logs"):
        self.terminal = sys.stdout
        directory = os.path.dirname(fileN)
        if directory:
            os.makedirs(directory, exist_ok=True)
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
      (parity, equality): scalar floats.  A metric is NaN when one of the
      demographic subgroups required by its conditional expectation is empty.
    """
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y0 = np.bitwise_and(idx_s0, labels == 0)
    idx_s1_y0 = np.bitwise_and(idx_s1, labels == 0)
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)

    def conditional_gap(left, right):
        # Eqs. (1)-(2) are undefined when either conditional subgroup is
        # empty.  Returning NaN preserves that meaning and prevents an
        # unavailable fairness measurement from being reported as zero gap.
        if left.size == 0 or right.size == 0:
            return float("nan")
        value = abs(float(left.mean()) - float(right.mean()))
        return value if np.isfinite(value) else float("nan")

    parity = conditional_gap(pred[idx_s0], pred[idx_s1])
    # Eq. (2): average the group gap in correct predictions for y=0 and y=1.
    y0_gap = conditional_gap(
        pred[idx_s0_y0] == 0,
        pred[idx_s1_y0] == 0,
    )
    y1_gap = conditional_gap(
        pred[idx_s0_y1] == 1,
        pred[idx_s1_y1] == 1,
    )
    equality = 0.5 * (y0_gap + y1_gap)
    return float(parity), float(equality)
