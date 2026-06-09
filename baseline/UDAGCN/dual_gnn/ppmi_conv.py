from collections import Counter
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import numpy as np
from tqdm import tqdm
from dual_gnn.cached_gcn_conv import CachedGCNConv


class PPMIConv(CachedGCNConv):

    def __init__(self, in_channels, out_channels,
                 weight=None, bias=None, improved=False, use_bias=True,
                 path_len=5,
                 **kwargs):
        super().__init__(in_channels, out_channels, weight, bias, improved, use_bias, **kwargs)
        self.path_len = path_len



    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):

        adj_dict = {}

        def add_edge(a, b):
            if a in adj_dict:
                neighbors = adj_dict[a]
            else:
                neighbors = set()
                adj_dict[a] = neighbors
            if b not in neighbors:
                neighbors.add(b)

        cpu_device = torch.device("cpu")
        for a, b in edge_index.t().detach().to(cpu_device).numpy():
            a = int(a)
            b = int(b)
            add_edge(a, b)
            add_edge(b, a)

        adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}

        def sample_neighbor(a):
            neighbors = adj_dict[a]
            random_index = np.random.randint(0, len(neighbors))
            return neighbors[random_index]


        # word_counter = Counter()
        walk_counters = {}

        def norm(counter):
            s = sum(counter.values())
            new_counter = Counter()
            for a, count in counter.items():
                new_counter[a] = counter[a] / s
            return new_counter

        for _ in tqdm(range(40)):
            for a in adj_dict:
                current_a = a
                current_path_len = np.random.randint(1, self.path_len + 1)
                for _ in range(current_path_len):
                    b = sample_neighbor(current_a)
                    if a in walk_counters:
                        walk_counter = walk_counters[a]
                    else:
                        walk_counter = Counter()
                        walk_counters[a] = walk_counter

                    walk_counter[b] += 1

                    current_a = b

        normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}

        prob_sums = Counter()

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                prob_sums[b] += prob

        ppmis = {}

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                # PPMI 是 Positive PMI。原始实现直接保留负 PMI，可能导致后续 degree 为负，
                # 进而在 deg.pow(-0.5) 中产生 NaN。这里过滤非有限值和非正值。
                denominator = max(prob_sums[b], 1e-12)
                pmi_input = prob / denominator * len(prob_sums) / self.path_len
                if pmi_input <= 0:
                    continue
                ppmi = max(np.log(pmi_input), 0.0)
                if not np.isfinite(ppmi) or ppmi <= 0:
                    continue
                ppmis[(a, b)] = ppmi

        new_edge_index = []
        edge_weight = []
        for (a, b), ppmi in ppmis.items():
            new_edge_index.append([a, b])
            edge_weight.append(ppmi)

        target_device = edge_index.device
        if len(new_edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=target_device)
            edge_weight = torch.empty((0, ), dtype=torch.float32, device=target_device)
        else:
            edge_index = torch.tensor(new_edge_index, dtype=torch.long, device=target_device).t()
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32, device=target_device)


        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg = torch.clamp(deg, min=1e-12)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        return edge_index, norm.type(torch.float32)





