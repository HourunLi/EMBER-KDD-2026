import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from .classifier import Classifier


class MLP(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super().__init__()
        self.lin = nn.Sequential(
            Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index=None):
        return self.lin(x)


class GCN_Body(nn.Module):
    """Single-layer GCN, used as 'vanilla' backbone."""

    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GNN(nn.Module):
    """Multi-layer GNN with layer-mean pooling."""

    def __init__(self, args, input_dim, hidden_dim, encoder_type):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.feat_proj = nn.Linear(input_dim, hidden_dim)

        etype = encoder_type.lower()
        if etype == "gcn":
            self.gnn_layers = nn.ModuleList(
                [GCNConv(hidden_dim, hidden_dim) for _ in range(args.n_layers)]
            )
        elif etype == "sage":
            self.gnn_layers = nn.ModuleList(
                [SAGEConv(hidden_dim, hidden_dim) for _ in range(args.n_layers)]
            )
        elif etype == "gat":
            self.gnn_layers = nn.ModuleList(
                [GATConv(hidden_dim, hidden_dim, heads=1) for _ in range(args.n_layers)]
            )
        else:
            raise NotImplementedError(
                f"Unknown encoder type '{encoder_type}'. Choose from GCN, SAGE, GAT."
            )

    def forward(self, feat, edge_index):
        x = self.feat_proj(feat)
        layer_outputs = []
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            layer_outputs.append(x)
        return torch.stack(layer_outputs, dim=1).mean(dim=1)


def _build_encoder(args, encoder_type):
    """Build one independent representation encoder."""
    if encoder_type == "MLP":
        return MLP(args, args.num_features, args.hidden_dim)
    if encoder_type == "vanilla":
        return GCN_Body(args.num_features, args.hidden_dim, args.dropout)
    return GNN(args, args.num_features, args.hidden_dim, encoder_type)


class FairGNN(nn.Module):
    """EMBER with independent task-classification and sensitive encoders."""

    def __init__(
        self,
        args,
        cls_encoder_type,
        sens_encoder_type,
        include_sensitive_classifier=True,
    ):
        super().__init__()

        # These calls intentionally create two different modules even when the
        # configured encoder types are identical.  No representation parameter
        # is shared between the classification and sensitive branches.
        self.cls_encoder = _build_encoder(args, cls_encoder_type)
        self.sens_encoder = _build_encoder(args, sens_encoder_type)
        self.cls_head = Classifier(input_dim=args.hidden_dim, num_cls=1)
        self.sens_head = (
            Classifier(input_dim=args.hidden_dim, num_cls=1)
            if include_sensitive_classifier
            else None
        )

        self._init_weights()
        self.to(args.device)

    def forward(self, feat, edge_index):
        """Return the task view and task logits used at inference time."""
        task_view, _ = self.encode_views(feat, edge_index)
        return task_view, self.cls_head(task_view)

    def encode_views(self, feat, edge_index):
        """Return task view ``z`` and sensitive view ``e`` from separate encoders."""
        task_view = self.cls_encoder(feat, edge_index)
        sensitive_view = self.sens_encoder(feat, edge_index)
        return task_view, sensitive_view

    def source_forward(self, feat, edge_index):
        """Return both views and both source-only classifier outputs."""
        if self.sens_head is None:
            raise RuntimeError(
                "The sensitive classifier is available only during source training"
            )
        task_view, sensitive_view = self.encode_views(feat, edge_index)
        return (
            task_view,
            sensitive_view,
            self.cls_head(task_view),
            self.sens_head(sensitive_view),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
