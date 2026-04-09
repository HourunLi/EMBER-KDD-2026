import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from .classifier import Classifier


class MLP(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(MLP, self).__init__()
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
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GNN(nn.Module):
    """
    Multi-layer GNN backbone with skip connections via layer-mean pooling.
    Supports GCN, SAGE, GAT.
    """
    def __init__(self, args, input_dim, hidden_dim, encoder_type):
        super(GNN, self).__init__()
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
        # collect GNN layer outputs (exclude feat_proj output from pooling)
        layer_outputs = []
        for i, gnn in enumerate(self.gnn_layers):
            x = gnn(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            layer_outputs.append(x)

        # mean pooling over GNN layer outputs
        out = torch.stack(layer_outputs, dim=1).mean(dim=1)  # [N, hidden_dim]
        return out


class Encoder(nn.Module):
    """Legacy encoder (single-head) kept for compatibility."""
    def __init__(self, args, encoder_type):
        super(Encoder, self).__init__()
        if encoder_type == "MLP":
            self.body = MLP(args, args.num_features, args.hidden_dim)
        elif encoder_type == "vanilla":
            self.body = GCN_Body(args.num_features, args.hidden_dim, args.dropout)
        else:
            self.body = GNN(args, args.num_features, args.hidden_dim, encoder_type)
        self.fc = Classifier(args, input_dim=args.hidden_dim, num_cls=1)

        self._init_weights()
        self.to(args.device)

    def forward(self, feat, edge_index):
        embeddings = self.body(feat, edge_index)
        cls = self.fc(embeddings)
        return embeddings, cls

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class FairGNN(nn.Module):
    """
    Shared GNN backbone with a single classification head.

    Fairness is enforced at training time via a differentiable EO/DP loss,
    coordinated with the cls loss through MetaAlign gradient alignment.

      backbone  ->  emb  ->  cls_head  ->  cls_logit  (predicts y)
    """
    def __init__(self, args, encoder_type):
        super(FairGNN, self).__init__()

        if encoder_type == "MLP":
            self.backbone = MLP(args, args.num_features, args.hidden_dim)
        elif encoder_type == "vanilla":
            self.backbone = GCN_Body(args.num_features, args.hidden_dim, args.dropout)
        else:
            self.backbone = GNN(args, args.num_features, args.hidden_dim, encoder_type)

        self.cls_head = Classifier(args, input_dim=args.hidden_dim, num_cls=1)

        self._init_weights()
        self.to(args.device)

    def forward(self, feat, edge_index):
        """
        Returns:
          emb       : [N, hidden_dim]  node embeddings
          cls_logit : [N, 1]           task prediction logits
        """
        emb       = self.backbone(feat, edge_index)
        cls_logit = self.cls_head(emb)
        return emb, cls_logit

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
