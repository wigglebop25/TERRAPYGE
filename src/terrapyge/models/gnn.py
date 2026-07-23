"""GNN architectures for TERRAPYGE.

Heterogeneous Graph Neural Network models using PyTorch Geometric
HeteroConv for dual-edge (spatial + hydrological) slope unit graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, HeteroConv


class HeteroGCN(nn.Module):
    """Graph Convolutional Network with HeteroConv for dual-edge graph.

    Message passing over two edge types:
    - ('su', 'spatial', 'su'): Undirected Queen contiguity
    - ('su', 'hydro', 'su'): Directed D8 flow

    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output classes.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(
            HeteroConv({
                ('su', 'spatial', 'su'): GCNConv(in_channels, hidden_channels),
                ('su', 'hydro', 'su'): GCNConv(in_channels, hidden_channels),
            }, aggr='sum')
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv({
                    ('su', 'spatial', 'su'): GCNConv(hidden_channels, hidden_channels),
                    ('su', 'hydro', 'su'): GCNConv(hidden_channels, hidden_channels),
                }, aggr='sum')
            )

        # Output layer
        self.convs.append(
            HeteroConv({
                ('su', 'spatial', 'su'): GCNConv(hidden_channels, hidden_channels),
                ('su', 'hydro', 'su'): GCNConv(hidden_channels, hidden_channels),
            }, aggr='sum')
        )

        # Classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                          for k, v in x_dict.items()}

        out = self.classifier(x_dict['su'])
        return out


class HeteroSAGE(nn.Module):
    """GraphSAGE with HeteroConv for dual-edge graph.

    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output classes.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(
            HeteroConv({
                ('su', 'spatial', 'su'): SAGEConv(in_channels, hidden_channels),
                ('su', 'hydro', 'su'): SAGEConv(in_channels, hidden_channels),
            }, aggr='sum')
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv({
                    ('su', 'spatial', 'su'): SAGEConv(hidden_channels, hidden_channels),
                    ('su', 'hydro', 'su'): SAGEConv(hidden_channels, hidden_channels),
                }, aggr='sum')
            )

        # Output layer
        self.convs.append(
            HeteroConv({
                ('su', 'spatial', 'su'): SAGEConv(hidden_channels, hidden_channels),
                ('su', 'hydro', 'su'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='sum')
        )

        # Classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                          for k, v in x_dict.items()}

        out = self.classifier(x_dict['su'])
        return out


class HeteroGAT(nn.Module):
    """Graph Attention Network with HeteroConv for dual-edge graph.

    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output classes.
        num_layers: Number of GNN layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(
            HeteroConv({
                ('su', 'spatial', 'su'): GATConv(in_channels, hidden_channels // num_heads,
                                                  heads=num_heads, dropout=dropout),
                ('su', 'hydro', 'su'): GATConv(in_channels, hidden_channels // num_heads,
                                                heads=num_heads, dropout=dropout),
            }, aggr='sum')
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv({
                    ('su', 'spatial', 'su'): GATConv(hidden_channels, hidden_channels // num_heads,
                                                      heads=num_heads, dropout=dropout),
                    ('su', 'hydro', 'su'): GATConv(hidden_channels, hidden_channels // num_heads,
                                                    heads=num_heads, dropout=dropout),
                }, aggr='sum')
            )

        # Output layer
        self.convs.append(
            HeteroConv({
                ('su', 'spatial', 'su'): GATConv(hidden_channels, hidden_channels // num_heads,
                                                  heads=num_heads, dropout=dropout),
                ('su', 'hydro', 'su'): GATConv(hidden_channels, hidden_channels // num_heads,
                                                heads=num_heads, dropout=dropout),
            }, aggr='sum')
        )

        # Classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                          for k, v in x_dict.items()}

        out = self.classifier(x_dict['su'])
        return out


def get_model(model_type, in_channels, hidden_channels, out_channels,
              num_layers=3, dropout=0.2, num_heads=4):
    """Factory function to create GNN models.

    Args:
        model_type: One of 'HeteroGCN', 'HeteroSAGE', 'HeteroGAT'.
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output classes.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
        num_heads: Number of attention heads (GAT only).

    Returns:
        GNN model instance.
    """
    models = {
        'HeteroGCN': HeteroGCN,
        'HeteroSAGE': HeteroSAGE,
        'HeteroGAT': HeteroGAT,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    model_cls = models[model_type]

    if model_type == 'HeteroGAT':
        return model_cls(in_channels, hidden_channels, out_channels,
                         num_layers, num_heads, dropout)
    else:
        return model_cls(in_channels, hidden_channels, out_channels,
                         num_layers, dropout)
