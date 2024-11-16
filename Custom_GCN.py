import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, remove_self_loops
from torch_scatter import scatter


def glorot(tensor):
    """Initialize the tensor with glorot initialization."""
    if tensor is not None:
        nn.init.xavier_uniform_(tensor)


class CustomGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, version='base', aggr='add'):
        """
        A unified GCN layer that adapts based on the model version.

        Parameters:
            in_channels (int): Input feature dimensions.
            out_channels (int): Output feature dimensions.
            version (str): The model version ('base', 'MIv1', 'MIv2', 'MIv3', 'MIv4', 'same-0.5opp').
            aggr (str): The aggregation method ('mean', 'add', 'mul'). Ignored for 'same-0.5opp'.
        """
        # Handle MIv3-specific aggregation
        if version == 'MIv3' and aggr != 'mul':
            raise ValueError("MIv3 model only supports 'mul' aggregation.")

        super(CustomGCNLayer, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.version = version
        self.aggr = aggr

        # Define learnable parameters for each version
        if version == 'base':
            self.lin = Linear(in_channels, out_channels)
            self.bn = nn.BatchNorm1d(out_channels)
        elif version in ['MIv1', 'MIv2']:
            self.lin1 = Linear(in_channels, out_channels)
            self.lin2 = Linear(in_channels, out_channels)
            self.bn = nn.BatchNorm1d(out_channels)
            if version == 'MIv2':
                self.W2 = Parameter(torch.Tensor(out_channels, out_channels))
                self.W3 = Parameter(torch.Tensor(out_channels, out_channels))
        elif version == 'MIv3':
            self.lin1 = Linear(in_channels, out_channels)
            self.lin2 = Linear(out_channels, out_channels)
            self.W2 = Parameter(torch.Tensor(out_channels, out_channels))
        elif version == 'MIv4':
            self.lin1 = Linear(in_channels, out_channels)
            self.lin2 = Linear(in_channels, out_channels)
            self.relu = nn.ReLU()
            self.epsilon = 1e-8
        elif version == 'same-0.5opp':
            self.lin = Linear(in_channels, out_channels)
            self.bn = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()
        if hasattr(self, 'lin1'):
            self.lin1.reset_parameters()
        if hasattr(self, 'lin2'):
            self.lin2.reset_parameters()
        if hasattr(self, 'bn'):
            self.bn.reset_parameters()
        if hasattr(self, 'W2'):
            glorot(self.W2)
        if hasattr(self, 'W3'):
            glorot(self.W3)

    def forward(self, x, edge_index, edge_weight=None, team_labels=None, alpha=0.5):
        """
        Forward pass for the GCN layer.

        For 'same-0.5opp', team_labels and alpha are required.
        """
        # Handle masking for padded edges
        mask = edge_index[0] != -1
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask] if edge_weight is not None else None

        # Optionally remove self-loops
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if self.version == 'base':
            x = self.lin(x)
            x = x.view(-1, self.lin.out_features)
            x = self.bn(x)
            x = x.view(-1, x.size(1))
        elif self.version in ['MIv1', 'MIv2']:
            h = self.lin2(x)
            x = self.lin1(x)
            x = x.view(-1, self.lin1.out_features)
            x = self.bn(x)
            x = x.view(-1, x.size(1))
        elif self.version == 'MIv3':
            h = self.lin1(x)
            x = self.lin2(h)
            x = x.view(-1, self.lin2.out_features)
        elif self.version == 'MIv4':
            h1 = self.relu(self.lin1(x))
            return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, h1=h1, edge_weight=edge_weight)
        elif self.version == 'same-0.5opp':
            x = self.lin(x)
            x = x.view(-1, self.lin.out_features)
            x = self.bn(x)
            x = x.view(-1, x.size(1))
            return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_weight=edge_weight, team_labels=team_labels, alpha=alpha)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, h=h if self.version in ['MIv1', 'MIv2'] else None, edge_weight=edge_weight)

    def message(self, x_j, edge_index, size, edge_weight, team_labels=None, alpha=0.5):
        """
        Compute messages for edges.
        """
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        norm = deg[row].pow(-0.5) * deg[col].pow(-0.5)

        if edge_weight is not None:
            edge_weight = edge_weight.view(-1, 1)
            edge_weight = 1.0 / edge_weight
            edge_weight[torch.isinf(edge_weight)] = 0
            norm = norm * edge_weight.squeeze()

        if self.version == 'same-0.5opp':
            same_team_mask = team_labels[row] == team_labels[col]
            same_team_contrib = torch.where(same_team_mask.view(-1, 1), norm.view(-1, 1) * x_j, torch.zeros_like(x_j))
            opposing_team_contrib = torch.where(~same_team_mask.view(-1, 1), norm.view(-1, 1) * x_j, torch.zeros_like(x_j))
            return same_team_contrib, opposing_team_contrib

        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None, alpha=0.5):
        """
        Customize aggregation for 'same-0.5opp'.
        """
        if self.version == 'same-0.5opp':
            same_team_contrib, opposing_team_contrib = inputs

            # Ensure the aggregated outputs have the same shape as the input features (num_nodes x out_channels)
            same_team_aggr = torch.zeros((dim_size, same_team_contrib.size(-1)), device=same_team_contrib.device)
            opposing_team_aggr = torch.zeros((dim_size, opposing_team_contrib.size(-1)), device=opposing_team_contrib.device)

            same_team_aggr.scatter_add_(0, index.unsqueeze(-1).expand_as(same_team_contrib), same_team_contrib)
            opposing_team_aggr.scatter_add_(0, index.unsqueeze(-1).expand_as(opposing_team_contrib), opposing_team_contrib)

            opposing_team_aggr = alpha * opposing_team_aggr

            return same_team_aggr - opposing_team_aggr

        return super(CustomGCNLayer, self).aggregate(inputs, index, dim_size=dim_size)

    def update(self, aggr_out, x=None, h=None, h1=None):
        """
        Update node features after aggregation.
        """
        if self.version == 'base' or self.version == 'same-0.5opp':
            return aggr_out
        elif self.version in ['MIv1', 'MIv2']:
            return aggr_out * x
        elif self.version == 'MIv2':
            return h + torch.matmul(aggr_out, self.W2) + torch.matmul(x * aggr_out, self.W3)
        elif self.version == 'MIv3':
            return h * aggr_out
        elif self.version == 'MIv4':
            return h1 * self.lin2(aggr_out)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, version={self.version}, aggr={self.aggr})"
