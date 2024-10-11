import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_Hidden_Optim_Anchored(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model with hidden optimization and anchoring.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=False, dropout_prob=0.5):
        """
        Initializes the GCN model.

        Args:
            in_channels (int): Number of input features.
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output classes.
            dropout (bool, optional): Whether to apply dropout. Defaults to False.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.5.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(2 * hidden_channels, out_channels)
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.batch_norm = torch.nn.BatchNorm1d(2 * hidden_channels)
        self.mean = torch.nn.Parameter(torch.zeros(hidden_channels) + 1e-6)
        self.log_std_dev = torch.nn.Parameter(torch.zeros(hidden_channels) + 1e-6)

    def freeze_distribution(self):
        """Freezes the distribution parameters (mean and log_std_dev)."""
        self.mean.requires_grad = False
        self.log_std_dev.requires_grad = False

    def unfreeze_distribution(self):
        """Unfreezes the distribution parameters (mean and log_std_dev)."""
        self.mean.requires_grad = True
        self.log_std_dev.requires_grad = True

    def freeze_gnn(self):
        """Freezes the GCN parameters."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False

    def unfreeze_gnn(self):
        """Unfreezes the GCN parameters."""
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.conv2.parameters():
            param.requires_grad = True

    def forward(self, x, edge_index):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph edge indices.

        Returns:
            Tensor: Raw logits.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_prob)

        # Sample from the distribution and apply anchoring
        epsilon = torch.randn(x.size(0), self.hidden_channels, device=x.device)
        std_dev = torch.exp(self.log_std_dev)
        c = std_dev * epsilon + self.mean

        # Combine the original features with the anchored features
        x = torch.cat((x - c, c), dim=1)
        x = self.conv2(x, edge_index)
        return x
