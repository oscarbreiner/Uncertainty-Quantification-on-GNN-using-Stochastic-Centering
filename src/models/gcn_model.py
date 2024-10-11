import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """
    A simple Graph Convolutional Network (GCN) model.
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
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = dropout
        self.dropout_prob = dropout_prob

    def forward(self, x, edge_index):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph edge indices.

        Returns:
            Tensor: Raw logits.
        """
        x = F.relu(self.conv1(x, edge_index))
        if self.dropout:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.conv2(x, edge_index)
        return x
