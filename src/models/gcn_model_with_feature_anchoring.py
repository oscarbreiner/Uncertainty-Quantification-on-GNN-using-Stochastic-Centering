import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_Feature_Anchored(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model with feature anchoring.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=False, dropout_prob=0.5, anchoring_dist=None):
        """
        Initializes the GCN model.

        Args:
            in_channels (int): Number of input features.
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output classes.
            dropout (bool, optional): Whether to apply dropout. Defaults to False.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.5.
            anchoring_dist (torch.distributions.Distribution, optional): Distribution for sampling anchors.
        """
        super().__init__()
        self.conv1 = GCNConv(2 * in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.anchoring_dist = anchoring_dist

    def forward(self, x, edge_index, inferenz_anchors=None):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph edge indices.
            inferenz_anchors (Tensor, optional): Precomputed anchors for inference. Defaults to None.

        Returns:
            Tensor: Raw logits.
        """
        anchors = inferenz_anchors if inferenz_anchors is not None else self.anchoring_dist.sample((x.shape[0],)).to(x.device)
        x = torch.cat((x - anchors, anchors), dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_prob)

        x = self.conv2(x, edge_index)
        return x

class GCN_Class_Based_Feature_Anchored(torch.nn.Module):
    """
    GCN model with class-based feature anchoring.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=False, dropout_prob=0.5, class_anchoring_dist=None):
        """
        Initializes the GCN model.

        Args:
            in_channels (int): Number of input features.
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output classes.
            dropout (bool, optional): Whether to apply dropout. Defaults to False.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.5.
            class_anchoring_dist (dict, optional): Dictionary of distributions for sampling class-specific anchors.
        """
        super().__init__()
        self.conv1 = GCNConv(2 * in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.class_anchoring_dist = class_anchoring_dist

    def forward(self, x, edge_index, labels, inferenz_anchors=None):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph edge indices.
            labels (Tensor): Class labels for the nodes.
            inferenz_anchors (Tensor, optional): Precomputed anchors for inference. Defaults to None.

        Returns:
            Tensor: Raw logits.
        """
        anchors = inferenz_anchors if inferenz_anchors is not None else self.sample_class_anchors(labels, x.device)
        x = torch.cat((x - anchors, anchors), dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_prob)

        x = self.conv2(x, edge_index)
        return x

    def sample_class_anchors(self, labels, device=None):
        """
        Samples class-specific anchors for each node based on their class labels.

        Args:
            labels (Tensor): Class labels for the nodes.
            device (torch.device, optional): The device for computations.

        Returns:
            Tensor: Anchors for each node.
        """
        node_count = labels.size(0)
        feature_dim = list(self.class_anchoring_dist.values())[0].mean.size(0)
        anchors = torch.zeros(node_count, feature_dim, device=device)

        for class_label, dist in self.class_anchoring_dist.items():
            class_mask = labels == class_label
            class_anchors = dist.sample([class_mask.sum()]).to(device)
            anchors[class_mask] = class_anchors

        return anchors
