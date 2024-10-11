import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_Hidden_Anchored(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model with hidden feature anchoring.
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
            x = F.dropout(x, p=self.dropout_prob)

        # Randomly permute the node features for anchoring
        c = x[torch.randperm(x.size(0))]
        x = torch.cat((x - c, c), dim=1)
        x = self.conv2(x, edge_index)
        return x

class GCN_AVG_KHop_Anchored(torch.nn.Module):
    """
    GCN model that anchors features using the average of the k-hop neighborhood.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, k_hop, dropout=False, dropout_prob=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(2 * hidden_channels, out_channels)
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.k_hop = k_hop
        self.k_hop_adj = None

    def precompute_k_hop_adj(self, edge_index, num_nodes):
        # Construct k-hop adjacency matrix for averaging
        adj_matrix = self._create_adjacency_matrix(edge_index, num_nodes)
        k_hop_adj = adj_matrix.clone()
        for _ in range(1, self.k_hop):
            k_hop_adj = (torch.sparse.mm(k_hop_adj, adj_matrix) > 0).float()
        self.k_hop_adj = self._normalize_adj(k_hop_adj, num_nodes)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        if self.k_hop_adj is None or self.k_hop_adj.device != x.device:
            self.precompute_k_hop_adj(edge_index, num_nodes)

        x = F.relu(self.conv1(x, edge_index))
        if self.dropout:
            x = F.dropout(x, p=self.dropout_prob)

        # Compute k-hop anchored features
        k_hop_features = torch.sparse.mm(self.k_hop_adj, x)
        x = torch.cat((x - k_hop_features, k_hop_features), dim=1)
        return self.conv2(x, edge_index)

    @staticmethod
    def _create_adjacency_matrix(edge_index, num_nodes):
        i, j = edge_index
        values = torch.ones(i.size(0), dtype=torch.float32)
        return torch.sparse.FloatTensor(torch.stack((i, j)), values, torch.Size([num_nodes, num_nodes]))

    @staticmethod
    def _normalize_adj(adj, num_nodes):
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        normalization = torch.sparse.FloatTensor(
            torch.stack((torch.arange(num_nodes), torch.arange(num_nodes))), 1.0 / row_sum, torch.Size([num_nodes, num_nodes])
        )
        return torch.sparse.mm(normalization, adj)

class GCN_AVG_KHop_Anchored_Shift(GCN_AVG_KHop_Anchored):
    """
    GCN model that anchors features using the average of the k-hop neighborhood, cycling through different k values.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=False, dropout_prob=0.5):
        super().__init__(in_channels, hidden_channels, out_channels, 3, dropout, dropout_prob)
        self.k_hop_adjs = {}
        self.current_k = 3

    def precompute_k_hop_adj(self, edge_index, num_nodes, ks=[3, 4, 5]):
        adj_matrix = self._create_adjacency_matrix(edge_index, num_nodes)
        for k in ks:
            k_hop_adj = adj_matrix.clone()
            for _ in range(1, k):
                k_hop_adj = (torch.sparse.mm(k_hop_adj, adj_matrix) > 0).float()
            self.k_hop_adjs[k] = self._normalize_adj(k_hop_adj, num_nodes)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        if not self.k_hop_adjs or self.k_hop_adjs[self.current_k].device != x.device:
            self.precompute_k_hop_adj(edge_index, num_nodes)

        self.k_hop_adj = self.k_hop_adjs[self.current_k]
        output = super().forward(x, edge_index)

        # Cycle k value for the next forward pass
        self.current_k = 3 + (self.current_k % 3)
        return output

class GCN_AVG_Learnable_KHop_Anchored(GCN_AVG_KHop_Anchored):
    """
    GCN model with learnable k-hop parameter for adaptive feature anchoring.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=False, dropout_prob=0.5, initial_k=3.0, max_k_hop=8):
        super().__init__(in_channels, hidden_channels, out_channels, int(initial_k), dropout, dropout_prob)
        self.learnable_k = torch.nn.Parameter(torch.tensor([initial_k]))
        self.max_k_hop = max_k_hop
        self.last_computed_k = None

    def precompute_k_hop_adj(self, edge_index, num_nodes):
        current_k = int(torch.round(self.learnable_k).clamp(1, self.max_k_hop).item())
        if current_k == self.last_computed_k:
            return
        self.last_computed_k = current_k
        super().precompute_k_hop_adj(edge_index, num_nodes)

class GCN_AVG_ADAPTIVE_KHop_Anchored(GCN_AVG_KHop_Anchored):
    """
    GCN model that anchors features based on adaptive k-hop values derived from node degrees.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, max_k_hop, dropout=False, dropout_prob=0.5):
        super().__init__(in_channels, hidden_channels, out_channels, max_k_hop, dropout, dropout_prob)
        self.max_k_hop = max_k_hop

    def precompute_k_hop_adj(self, edge_index, num_nodes):
        adj_matrix = self._create_adjacency_matrix(edge_index, num_nodes)
        degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        adaptive_k = torch.clamp(degrees / degrees.mean(), 1, self.max_k_hop).int()

        k_hop_adj = torch.eye(num_nodes)
        power_adj = adj_matrix.clone()

        for _ in range(self.max_k_hop):
            power_adj = torch.sparse.mm(power_adj, adj_matrix)
            increment_adj = (power_adj.to_dense() > 0).float()
            k_hop_adj += increment_adj.to_sparse()

        k_hop_adj = (k_hop_adj > 0).float()

        # Mask with adaptive k values
        for idx in range(num_nodes):
            k_hop_adj[idx] *= (torch.arange(num_nodes) <= adaptive_k[idx]).float()

        self.k_hop_adj = self._normalize_adj(k_hop_adj, num_nodes)
