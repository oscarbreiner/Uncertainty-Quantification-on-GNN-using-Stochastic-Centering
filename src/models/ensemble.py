import torch
import torch.nn.functional as F

class EnsembleModel(torch.nn.Module):
    """
    An ensemble of for example GCN models.
    """
    def __init__(self, model_class, num_models, in_channels, hidden_channels, out_channels):
        """
        Initializes the ensemble of GCN models.

        Args:
            model_class (nn.module): For Example GCN model.
            num_models (int): Number of GCN models in the ensemble.
            in_channels (int): Number of input features for each model.
            hidden_channels (int): Number of hidden units for each model.
            out_channels (int): Number of output classes for each model.
        """
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList([
            model_class(in_channels, hidden_channels, out_channels) for _ in range(num_models)
        ])

    def forward(self, x, edge_index):
        """
        Forward pass through the ensemble.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph edge indices.

        Returns:
            Tensor: Final logits after averaging predictions from all models.
        """
        outputs = [model(x, edge_index) for model in self.models]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        uncertainty = torch.std(torch.stack(outputs), dim=0)
        return F.log_softmax(avg_output, dim=1), uncertainty
