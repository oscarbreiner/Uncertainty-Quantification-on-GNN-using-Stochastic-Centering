import torch
import torch.nn as nn


class ExpectedCalibrationErrorLoss(nn.Module):
    def __init__(self, num_bins=10):
        super(ExpectedCalibrationErrorLoss, self).__init__()
        self.num_bins = num_bins
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, outputs, targets, reduction = "mean"):
        """
        Calculate Expected Calibration Error (ECE) as a loss function.
        
        Args:
        - outputs (tensor): Predicted probabilities from the model of shape (batch_size, num_classes).
        - targets (tensor): True labels of shape (batch_size,).
        
        Returns:
        - ece_loss (tensor): ECE loss value.
        """
        return self.ece(outputs, targets, self.num_bins)
    
    def ece(self, predictions, labels, n_bins=10):
        """
        Compute the Expected Calibration Error (ECE) for softmax predictions using PyTorch tensors.

        Parameters:
        - predictions: logits
        - labels: torch tensor of shape (num_samples,) with true class labels (integers)
        - n_bins: number of bins to use

        Returns:
        - ece: Expected Calibration Error
        """
        # Ensure predictions and labels are torch tensors
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        labels = labels.type(torch.int64)
        
        # Number of samples
        num_samples = predictions.size(0)
        
        # Get the predicted probabilities for the true class
        true_class_probs = predictions[torch.arange(num_samples), labels]
        
        # Create the bin boundaries
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        # Initialize variables to keep track of the weighted sum of absolute differences
        ece = 0.0
        total_count = 0
        
        # Loop over each bin
        for i in range(n_bins):
            # Find the indices of predictions that fall into the current bin
            bin_indices = torch.where((true_class_probs > bin_boundaries[i]) & (true_class_probs <= bin_boundaries[i + 1]))[0]
            
            if bin_indices.size(0) > 0:
                # Calculate the average predicted probability for the current bin
                avg_pred_prob = torch.mean(true_class_probs[bin_indices])
                # Calculate the accuracy for the current bin
                accuracy = torch.mean((labels[bin_indices] == torch.argmax(predictions[bin_indices], dim=1)).float())
                # Calculate the proportion of predictions in the current bin
                bin_count = bin_indices.size(0)
                total_count += bin_count
                
                # Update the ECE with the weighted absolute difference for the current bin
                ece += bin_count * torch.abs(avg_pred_prob - accuracy)
        
        # Check if total_count is zero to avoid division by zero
        if total_count > 0:
            ece /= total_count
        else:
            ece = torch.tensor(0.0)  # or some indicative value, e.g., torch.tensor(float('nan'))
        
        return ece
