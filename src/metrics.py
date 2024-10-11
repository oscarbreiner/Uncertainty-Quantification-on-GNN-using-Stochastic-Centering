import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score



def aleatoric_measure(logits, labels):
    """
    Calculates the aleatoric measure for uncertainty estimation on graphs.

    Args:
        logits (torch.Tensor): The predicted logits from the model.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        float: The area under the ROC curve (ROC AUC) for the aleatoric scores.
    """
    logits = torch.nn.functional.softmax(logits, dim=1)
    aleatoric_scores = -logits.max(1)[0] + 1e-10
    fpr_alea, tpr_alea, _ = roc_curve(labels, aleatoric_scores)
    roc_auc_alea = auc(fpr_alea, tpr_alea)
    return roc_auc_alea

def aleatoric_measure_entropy(all_probs, labels):
    """
    Calculates the aleatoric measure for uncertainty estimation on graphs.

    Args:
        all_probs (torch.Tensor): The predicted probs from the models.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        float: The area under the ROC curve (ROC AUC) for the aleatoric scores.
    """
    aleatoric_scores = torch.mean(torch.sum(-all_probs * torch.log(all_probs + 1e-10), axis=2), axis=0)  # (num_samples,)
    fpr_alea, tpr_alea, _ = roc_curve(labels, aleatoric_scores)
    roc_auc_alea = auc(fpr_alea, tpr_alea)
    return roc_auc_alea

def epistemic_measure(logits, labels, uncertainty):
    """
    Calculate the epistemic measure for uncertainty estimation on graphs.
    NOTE: This only works for non-dirichtlet models

    Args:
        logits (torch.Tensor): The predicted logits from the model.
        labels (torch.Tensor): The true labels for the data.
        uncertainty (torch.Tensor): Uncertainty measure

    Returns:
        float: The area under the ROC curve (ROC AUC) for the epistemic scores.
    """
    probs = torch.nn.functional.softmax(logits, dim=1)
    predicted_classes = probs.max(1)[1]

    row_indices = torch.arange(uncertainty.size(0))
    epistemic_scores = uncertainty[row_indices, predicted_classes] + 1e-10

    fpr_epist, tpr_epist, _ = roc_curve(labels, epistemic_scores)
    roc_auc_epist = auc(fpr_epist, tpr_epist)
    return roc_auc_epist

def epistemic_measure_mutual_information(all_probs, labels):
    """
    Calculate the epistemic measure for uncertainty estimation on graphs.
    NOTE: This only works for non-dirichtlet models

    Args:
        all_probs (torch.Tensor): Output of all models in the ensemble (or other models)
        labels (torch.Tensor): The true labels for the data.
        
    Returns:
        float: The area under the ROC curve (ROC AUC) for the epistemic scores.
    """

    #num_stochastic_passes, num_samples, num_classes = all_probs.shape

    # Calculate the mean probabilities across the ensemble
    mean_probs = torch.mean(all_probs, dim=0)
    
    # Calculate entropy of the mean probabilities
    predictive_entropy = torch.sum(-mean_probs * torch.log(mean_probs + 1e-10), dim=1)
    
    # Calculate expected entropy
    # Calculate the entropy for each model's prediction
    individual_entropies = torch.sum(-all_probs * torch.log(all_probs + 1e-10), dim=2)
    
    # Average these entropies across the ensemble
    expected_entropy = torch.mean(individual_entropies, dim=0)
    
    # Mutual Information (Epistemic uncertainty)
    mi = predictive_entropy - expected_entropy
    mutual_information_np = mi.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    roc_auc = roc_auc_score(labels_np, mutual_information_np)

    #fpr_epist, tpr_epist, _ = roc_curve(labels, mi)

    #roc_auc_epist = auc(fpr_epist, tpr_epist)
    return roc_auc


def accuracy(prediction, labels):
    """
    Calculates the accuracy of a prediction given the true labels.

    Args:
        prediction (torch.Tensor): The predicted labels.
        labels (torch.Tensor): The true labels.

    Returns:
        float: The accuracy of the prediction.
    """
    correct = prediction.eq(labels).sum().item()
    total = len(labels)
    acc = correct / total
    return acc

def ece(predictions, labels, n_bins=10):
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
    
    return ece.item()


def brier_score(predictions, targets):
    """
    Computes the Brier score for probabilistic predictions.

    Args:
    predictions (torch.Tensor): Predicted probabilities for the positive class. 
                                Shape should be (n_samples,).
    targets (torch.Tensor): True binary labels. Shape should be (n_samples,).

    Returns:
    torch.Tensor: The Brier score.
    """
    predictions = torch.nn.functional.softmax(predictions, dim=1)

    row_indices = torch.arange(predictions.size(0))
    predictions = predictions[row_indices, targets]
    
    # Compute the Brier score
    brier = torch.mean((predictions - targets) ** 2)
    
    return brier.item()




