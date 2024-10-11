import torch
from .data_split import sample_stratified_mask
from torch.distributions import Normal, Bernoulli
from torch_geometric.data import Data
import torch.nn.functional as F

def feature_perturbation_masks(data: Data, config):
    """
    Creates in-distribution (ID) and out-of-distribution (OOD) masks based on the specified OOD fraction.

    Args:
        data (Data): The input graph data.
        config (dict): Configuration dictionary containing "ood_fraction" key.

    Returns:
        tuple: A tuple (id_mask, ood_mask) where:
            - id_mask (Tensor): Boolean mask for in-distribution nodes.
            - ood_mask (Tensor): Boolean mask for out-of-distribution nodes.
    """
    ood_mask = sample_stratified_mask(data, fraction=config["ood_fraction"])
    id_mask = ~ood_mask
    return id_mask, ood_mask

def apply_feature_distribution_shift_test_time(data, dataset, config):
    """
    Applies a feature perturbation based on the specified distribution type during test time.

    Args:
        data (Data): The input graph data.
        dataset (Dataset): The dataset object containing OOD masks.
        config (dict): Configuration dictionary with keys:
            - "distribution": Type of distribution ("normal" or "bernoulli").
            - "mean": Mean for normal distribution (used if "normal" is chosen).
            - "std": Standard deviation for normal distribution (used if "normal" is chosen).
            - "p": Probability for Bernoulli distribution (used if "bernoulli" is chosen).

    Returns:
        Data: The modified graph data with perturbed features.
    """
    if config["distribution"] == "normal":
        return apply_normal_feature_perturbation(data, dataset.ood_mask, config["mean"], config["std"])
    elif config["distribution"] == "bernoulli":
        return apply_bernoulli_feature_perturbation(data, dataset.ood_mask, config["p"])
    return None

def apply_normal_feature_perturbation(data, ood_mask, mean=0, std=1):
    """
    Applies normal distribution-based perturbation to the features of out-of-distribution nodes.

    Args:
        data (Data): The input graph data.
        ood_mask (Tensor): Boolean mask indicating out-of-distribution nodes.
        mean (float, optional): Mean for the normal distribution. Defaults to 0.
        std (float, optional): Standard deviation for the normal distribution. Defaults to 1.

    Returns:
        Data: The modified graph data with perturbed features for OOD nodes.
    """
    data.to('cpu')
    print("Applying normal distribution feature shift")
    dist = Normal(mean, std)
    perturbation = dist.sample(data.x[ood_mask].size())
    data.x[ood_mask] = perturbation
    data.x = F.normalize(data.x, p=2, dim=1)
    return data

def apply_bernoulli_feature_perturbation(data, ood_mask, p):
    """
    Applies Bernoulli distribution-based perturbation to the features of out-of-distribution nodes.

    Args:
        data (Data): The input graph data.
        ood_mask (Tensor): Boolean mask indicating out-of-distribution nodes.
        p (float): Probability for the Bernoulli distribution.

    Returns:
        Data: The modified graph data with perturbed features for OOD nodes.
    """
    print("Applying Bernoulli feature shift")
    dist = Bernoulli(p)
    perturbation = dist.sample(data.x[ood_mask].size())
    data.x[ood_mask] = perturbation
    data.x = F.normalize(data.x, p=2, dim=1)
    return data

def leave_out_class_masks(data: Data, leave_out_classes):
    """
    Creates masks by leaving out specified classes from the in-distribution set.

    Args:
        data (Data): The input graph data.
        leave_out_classes (Tensor): Tensor containing the class labels to be excluded from the in-distribution set.

    Returns:
        tuple: A tuple (id_mask, ood_mask) where:
            - id_mask (Tensor): Boolean mask for in-distribution nodes.
            - ood_mask (Tensor): Boolean mask for out-of-distribution nodes.
    """
    if len(leave_out_classes) == 1 and leave_out_classes == torch.Tensor([-1]):
        leave_out_classes = torch.Tensor([len(data.y.unique()) - 1])

    class_indices = torch.isin(data.y, leave_out_classes).logical_not()
    id_mask = class_indices
    ood_mask = ~class_indices
    return id_mask, ood_mask
