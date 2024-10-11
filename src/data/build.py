from copy import deepcopy
from src.init import calculate_ood_id_masks
from .data_split import apply_dataset_split
from .data_loader import Dataset

def apply_split_and_create_masks(base_data, config, num_train_masks=1):
    """
    Applies dataset splitting and creates masks for training, validation, and testing.

    Args:
        base_data: The base dataset to be split.
        config: Configuration parameters for calculating masks.
        num_train_masks (int, optional): The number of training masks to create. Defaults to 1.

    Returns:
        Dataset: The dataset object with split masks and data.
    """
    id_mask, ood_mask = calculate_ood_id_masks(config, base_data)
    
    train_masks, val_masks, test_mask = apply_dataset_split(
        base_data,
        ood_mask,
        id_mask,
        train_size=20,
        val_size=20,
        test_size=-1,
        num_train_masks=num_train_masks
    )
    
    dataset = Dataset(id_mask, ood_mask, base_data, train_masks, val_masks, test_mask)
    
    return dataset

