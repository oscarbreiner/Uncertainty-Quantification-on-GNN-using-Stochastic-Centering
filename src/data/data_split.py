import torch
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data

def sample_stratified_mask(data: Data, num_per_class=None, fraction=None, mask=None):
    """
    Creates a stratified sample mask for the dataset.

    Args:
        data (Data): A torch_geometric Data object containing node labels.
        num_per_class (int, optional): Number of samples per class. Must specify either this or 'fraction'.
        fraction (float, optional): Fraction of nodes to sample per class. Ignored if 'num_per_class' is specified.
        mask (Tensor, optional): Boolean mask indicating which nodes can be sampled.

    Returns:
        Tensor: A boolean mask indicating which nodes are sampled.
    
    Raises:
        ValueError: If neither or both of 'num_per_class' and 'fraction' are specified.
    """
    labels = data.y.numpy()
    num_classes = labels.max().item() + 1

    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        if mask is None or mask[idx]:
            class_indices[label].append(idx)

    if num_per_class is not None and fraction is None:
        samples_per_class = {class_id: num_per_class for class_id in range(num_classes)}
    elif num_per_class is None and fraction is not None:
        samples_per_class = {class_id: int(len(indices) * fraction) for class_id, indices in class_indices.items()}
    else:
        raise ValueError("Specify exactly one of 'num_per_class' or 'fraction'.")

    mask_sampled = torch.zeros(len(labels), dtype=torch.bool)
    for class_id, indices in class_indices.items():
        indices = torch.tensor(indices)
        num_samples = samples_per_class[class_id]
        sampled_indices = indices[torch.randperm(len(indices))[:num_samples]] if len(indices) > num_samples else indices
        mask_sampled[sampled_indices] = True

    return mask_sampled

def apply_dataset_split(
    base_data,
    ood_mask,
    id_mask,
    train_size: float | int,
    val_size: float | int,
    test_size: float | int = -1,
    num_train_masks: int = 1
):
    """
    Splits the dataset into stratified train, validation, and test sets, and creates the corresponding masks.

    Args:
        base_data (Data): The base graph data.
        ood_mask (Tensor): Mask indicating out-of-distribution nodes.
        id_mask (Tensor): Mask indicating in-distribution nodes.
        train_size (float | int): Number of nodes or fraction of nodes for the training set.
        val_size (float | int): Number of nodes or fraction of nodes for the validation set.
        test_size (float | int, optional): Number of nodes or fraction of nodes for the test set. Defaults to -1 (remaining nodes).
        num_train_masks (int, optional): Number of training masks to create. Defaults to 1.

    Returns:
        tuple: Lists of training masks, validation masks, and a test mask.

    Raises:
        ValueError: If 'val_size' is not an even number.
    """
    if val_size % 2 != 0:
        raise ValueError("Validation size must be an even number.")

    id_num = int(train_size + val_size)
    print(f"Number of nodes to sample for the in-distribution pool: {id_num}")

    # Create a stratified mask for the in-distribution training pool
    pool_id_mask_train = sample_stratified_mask(
        base_data,
        num_per_class=id_num,
        mask=id_mask
    )

    train_masks = []
    val_masks = []

    for _ in range(num_train_masks):
        # Generate a training mask from the pool
        train_mask = sample_stratified_mask(
            base_data,
            num_per_class=train_size,
            mask=pool_id_mask_train
        )

        # Generate a validation mask as the remaining in the pool not in the training mask
        val_mask = pool_id_mask_train & ~train_mask

        train_masks.append(train_mask)
        val_masks.append(val_mask)

    # Generate a fixed test mask based on the remaining nodes or specified size
    if test_size == -1:
        test_mask = ~pool_id_mask_train
    else:
        test_mask = sample_stratified_mask(
            base_data,
            num_per_class=test_size,
            mask=~train_mask & ~val_mask
        )

    return train_masks, val_masks, test_mask
