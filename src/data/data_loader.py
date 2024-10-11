import torch
from torch_geometric.datasets import Planetoid, WebKB, Amazon, Coauthor
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import subgraph
import copy

def safe_load(dataset_class, root, name, transform=None):
    """
    Safely loads a dataset and handles potential loading errors.

    Args:
        dataset_class: The class of the dataset to load.
        root (str): The root directory for dataset storage.
        name (str): The name of the dataset.
        transform (optional): A transform to apply to the dataset.

    Returns:
        Dataset or None: Loaded dataset object or None if loading fails.
    """
    try:
        return dataset_class(root=root, name=name, transform=transform)
    except Exception as e:
        print(f"Error loading dataset {name}: {e}")
        return None

def load_dataset(name):
    """
    Loads a specified dataset and displays basic statistics.

    Args:
        name (str): The name of the dataset to load.

    Returns:
        torch_geometric.data.Data: The loaded dataset.

    Raises:
        ValueError: If the dataset name is unsupported.
    """
    root = 'datasets'

    if name == 'Cora':
        base_dataset = safe_load(Planetoid, root, 'Cora', NormalizeFeatures())
    elif name == 'PubMed':
        base_dataset = safe_load(Planetoid, root, 'PubMed', NormalizeFeatures())
    elif name == 'Citeseer':
        base_dataset = safe_load(Planetoid, root, name, NormalizeFeatures())
    elif name in ['Coauthor-Physics', 'Coauthor-Computers']:
        base_dataset = safe_load(Coauthor, root, name.split('-')[-1], NormalizeFeatures())
    elif name in ['Amazon-Photo', 'Amazon-Computer']:
        base_dataset = safe_load(Amazon, root, name.split('-')[-1], NormalizeFeatures())
    else:
        raise ValueError(f"Dataset {name} is not supported.")

    if base_dataset is None:
        raise ValueError(f"Dataset {name} could not be loaded.")

    data = base_dataset[0]

    print(f"Dataset Name: {name}")
    print(f"Number of graphs: {len(base_dataset)}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {base_dataset.num_classes}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")

    return base_dataset

class Dataset:
    """A dataset containing in-distribution and out-of-distribution data."""

    def __init__(self, id_mask, ood_mask, data_base, train_masks=None, val_masks=None, test_mask=None):
        """
        Initializes the dataset with masks for training, validation, and testing.

        Args:
            id_mask (Tensor): Mask for in-distribution data.
            ood_mask (Tensor): Mask for out-of-distribution data.
            data_base (torch_geometric.data.Data): The base graph data.
            train_masks (list, optional): List of training masks.
            val_masks (list, optional): List of validation masks.
            test_mask (Tensor, optional): Test mask.
        """
        self.data_base = data_base
        self.id_mask = id_mask
        self.ood_mask = ood_mask
        self.num_train_masks = len(train_masks) if train_masks is not None else 0
        self.train_masks = train_masks
        self.val_masks = val_masks
        self.test_mask = test_mask

        # Prepare in-distribution data
        self.id_data = copy.deepcopy(data_base[0])
        self.id_data.edge_index, _ = subgraph(id_mask, self.id_data.edge_index, relabel_nodes=True)
        self.id_data.x = self.id_data.x[id_mask]
        self.id_data.y = self.id_data.y[id_mask]

        # Transfer masks to subset
        old_node_keys = torch.arange(data_base[0].num_nodes)
        new_node_keys = torch.arange(len(self.id_data.x))
        self.train_masks_inductive = self._transfer_masks(train_masks, old_node_keys, new_node_keys)
        self.val_masks_inductive = self._transfer_masks(val_masks, old_node_keys, new_node_keys)

    @staticmethod
    def _transfer_masks(masks, old_node_keys, new_node_keys):
        """
        Transfers masks to a new set of node indices.

        Args:
            masks (list): List of masks to transfer.
            old_node_keys (Tensor): Original node indices.
            new_node_keys (Tensor): New node indices.

        Returns:
            list: Transferred masks for the new node indices.
        """
        if masks is None:
            return None

        key_to_idx_src = {key.item(): idx for idx, key in enumerate(old_node_keys.tolist())}
        return [
            torch.tensor(
                [mask[key_to_idx_src[key]] if key in key_to_idx_src else False for key in new_node_keys.tolist()],
                dtype=torch.bool,
                device=masks[0].device
            ) for mask in masks
        ]

    def print_summary(self):
        """Prints a summary of the dataset."""
        print('Dataset summary')
        print(f'Number of training masks: {self.num_train_masks}')
        for i, mask in enumerate(self.train_masks or []):
            print(f'Train Mask {i + 1}: size {mask.sum()}')

        if self.val_masks:
            for i, mask in enumerate(self.val_masks):
                print(f'Val Mask {i + 1}: size {mask.sum()}')
        else:
            print('No validation masks')

        if self.test_mask:
            print(f'Test Mask: size {self.test_mask.sum()}')
        else:
            print('No test mask')
