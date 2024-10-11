import torch
import yaml
from src.models.gcn_model import GCN
from src.models.gcn_model_with_hidden_layer_anchoring import GCN_Hidden_Anchored, GCN_AVG_KHop_Anchored, GCN_AVG_KHop_Anchored, GCN_AVG_Learnable_KHop_Anchored, GCN_AVG_KHop_Anchored_Shift
from src.models.gcn_model_with_feature_anchoring import GCN_Feature_Anchored, GCN_Class_Based_Feature_Anchored
from src.models.gcn_model_with_hidden_layer_anchoring_optim import GCN_Hidden_Optim_Anchored
from src.models.ensemble import EnsembleModel
from src.data.data_loader import load_dataset
import torch.nn.functional as F
from src.data.distribution_shifts import feature_pertubation_masks, leave_out_class_masks

def load_config(filepath):
    """
    Load configuration from a YAML file.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def init_dataset(dataset_name):
    """
    Initialize the dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        tuple: The dataset and the data loader.
    """
    dataset = load_dataset(dataset_name)

    return dataset

def init_model(model_config, num_features, num_classes, dist=None):
    """
    Initialize the model.

    Args:
        model_config (dict): Configuration dictionary for the model.
        num_features (int): Number of input features.
        num_classes (int): Number of output classes.

    Returns:
        list: List of initialized models.
    """
    print(model_config)
    if model_config['type'] == 'ensemble':
        if model_config['base_model']['type'] == 'gcn':
            print('### Ensemble ###')
            return EnsembleModel(GCN, model_config['num_models'], num_features, model_config['base_model']['hidden_channels'], num_classes)
        else:
            raise ValueError(f"(Work in process), Not yet supported model type: {model_config['type']}")    
    elif  model_config['type'] == 'gcn':
        return GCN(num_features, model_config['hidden_channels'], num_classes, model_config['dropout_prob'] > 0, model_config['dropout_prob'])
    elif  model_config['type'] == 'gcn_hidden':
        return GCN_Hidden_Anchored(num_features, model_config['hidden_channels'], num_classes, model_config['dropout_prob'] > 0, model_config['dropout_prob'])
    elif  model_config['type'] == 'gcn_hidden_optim':
        return GCN_Hidden_Optim_Anchored(num_features, model_config['hidden_channels'], num_classes, model_config['dropout_prob'] > 0, model_config['dropout_prob'])
    elif model_config['type'] == 'gcn_feature_anchor':
        return GCN_Feature_Anchored(num_features, model_config['hidden_channels'], num_classes, model_config['dropout_prob'] > 0, model_config['dropout_prob'], dist)
    elif model_config['type'] == 'gcn_cluster_anchored_feature':
        # dist is here dic: {class: dist}
        return GCN_Class_Based_Feature_Anchored(num_features, model_config['hidden_channels'], num_classes, model_config['dropout_prob'] > 0, model_config['dropout_prob'], dist)
    elif model_config['type'] == 'gcn_hidden_avg_k_hop':
        # dist is here dic: {class: dist}
        print(f"K-Hop: {model_config['k_hop']}")
        return GCN_AVG_KHop_Anchored(num_features, model_config['hidden_channels'], num_classes, k_hop=model_config['k_hop'], dropout = model_config['dropout_prob'] > 0, dropout_prob = model_config['dropout_prob'])
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")


def init_task(task_config, model):
    """
    Initialize the task.

    Args:
        task_config (dict): Configuration dictionary for the task.
        models (list): List of models to be used in the ensemble.

    Returns:
        tuple: The ensemble model, optimizer, and loss criterion.
    """
    if task_config['type'] == 'node_classification':
        optimizer = torch.optim.Adam(model.parameters(), lr=task_config['learning_rate'], weight_decay=task_config['weight_decay'])
        criterion = F.cross_entropy
        return optimizer, criterion, task_config['log_interval']
    else:
        raise ValueError(f"Unsupported task type: {task_config['type']}")

def init_ensemble_task(task_config, models):
    """
    Initialize the task.

    Args:
        task_config (dict): Configuration dictionary for the task.
        models (list): List of models to be used in the ensemble.

    Returns:
        tuple: The ensemble model, optimizer, and loss criterion.
    """
    if task_config['type'] == 'node_classification':
        optimizers = [torch.optim.Adam(model.parameters(), lr=task_config['learning_rate'], weight_decay=task_config['weight_decay']) for model in models]
        criterion = F.cross_entropy
        return optimizers, criterion, task_config['log_interval']
        
    else:
        raise ValueError(f"Unsupported task type: {task_config['type']}")


def calculate_ood_id_masks(config, data):
    if config['type'] == 'no_shift':
        print("No shift applied!")
        labels = data.y.numpy()
        id_mask = torch.ones(len(labels), dtype=torch.bool)
        ood_mask = torch.zeros(len(labels), dtype=torch.bool)
        return id_mask, ood_mask
    elif config['type'] == 'feature_pertubations':
        print("Applied Feature Pertubations!")
        return feature_pertubation_masks(data, config)     
    elif config['type'] == 'leave_out_class':
        print("Applied Leave Out Class!")
        return leave_out_class_masks(data, torch.Tensor(config['class']))
    else:
        raise ValueError(f"Unsupported pertubation: {config['distribution']}")


    
    
