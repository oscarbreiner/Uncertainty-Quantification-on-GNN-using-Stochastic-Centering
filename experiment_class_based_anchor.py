import torch
from seml.experiment import Experiment
from src.init import init_dataset, init_model, init_task, calculate_ood_id_masks
from src.training.train import train_model
from src.training.evaluate import val_model_with_anchoring, test_model_with_anchoring
from src.anchoring.anchoring import initialize_class_feature_anchoring_dist
from src.data.build import apply_split_and_create_masks
import os
import torch.nn.functional as F
from src.utils import EarlyStopper
from src.utils.pickl_save import save_data_to_pickle
from src.data.distribution_shifts import apply_feature_distribution_shift_test_time
from src.utils.seed import set_seed

experiment = Experiment()

@experiment.automain
def main(dataset, model, task, distribution_shifts, _run, seed, anchoring_mode = 'Gaussian', patience=50, min_delta=0.1, number_ensemble_anchors=10, inductive_setting=False):

    """
    Main function to load data, create models, train, and evaluate the GdUQ model.

    Args:
        dataset (str): The name of the dataset to be used.
        model (dict): Configuration dictionary for the model.
        task (dict): Configuration dictionary for the task.
        distribution_shifts (dict): Configuration dictionary for the distribution_shifts.
        _run (sacred.Run): The Sacred run object for logging.
        seed (int): Random seed for reproducibility.
        anchoring_mode (str, optional): Type of anchoring used in G-dUQ. Defaults to Gaussian.
        number_ensemble_anchors: how many different anchor settings during test time. Defaults to 10.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 50.
        min_delta (float, optional): Minimum change in loss to qualify as an improvement. Defaults to 0.1.
    """
    
    dataset_config = dataset
    model_config = model
    task_config = task
    distribution_shifts_config = distribution_shifts
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')

    print(f'Using device: {device}')
    print(f'Inductive Setting: {inductive_setting}')
        
    print('Starting Node Feature G-dUQ Experiment!')
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    set_seed(seed)

    # Initialize dataset
    data = init_dataset(dataset_config)
    
    # Apply pertubations and split to data and receive Dataset
    # - Dasaset is a class containing ...
    # -- id_mask, ood_mask
    # -- train_masks, val_masks, one test_mask
    dataset = apply_split_and_create_masks(data, distribution_shifts_config, num_train_masks=1)

    data = None
    if inductive_setting:
        train_mask = dataset.train_masks_inductive[0]
        val_mask = dataset.val_masks_inductive[0]
        data = dataset.id_data
    else: 
        train_mask = dataset.train_masks[0]
        val_mask = dataset.val_masks[0]
        data = dataset.data_base[0]

    test_mask = dataset.test_mask

    # Initialize anchors
    cluster_anchoring_dist = initialize_class_feature_anchoring_dist(data,anchoring_mode, train_mask)
    node_count = data.x.shape[0]
    feature_dim = list(cluster_anchoring_dist.values())[0].mean.shape[0]
    inference_anchors = [torch.zeros(node_count, feature_dim, device=device) for _ in range(number_ensemble_anchors)]
    
    for class_label, _ in cluster_anchoring_dist.items():
        class_mask = (data.y == class_label).nonzero(as_tuple=True)[0]
        num_nodes_in_class = class_mask.size(0)
        
        for i in range(number_ensemble_anchors):
            class_anchor = cluster_anchoring_dist[class_label].sample().to(device)
            repeated_anchor = class_anchor.repeat(num_nodes_in_class, 1)
            inference_anchors[i][class_mask] = repeated_anchor

    # Initialize models that can sample from the distribution
    gcn = init_model(model_config, dataset.data_base.num_features, dataset.data_base.num_classes, cluster_anchoring_dist).to(device)

    # Initialize task
    optimizer, criterion, log_interval = init_task(task_config, gcn)
    
    early_stopper = EarlyStopper.EarlyStopper(patience=patience, min_delta=min_delta)

    # Training & Validation loop
    print('Starting Training!')
    for epoch in range(task_config['epochs']):

        train_loss = train_model(gcn, data, train_mask, optimizer, criterion, _run, epoch, device, class_based_anchor=True)
        val_loss, val_acc = val_model_with_anchoring(gcn, data, val_mask, criterion, inference_anchors, _run, epoch, device, class_based_anchor=True)
        
        if early_stopper.early_stop(val_loss) and patience > 0: 
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        if epoch % log_interval == 0:
            print(f'Epoch {epoch}, Train_Loss: {train_loss}, Val_Loss: {val_loss}, Val_Acc: {val_acc}')

    ### Not Depending on INDUCTIVE OR TRANSDUCTIVE - TESTING ON COMPLETE GRAPH ###
    
    data = dataset.data_base[0]
    if inductive_setting:
        node_count = data.x.shape[0]
        feature_dim = list(cluster_anchoring_dist.values())[0].mean.shape[0]
        inference_anchors = [torch.zeros(node_count, feature_dim, device=device) for _ in range(number_ensemble_anchors)]
        
        for class_label, _ in cluster_anchoring_dist.items():
            class_mask = (data.y == class_label).nonzero(as_tuple=True)[0]
            num_nodes_in_class = class_mask.size(0)
            
            for i in range(number_ensemble_anchors):
                class_anchor = cluster_anchoring_dist[class_label].sample().to(device)
                repeated_anchor = class_anchor.repeat(num_nodes_in_class, 1)
                inference_anchors[i][class_mask] = repeated_anchor
    
    ################################################################

    # Evaluate each model
    print(f'Starting Final Evaluation on Test Set with {len(inference_anchors)} different anchors!')

    if distribution_shifts_config['type'] == 'feature_pertubations':
        data = apply_feature_distribution_shift_test_time(data ,dataset, distribution_shifts_config)

    results = test_model_with_anchoring(gcn, data, test_mask, inference_anchors, _run, device, ood_mask=dataset.ood_mask, id_mask=dataset.id_mask, class_based_anchor=True)
    logits = results['logits']
    all_logits = results['all_logits']

    save_data_to_pickle(f"Class_Node_Feature_G-dUQ_experiment", data, logits.cpu(), dataset.ood_mask, all_logits.cpu())

    del results['logits']
    del results['all_logits']

    print(f'Test Reults: {results}')

    print('Training and Evaluation Completed')

    return results