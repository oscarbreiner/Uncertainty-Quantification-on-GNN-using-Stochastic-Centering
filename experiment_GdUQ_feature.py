import torch
from seml.experiment import Experiment
from src.init import init_dataset, init_model, init_task
from src.training.train import train_model
from src.training.evaluate import val_model_with_anchoring, test_model_with_anchoring
from src.anchoring.anchoring import initialize_feature_anchoring_dist
from src.data.build import apply_split_and_create_masks
from src.utils import EarlyStopper
from src.data.distribution_shifts import apply_feature_distribution_shift_test_time
from src.utils.seed import set_seed

experiment = Experiment()

@experiment.automain
def main(dataset, model, task, distribution_shifts, _run, seed, anchoring_mode='Gaussian', patience=50, min_delta=0.1, number_ensemble_anchors=10, inductive_setting=False):
    """
    Main function to load data, create models, train, and evaluate the GdUQ model.

    Args:
        dataset (str): Name of the dataset to be used.
        model (dict): Configuration dictionary for the model.
        task (dict): Configuration dictionary for the task.
        distribution_shifts (dict): Configuration for distribution shifts.
        _run (sacred.Run): Sacred run object for logging.
        seed (int): Random seed for reproducibility.
        anchoring_mode (str, optional): Type of anchoring used in G-dUQ. Defaults to 'Gaussian'.
        patience (int, optional): Epochs to wait before early stopping. Defaults to 50.
        min_delta (float, optional): Minimum change in loss to qualify as an improvement. Defaults to 0.1.
        number_ensemble_anchors (int, optional): Number of different anchor settings during testing. Defaults to 10.
        inductive_setting (bool, optional): Whether to use inductive setting. Defaults to False.

    Returns:
        dict: Test accuracy results.
    """
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}\nInductive Setting: {inductive_setting}')

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    set_seed(seed)

    # Initialize dataset and apply distribution shifts
    data = init_dataset(dataset)
    dataset = apply_split_and_create_masks(data, distribution_shifts, num_train_masks=1)

    # Select data and masks based on inductive or transductive setting
    if inductive_setting:
        train_mask, val_mask, data = dataset.train_masks_inductive[0], dataset.val_masks_inductive[0], dataset.id_data
    else:
        train_mask, val_mask, data = dataset.train_masks[0], dataset.val_masks[0], dataset.data_base[0]

    test_mask = dataset.test_mask

    # Initialize anchors
    anchoring_dist = initialize_feature_anchoring_dist(data.x[train_mask], anchoring_mode)
    node_count = data.x.shape[0]
    inference_anchors = [anchoring_dist.sample().repeat(node_count, 1) for _ in range(number_ensemble_anchors)]

    # Initialize the model
    model = init_model(model, dataset.data_base.num_features, dataset.data_base.num_classes, anchoring_dist).to(device)

    # Initialize task
    optimizer, criterion, log_interval = init_task(task, model)
    early_stopper = EarlyStopper.EarlyStopper(patience=patience, min_delta=min_delta)

    # Training and validation loop
    print('Starting Training!')
    for epoch in range(task['epochs']):
        train_loss = train_model(model, data, train_mask, optimizer, criterion, _run, epoch, device)
        val_loss, val_acc = val_model_with_anchoring(model, data, val_mask, criterion, inference_anchors, _run, epoch, device)

        if early_stopper.early_stop(val_loss) and patience > 0:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if epoch % log_interval == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Evaluation on the test set
    data = dataset.data_base[0]
    if inductive_setting:
        node_count = data.x.shape[0]
        inference_anchors = [anchoring_dist.sample().repeat(node_count, 1) for _ in range(number_ensemble_anchors)]

    print(f'Starting Final Evaluation with {len(inference_anchors)} different anchors!')
    if distribution_shifts['type'] == 'feature_pertubations':
        data = apply_feature_distribution_shift_test_time(data, dataset, distribution_shifts)

    results = test_model_with_anchoring(model, data, test_mask, inference_anchors, _run, device, ood_mask=dataset.ood_mask, id_mask=dataset.id_mask)

    # Cleanup results dictionary
    del results['logits']
    del results['all_logits']

    print(f'Test Results: {results}')
    print('Training and Evaluation Completed')

    return results
