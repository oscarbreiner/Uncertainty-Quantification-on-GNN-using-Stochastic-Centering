import torch
from seml.experiment import Experiment
from src.init import init_dataset, init_model, init_task
from src.training.train import train_model
from src.training.evaluate import val_ensemble, test_ensemble
from src.data.build import apply_split_and_create_masks
from src.utils import EarlyStopper
from src.utils.pickl_save import save_data_to_pickle
from src.data.distribution_shifts import apply_feature_distribution_shift_test_time

experiment = Experiment()

@experiment.automain
def main(dataset, model, task, distribution_shifts, _run, seed, patience=50, min_delta=0.1, number_anchors=10, inductive_setting=False):
    """
    Main function to load data, create models, train, and evaluate the GdUQ model.

    Args:
        dataset (str): Name of the dataset to be used.
        model (dict): Configuration dictionary for the model.
        task (dict): Configuration dictionary for the task.
        distribution_shifts (dict): Configuration dictionary for distribution shifts.
        _run (sacred.Run): Sacred run object for logging.
        seed (int): Random seed for reproducibility.
        patience (int, optional): Epochs to wait before early stopping. Defaults to 50.
        min_delta (float, optional): Minimum change in loss for improvement. Defaults to 0.1.
        number_anchors (int, optional): Number of different anchor settings during testing. Defaults to 10.
        inductive_setting (bool, optional): Whether to use inductive setting. Defaults to False.

    Returns:
        dict: Test accuracy results.
    """
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}\nInductive Setting: {inductive_setting}')

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Initialize dataset and apply distribution shifts
    data = init_dataset(dataset)
    dataset = apply_split_and_create_masks(data, distribution_shifts, num_train_masks=1)

    # Select data and masks based on inductive or transductive setting
    if inductive_setting:
        train_mask, val_mask, data = dataset.train_masks_inductive[0], dataset.val_masks_inductive[0], dataset.id_data
    else:
        train_mask, val_mask, data = dataset.train_masks[0], dataset.val_masks[0], dataset.data_base[0]

    test_mask = dataset.test_mask

    # Initialize model and task
    model = init_model(model, dataset.data_base.num_features, dataset.data_base.num_classes).to(device)
    optimizer, criterion, log_interval = init_task(task, model)
    early_stopper = EarlyStopper.EarlyStopper(patience=patience, min_delta=min_delta)

    # Training and validation loop
    print('Starting Training!')
    for epoch in range(task['epochs']):
        # Train the model
        train_loss = train_model(model, data, train_mask, optimizer, criterion, _run, epoch, device)

        # Validate the model using an ensemble
        models = [model for _ in range(number_anchors)]
        val_loss, val_acc = val_ensemble(models, data, val_mask, criterion, _run, epoch, device)

        # Early stopping check
        if early_stopper.early_stop(val_loss) and patience > 0:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        # Logging interval
        if epoch % log_interval == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Final evaluation on the test set
    data = dataset.data_base[0]
    print(f'Starting Final Evaluation on Test Set with {number_anchors} different anchors!')
    if distribution_shifts['type'] == 'feature_pertubations':
        data = apply_feature_distribution_shift_test_time(data, dataset, distribution_shifts)

    # Test the model using an ensemble
    models = [model for _ in range(number_anchors)]
    results = test_ensemble(models, data, test_mask, _run, device, ood_mask=dataset.ood_mask, id_mask=dataset.id_mask)

    # Save results
    save_data_to_pickle("Hidden_Layers_G-dUQ_experiment", data, results['logits'].cpu(), dataset.ood_mask, results['all_logits'].cpu())

    # Cleanup results dictionary
    del results['logits']
    del results['all_logits']

    print(f'Test Results: {results}')
    print('Training and Evaluation Completed')

    return results
