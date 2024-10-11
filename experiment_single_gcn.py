import torch
from seml.experiment import Experiment
from src.init import init_dataset, init_model, init_task
from src.data.build import apply_split_and_create_masks
from src.training.train import train_model
from src.training.evaluate import val_model, test_model
from src.utils import EarlyStopper
from src.data.distribution_shifts import apply_feature_distribution_shift_test_time

experiment = Experiment()

@experiment.automain
def main(dataset, model, task, distribution_shifts, _run, seed, patience=50, min_delta=0.1):
    """
    Main function to load data, create models, train, and evaluate a GCN model.

    Args:
        dataset (str): Name of the dataset to be used.
        model (dict): Configuration dictionary for the model.
        task (dict): Configuration dictionary for the task.
        distribution_shifts (dict): Configuration dictionary for distribution shifts.
        _run (sacred.Run): Sacred run object for logging.
        seed (int): Random seed for reproducibility.
        patience (int, optional): Epochs to wait before early stopping. Defaults to 50.
        min_delta (float, optional): Minimum change in loss to qualify as an improvement. Defaults to 0.1.

    Returns:
        float: Test accuracy of the model.
    """
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}\nStarting Single GCN Experiment!')

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Initialize dataset and apply distribution shifts
    data = init_dataset(dataset)
    dataset = apply_split_and_create_masks(data, distribution_shifts, num_train_masks=1)
    data = data[0]

    # Initialize model and task
    model = init_model(model, dataset.data_base.num_features, dataset.data_base.num_classes).to(device)
    optimizer, criterion, log_interval = init_task(task, model)
    early_stopper = EarlyStopper.EarlyStopper(patience=patience, min_delta=min_delta)

    train_mask = dataset.train_masks[0]
    val_mask = dataset.val_masks[0]
    test_mask = dataset.test_mask

    # Training and validation loop
    print('Starting Training!')
    for epoch in range(task['epochs']):
        train_loss = train_model(model, data, train_mask, optimizer, criterion, _run, epoch, device)
        val_loss, val_acc = val_model(model, data, val_mask, criterion, _run, epoch, device)

        # Early stopping check
        if early_stopper.early_stop(val_loss) and patience > 0:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        # Logging interval
        if epoch % log_interval == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Final evaluation on the test set
    print('Starting Final Evaluation on Test Set!')
    if distribution_shifts['type'] == 'feature_pertubations':
        data = apply_feature_distribution_shift_test_time(data, dataset, distribution_shifts)

    # Test the model
    test_acc = test_model(model, data, test_mask, _run, device, ood_mask=dataset.ood_mask, id_mask=dataset.id_mask)
    print(f'Test Accuracy: {test_acc}')
    print('Training and Evaluation Completed')

    return test_acc
