import torch
from seml import Experiment
from src.init import init_dataset, init_model, init_task
from src.training.train import train_model
from src.training.evaluate import val_model, test_ensemble
from src.data.build import apply_split_and_create_masks
from src.utils.pickl_save import save_data_to_pickle
from src.utils import EarlyStopper
from src.data.distribution_shifts import apply_feature_distribution_shift_test_time

ex = Experiment()

@ex.automain
def run(dataset, model, task, distribution_shifts, _run, seed, split_index, patience=50, min_delta=0.1, num_samples_dropout=8, inductive_setting=False):
    """
    Main function to load data, train, and evaluate the model.

    Args:
        dataset (str): The name of the dataset to use.
        model (dict): Configuration dictionary for the model.
        task (dict): Configuration dictionary for the task.
        distribution_shifts (dict): Configuration for distribution shifts.
        _run (sacred.Run): Sacred run object for logging.
        seed (int): Random seed for reproducibility.
        split_index (int): Index for data splitting.
        patience (int, optional): Epochs to wait before early stopping. Defaults to 50.
        min_delta (float, optional): Minimum change in loss for improvement. Defaults to 0.1.
        num_samples_dropout (int, optional): Number of samples for dropout testing. Defaults to 8.
        inductive_setting (bool, optional): Whether to use inductive setting. Defaults to False.

    Returns:
        dict: Test accuracy results.
    """
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Inductive Setting: {inductive_setting}')
    torch.manual_seed(seed)

    # Initialize dataset and apply distribution shifts
    data = init_dataset(dataset)
    dataset = apply_split_and_create_masks(data, distribution_shifts, num_train_masks=1)

    # Initialize model and task
    model = init_model(model, dataset.data_base.num_features, dataset.data_base.num_classes).to(device)
    optimizer, criterion, log_interval = init_task(task, model)
    early_stopper = EarlyStopper.EarlyStopper(patience=patience, min_delta=min_delta)

    # Select data and masks for training and validation
    if inductive_setting:
        train_mask, val_mask, data = dataset.train_masks_inductive[0], dataset.val_masks_inductive[0], dataset.id_data
    else:
        train_mask, val_mask, data = dataset.train_masks[0], dataset.val_masks[0], dataset.data_base[0]

    test_mask = dataset.test_mask

    # Training and validation loop
    print('Starting Training!')
    for epoch in range(task['epochs']):
        train_loss = train_model(model, data, train_mask, optimizer, criterion, _run, epoch, device)
        val_loss, val_acc = val_model(model, data, val_mask, criterion, _run, epoch, device)

        if early_stopper.early_stop(val_loss) and patience > 0:
            print(f'Early stopping at epoch {epoch+1}')
            break

        if epoch % log_interval == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Final evaluation on the test set
    print('Starting Final Evaluation on Test Set!')
    if distribution_shifts['type'] == 'feature_pertubations':
        data = apply_feature_distribution_shift_test_time(data, dataset, distribution_shifts)

    # Ensemble testing with dropout
    models = [model for _ in range(num_samples_dropout)]
    test_results = test_ensemble(models, data, test_mask, _run, device, ood_mask=dataset.ood_mask, id_mask=dataset.id_mask, dropout_test=True)

    # Save results
    save_data_to_pickle("dropout_experiment", data, test_results['logits'].cpu(), dataset.ood_mask, test_results['all_logits'].cpu())

    # Clean up results dictionary
    del test_results['logits']
    del test_results['all_logits']

    print(f'Test Accuracy: {test_results}')
    print('Training and Evaluation Completed')

    return test_results
