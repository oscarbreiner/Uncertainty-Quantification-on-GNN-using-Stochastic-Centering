import torch
from seml.experiment import Experiment
from src.init import init_dataset, init_model, init_ensemble_task
from src.training.train import train_model
from src.training.evaluate import test_ensemble, val_model
from src.data.build import apply_split_and_create_masks
from src.data.distribution_shifts import apply_feature_distribution_shift_test_time
from src.utils import EarlyStopper
from src.utils.pickl_save import save_data_to_pickle
from src.utils.seed import set_seed

experiment = Experiment()

@experiment.automain
def run(dataset, model, task, distribution_shifts, _run, seed, num_model_ensemble=8, patience=50, min_delta=0.1, inductive_setting=False):
    """
    Main function to load data, train, and evaluate an ensemble model.

    Args:
        dataset (str): The name of the dataset to be used.
        model (dict): Configuration for the model.
        task (dict): Configuration for the task.
        distribution_shifts (dict): Configuration for the distribution shifts.
        _run (sacred.Run): Sacred run object for logging.
        seed (int): Random seed for reproducibility.
        num_model_ensemble (int, optional): Number of models in the ensemble. Defaults to 8.
        patience (int, optional): Epochs to wait for improvement before early stopping. Defaults to 50.
        min_delta (float, optional): Minimum change in loss for improvement. Defaults to 0.1.
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
    dataset = apply_split_and_create_masks(data, distribution_shifts, num_train_masks=num_model_ensemble)

    # Set up data and masks based on inductive or transductive setting
    if inductive_setting:
        train_masks, val_masks, data = dataset.train_masks_inductive, dataset.val_masks_inductive, dataset.id_data
    else:
        train_masks, val_masks, data = dataset.train_masks, dataset.val_masks, dataset.data_base[0]

    test_mask = dataset.test_mask

    # Initialize ensemble training task
    _, criterion, log_interval = init_ensemble_task(task, [])

    # Train and validate each model
    models = []
    for idx in range(num_model_ensemble):
        print(f'Starting Training for Model {idx + 1}!')
        model_instance = init_model(model, dataset.data_base.num_features, dataset.data_base.num_classes).to(device)
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=task['learning_rate'], weight_decay=task['weight_decay'])
        early_stopper = EarlyStopper.EarlyStopper(patience=patience, min_delta=min_delta)

        for epoch in range(task['epochs']):
            train_loss = train_model(model_instance, data, train_masks[idx], optimizer, criterion, _run, epoch, device)
            val_loss, val_acc = val_model(model_instance, data, val_masks[idx], criterion, _run, epoch, device)

            if early_stopper.early_stop(val_loss) and patience > 0:
                print(f'Early stopping at epoch {epoch + 1}')
                break

            if epoch % log_interval == 0:
                print(f'Epoch {epoch}, Model {idx + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        models.append(model_instance)

    # Evaluate the ensemble on the test set
    print('Starting Final Evaluation on Test Set!')
    if distribution_shifts['type'] == 'feature_pertubations':
        data = apply_feature_distribution_shift_test_time(data, dataset, distribution_shifts)

    test_results = test_ensemble(models, data, test_mask, _run, device, ood_mask=dataset.ood_mask, id_mask=dataset.id_mask)

    # Save results
    save_data_to_pickle("ensemble_experiment", data, test_results['logits'].cpu(), dataset.ood_mask, test_results['all_logits'].cpu())

    # Clean up results dictionary
    del test_results['logits']
    del test_results['all_logits']

    print(f'Test Accuracy: {test_results}')
    print('Training and Evaluation Completed for All Models')

    return test_results
