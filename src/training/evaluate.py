import torch
from src.metrics import accuracy, ece, aleatoric_measure, epistemic_measure, brier_score, aleatoric_measure_entropy, epistemic_measure_mutual_information

def val_model(model, data, mask, loss_func, run=None, epoch=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data (torch_geometric.data.Data): The graph data.
        mask (Tensor): Boolean mask indicating the validation nodes.
        loss_func (callable): Loss function to compute the validation loss.
        run (sacred.Run, optional): Logging object for tracking metrics.
        epoch (int, optional): The current epoch number for logging.
        device (torch.device, optional): Device to run the evaluation.

    Returns:
        tuple: The validation loss and accuracy.
    """
    model.eval()
    data = data.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss = loss_func(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = accuracy(pred, data.y[mask])

        if run and epoch is not None:
            run.log_scalar('validation.loss', loss, epoch)
            run.log_scalar('evaluation.accuracy', acc, epoch)

    return loss, acc

def val_ensemble(models, data, mask, loss_func, run=None, epoch=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Evaluates an ensemble of models on the validation set.

    Args:
        models (list of torch.nn.Module): List of models to be evaluated.
        data (torch_geometric.data.Data): The graph data.
        mask (Tensor): Boolean mask indicating the validation nodes.
        loss_func (callable): Loss function to compute the validation loss.
        run (sacred.Run, optional): Logging object for tracking metrics.
        epoch (int, optional): The current epoch number for logging.
        device (torch.device, optional): Device to run the evaluation.

    Returns:
        tuple: The validation loss and accuracy.
    """
    data = data.to(device)
    mask = mask.to(device)

    for model in models:
        model.eval()

    with torch.no_grad():
        all_logits = [model(data.x, data.edge_index) for model in models]
        logits = torch.mean(torch.stack(all_logits), dim=0)
        loss = loss_func(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = accuracy(pred, data.y[mask])

        if run and epoch is not None:
            run.log_scalar('validation.loss', loss, epoch)
            run.log_scalar('evaluation.accuracy', acc, epoch)

    return loss, acc

def test_model(model, data, mask, run=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dropout_test=False, ood_mask=None, id_mask=None):
    """
    Evaluates the model on the test set.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data (torch_geometric.data.Data): The graph data.
        mask (Tensor): Boolean mask indicating the test nodes.
        run (sacred.Run, optional): Logging object for tracking metrics.
        device (torch.device, optional): Device to run the evaluation.
        dropout_test (bool, optional): Whether to apply dropout during testing.
        ood_mask (Tensor, optional): Mask for out-of-distribution nodes.
        id_mask (Tensor, optional): Mask for in-distribution nodes.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    model.train() if dropout_test else model.eval()
    data = data.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits[mask].max(1)[1]
        acc = accuracy(pred, data.y[mask])
        results = calculate_metrics(data, logits.cpu(), mask.cpu(), ood_mask=ood_mask, id_mask=id_mask)

        if run:
            run.log_scalar('test.accuracy', acc)

    results["accuracy"] = acc
    return results

def test_ensemble(models, data, mask, run=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dropout_test=False, ood_mask=None, id_mask=None):
    """
    Evaluates an ensemble of models on the test set.

    Args:
        models (list of torch.nn.Module): List of models to be evaluated.
        data (torch_geometric.data.Data): The graph data.
        mask (Tensor): Boolean mask indicating the test nodes.
        run (sacred.Run, optional): Logging object for tracking metrics.
        device (torch.device, optional): Device to run the evaluation.
        dropout_test (bool, optional): Whether to apply dropout during testing.
        ood_mask (Tensor, optional): Mask for out-of-distribution nodes.
        id_mask (Tensor, optional): Mask for in-distribution nodes.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    data = data.to(device)
    mask = mask.to(device)

    for model in models:
        model.train() if dropout_test else model.eval()

    with torch.no_grad():
        all_logits = [model(data.x, data.edge_index) for model in models]
        logits = torch.mean(torch.stack(all_logits), dim=0)
        pred = logits[mask].max(1)[1]
        acc = accuracy(pred, data.y[mask])
        results = calculate_metrics(data, logits.cpu(), mask.cpu(), ood_mask=ood_mask, id_mask=id_mask)

        if run:
            run.log_scalar('test.accuracy', acc)

    results['accuracy'] = acc
    return results

def calculate_metrics(data, logits, mask, uncertainty=None, all_probs=None, ood_mask=None, id_mask=None):
    """
    Calculate evaluation metrics for given logits and masks.

    Args:
        data (torch_geometric.data.Data): The input data.
        logits (Tensor): Predicted logits.
        mask (Tensor): Mask indicating the samples for evaluation.
        uncertainty (Tensor, optional): Uncertainty estimates for the predictions.
        all_probs (Tensor, optional): Predicted probabilities for all classes.
        ood_mask (Tensor, optional): Mask for out-of-distribution nodes.
        id_mask (Tensor, optional): Mask for in-distribution nodes.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    if ood_mask is None or ood_mask.sum() == 0:
        print("No perturbations, OOD mask was None or empty.")
        return {}

    groundtruth = data.y[mask].cpu()
    ece_score = ece(logits[mask].cpu(), groundtruth)
    test_ood_mask = (ood_mask & mask).cpu()
    ood_acc = accuracy(logits[test_ood_mask].max(1)[1].cpu(), data.y[test_ood_mask].cpu())
    test_id_mask = (id_mask & mask).cpu()
    id_acc = accuracy(logits[test_id_mask].max(1)[1].cpu(), data.y[test_id_mask].cpu())
    brier = brier_score(logits[mask].cpu(), groundtruth)
    roc_auc_alea = aleatoric_measure(logits[mask].cpu(), test_ood_mask)
    roc_auc_alea_entropy = aleatoric_measure_entropy(all_probs[:, mask, :].cpu(), test_ood_mask)

    results = {
        "ece": ece_score,
        "id_acc": id_acc,
        "ood_acc": ood_acc,
        "brier_score": brier,
        "roc_auc_alea": roc_auc_alea,
        "roc_auc_alea_entropy": roc_auc_alea_entropy,
    }

    if uncertainty is not None:
        results['roc_auc_epist'] = epistemic_measure(logits[mask].cpu(), test_ood_mask, uncertainty[mask].cpu())
        results['roc_auc_epist_mi'] = epistemic_measure_mutual_information(all_probs[:, mask, :].cpu(), test_ood_mask)

    return results

def val_model_with_anchoring(model, data, mask, loss_func, anchors_set, run=None, epoch=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), class_based_anchor=False):
    """
    Evaluates the model on the validation set using anchoring.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data (torch_geometric.data.Data): The graph data.
        mask (Tensor): Boolean mask indicating the validation nodes.
        loss_func (callable): Loss function to compute the validation loss.
        anchors_set (list): Set of anchors for the evaluation.
        run (sacred.Run, optional): Logging object for tracking metrics.
        epoch (int, optional): The current epoch number for logging.
        device (torch.device, optional): Device to run the evaluation.
        class_based_anchor (bool, optional): Whether to use class-based anchoring.

    Returns:
        tuple: The validation loss and accuracy.
    """
    model.eval()
    data = data.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        all_logits = [
            model(data.x, data.edge_index, data.y, anchors.to(device)) if class_based_anchor else model(data.x, data.edge_index, anchors.to(device))
            for anchors in anchors_set
        ]
        logits = torch.mean(torch.stack(all_logits), dim=0)
        loss = loss_func(logits[mask], data.y[mask]).item()
        acc = accuracy(logits[mask].max(1)[1], data.y[mask])

        if run and epoch is not None:
            run.log_scalar('validation.loss', loss, epoch)
            run.log_scalar('evaluation.accuracy', acc, epoch)

    return loss, acc

def test_model_with_anchoring(model, data, mask, anchors_set, run=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), ood_mask=None, id_mask=None, class_based_anchor=False):
    """
    Evaluates the model on the test set using anchoring.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data (torch_geometric.data.Data): The graph data.
        mask (Tensor): Boolean mask indicating the test nodes.
        anchors_set (list): Set of anchors for the evaluation.
        run (sacred.Run, optional): Logging object for tracking metrics.
        device (torch.device, optional): Device to run the evaluation.
        ood_mask (Tensor, optional): Mask for out-of-distribution nodes.
        id_mask (Tensor, optional): Mask for in-distribution nodes.
        class_based_anchor (bool, optional): Whether to use class-based anchoring.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    model.eval()
    data = data.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        all_logits = [
            model(data.x, data.edge_index, data.y, anchors.to(device)) if class_based_anchor else model(data.x, data.edge_index, anchors.to(device))
            for anchors in anchors_set
        ]
        logits = torch.mean(torch.stack(all_logits), dim=0)
        uncertainty = torch.var(torch.stack([torch.nn.functional.softmax(l, dim=1) for l in all_logits]), dim=0)
        results = calculate_metrics(data, logits.cpu(), mask.cpu(), uncertainty.cpu(), ood_mask=ood_mask, id_mask=id_mask)

        if run:
            run.log_scalar('test.accuracy', results['accuracy'])

    results['logits'] = logits.cpu()
    return results
