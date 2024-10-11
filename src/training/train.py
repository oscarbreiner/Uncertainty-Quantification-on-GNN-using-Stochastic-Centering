import torch

def train_model(model, data, mask, optimizer, loss_func, run=None, epoch=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), class_based_anchor=False):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        data (torch_geometric.data.Data): The graph data.
        mask (Tensor): Boolean mask indicating the training nodes.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_func (callable): The loss function.
        run (sacred.Run, optional): Logging object for tracking metrics.
        epoch (int, optional): The current epoch number for logging.
        device (torch.device, optional): Device to run the training.
        class_based_anchor (bool, optional): Whether to use class-based anchoring.

    Returns:
        float: The training loss.
    """
    data = data.to(device)
    mask = mask.to(device)

    model.train()
    optimizer.zero_grad()

    # Forward pass with or without class-based anchoring
    out = model(data.x, data.edge_index, data.y) if class_based_anchor else model(data.x, data.edge_index)

    # Compute loss and perform optimization step
    loss = loss_func(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()

    if run and epoch is not None:
        run.log_scalar('training.loss', loss.item(), epoch)

    return loss.item()

def train_model_with_anchoring(model, data, mask, optimizer, loss_func, anchors, run=None, epoch=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Trains the model for one epoch using anchoring.

    Args:
        model (torch.nn.Module): The model to be trained.
        data (torch_geometric.data.Data): The graph data.
        mask (Tensor): Boolean mask indicating the training nodes.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_func (callable): The loss function.
        anchors (Tensor): Anchors to be applied to the data.
        run (sacred.Run, optional): Logging object for tracking metrics.
        epoch (int, optional): The current epoch number for logging.
        device (torch.device, optional): Device to run the training.

    Returns:
        float: The training loss.
    """
    data = data.to(device)
    anchors = anchors.to(device)
    mask = mask.to(device)

    # Apply anchoring to the input features
    anchored_data = torch.cat([data.x - anchors, anchors], dim=1)

    model.train()
    optimizer.zero_grad()
    out = model(anchored_data, data.edge_index)

    # Compute loss and perform optimization step
    loss = loss_func(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()

    if run and epoch is not None:
        run.log_scalar('training.loss', loss.item(), epoch)

    return loss.item()
