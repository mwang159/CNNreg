import torch


def loss_prop(model):
    """Penalize negative cell proportions."""
    loss = 0.0
    for param in model.conv1.parameters():
        loss = loss + torch.sum(torch.abs(param[param < 0]))
    return loss


def loss_ref(model, pHash):
    """Ensure reference weights sum to ~1 per cell type."""
    loss = 0.0
    for param in model.refLayer.parameters():
        loss = torch.sum(torch.abs(param[param < 0]))
        for x in range(pHash["n_celltype"]):
            idx = [y for y in range(x*pHash["n_ref"], (x+1)*pHash["n_ref"])]
            ref_sum = torch.sum(param[idx])
            loss = loss +torch.max(torch.abs(ref_sum-1.0), torch.zeros_like(ref_sum))
    return loss


def loss_scale(model):
    """Constrain cell type scaling factors to reasonable range [0.25, 4]."""
    loss = 0.0
    for param in model.celltypeScaleLayer.parameters():
        loss = torch.sum(torch.abs(param[param < 0.25]))
        loss = loss + torch.sum(torch.abs(param[param > 4]))
    return loss


def loss_epsilon_insensitive(prediction, target, epsilon):
    """Custom robust loss, less sensitive to small errors (relative epsilon)."""
    return torch.mean(torch.max(torch.abs(prediction-target) - epsilon*target, torch.zeros_like(prediction)))


def loss_epsilon_insensitive_2(prediction, target, epsilon):
    """Custom robust loss, less sensitive to small errors (absolute epsilon)."""
    return torch.mean(torch.max(torch.abs(prediction-target) - epsilon, torch.zeros_like(prediction)))


