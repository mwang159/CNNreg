import torch
import torchsort

def spearmanr(pred, target, indx=None, **kw):
    """Differentiable Spearman correlation using soft ranking."""
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    if len(target.shape) == 1:
        target = target.unsqueeze(0)
    if indx is not None:
        pred = pred[:, indx]
        target = target[:, indx]
    pred_rank = torchsort.soft_rank(pred, **kw)
    target_rank = torchsort.soft_rank(target, **kw)
    pred_rank_center = pred_rank - pred_rank.mean(dim=-1, keepdim=True)
    target_rank_center = target_rank - target_rank.mean(dim=-1, keepdim=True)
    covariance = (pred_rank_center * target_rank_center).sum(dim=-1)
    pred_std_rank = torch.sqrt((pred_rank_center ** 2).sum(dim=-1))
    target_std_rank = torch.sqrt((target_rank_center ** 2).sum(dim=-1))
    return covariance / (pred_std_rank * target_std_rank)

def loss_spearmanr(pred, target, indx=None, **kw):
    """Spearman correlation loss."""
    return torch.mean(1 - spearmanr(pred, target, indx=indx))

def loss_prop(model):
    loss = 0.0
    for param in model.conv1.parameters():
        loss = loss + torch.sum(torch.abs(param[param < 0]))
        #prop_sum = torch.sum(torch.squeeze(param,1),1) ## sum of cell proportion for each sample
        #loss = loss + torch.mean(torch.max(torch.abs(prop_sum-1.0)-0.8, torch.zeros_like(prop_sum)))
    return loss


def loss_ref(model, pHash):
    loss = 0.0
    for param in model.refLayer.parameters():
        loss = torch.sum(torch.abs(param[param < 0]))
        for x in range(pHash["n_celltype"]):
            idx = [y for y in range(x*pHash["n_ref"], (x+1)*pHash["n_ref"])]
            ref_sum = torch.sum(param[idx])
            #loss = loss + torch.abs(ref_sum-1)
            loss = loss +torch.max(torch.abs(ref_sum-1.0), torch.zeros_like(ref_sum))
    return loss



def loss_scale(model):
    loss = 0.0
    #for param in model.celltypeScaleLayer.parameters():
    #    idx_lo = param < 0.2
    #    if torch.sum(idx_lo) > 0:
    #        loss = loss + torch.mean((0.2-param[idx_lo])*(0.2-param[idx_lo]))
    #    idx_hi = param > 5.0
    #    if torch.sum(idx_hi) > 0:
    #        loss = loss + torch.mean((5.0-param[idx_hi])*(5.0-param[idx_hi]))
    #return loss
    for param in model.stretchLayer.parameters():
        #loss = torch.sum(torch.abs(param[param < 0.0]))
        #loss = loss + torch.sum(torch.max(torch.abs(param-1.0)-0.1, torch.zeros_like(param)))
        loss = torch.sum(torch.abs(param[param < 0.5]))
        loss = loss + torch.sum(torch.abs(param[param > 2]))
        #for ii in range(pHash["n_celltype"]):
        #    loss = loss + loss_spearmanr(x[dHash["idx_feature_celltype"][ii]], dHash["ref_CSE"][ii])
    return loss



def loss_stretch(model,x, dHash, pHash):
    loss = 0.0
    for param in model.stretchLayer.parameters():
        #loss = torch.sum(torch.abs(param[param < 0.0]))
        #loss = loss + torch.sum(torch.max(torch.abs(param-1.0)-0.1, torch.zeros_like(param)))
        loss = torch.mean(torch.abs(param[param < 0.5]))
        loss = loss + torch.mean(torch.abs(param[param > 2]))
        #for ii in range(pHash["n_celltype"]):
        #    loss = loss + loss_spearmanr(x[dHash["idx_feature_celltype"][ii]], dHash["ref_CSE"][ii])
    return loss


def loss_epsilon_insensitive(prediction, target, epsilon):
    #return torch.mean(torch.max(torch.abs(prediction-target) - epsilon, torch.zeros_like(prediction)))
    return torch.mean(torch.max(torch.abs(prediction-target) - epsilon*target, torch.zeros_like(prediction)))


def loss_epsilon_insensitive_2(prediction, target, epsilon):
    #return torch.mean(torch.max(torch.abs(prediction-target) - epsilon, torch.zeros_like(prediction)))
    return torch.mean(torch.max(torch.abs(prediction-target) - epsilon, torch.zeros_like(prediction)))


def loss_large_insensitive(prediction, target, threshold):
    #return torch.mean(torch.min(torch.abs(prediction-target) , threshold*torch.ones_like(prediction)))
    return torch.mean(torch.min(torch.abs(prediction-target), threshold*target))



def loss_large_insensitive_2(prediction, target, threshold):
    #return torch.mean(torch.min(torch.abs(prediction-target) , threshold*torch.ones_like(prediction)))
    return torch.mean(torch.min(torch.abs(prediction-target), threshold*torch.ones_like(prediction)))


def loss_small_large_insensitive(prediction, target, epsilon, threshold):
    #return torch.mean(torch.min(torch.abs(prediction-target) , threshold*torch.ones_like(prediction)))
    loss = torch.min( torch.max(torch.abs(prediction-target) - epsilon*target, torch.zeros_like(prediction)), threshold*torch.ones_like(prediction))
    return torch.mean(loss)


