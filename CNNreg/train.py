import torch
import pandas as pd
import numpy as np
from .data import divide_by_row_sum, reformat_ref
from .layers import DeconvProp
from .losses import (loss_prop, loss_ref, loss_scale,
                     loss_epsilon_insensitive, loss_epsilon_insensitive_2)
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef


def get_indx(x, ref, target, pHash, dHash):
    """Select marker/variable gene indices for training loss computation.

    Parameters
    ----------
    x : Tensor
        Predicted expression (n_sample, n_gene).
    ref : Tensor
        Reshaped reference (n_gene, n_celltype).
    target : Tensor
        Bulk expression target (n_sample, n_gene).
    pHash : dict
        Parameter dictionary.
    dHash : dict
        Data dictionary (must contain target_mean, target_mean_adj,
        target_cv, th_target_cv, target_mean_hi).

    Returns
    -------
    indx_1 : Tensor
        High-variability gene indices.
    indx_2 : list
        Marker gene indices intersected with expressed genes.
    """
    diff       = torch.mean(x / dHash["target_mean_adj"] - target / dHash["target_mean_adj"], dim=0)
    diff_th    = torch.quantile(diff, 0.9)
    gene_var   = torch.var(ref, dim=1)
    quantile_marker = torch.quantile(ref, 0.5, dim=0)
    ll = []
    for ii in range(pHash["n_celltype"]):
        ll = ll + torch.where((ref[:, ii] >= quantile_marker[ii]))[0].tolist()
    indx_marker = list(set(ll))
    gene_cv    = gene_var / dHash["target_mean"]
    th         = torch.quantile(gene_cv, 0.75)
    indx_1     = torch.where((gene_cv >= th) | (dHash["target_cv"] >= dHash["th_target_cv"]) | (diff >= diff_th))[0]
    indx_2     = list(set(indx_marker).intersection(set(torch.where((dHash["target_mean"] >= dHash["target_mean_hi"]))[0].tolist())))
    return indx_1, indx_2


def check_stop(net_Prop, epoch, ct_stop, dict_loss, pHash, dHash):
    """Check early stopping condition based on loss trends.

    Parameters
    ----------
    net_Prop : DeconvProp
        The model.
    epoch : int
        Current epoch.
    ct_stop : int
        Counter of consecutive increases.
    dict_loss : dict
        Dictionary tracking loss_1, loss_2, loss_3 history.
    pHash : dict
        Parameter dictionary.
    dHash : dict
        Data dictionary.

    Returns
    -------
    ct_stop : int
        Updated counter.
    indx_stop : int
        1 if training should stop, 0 otherwise.
    indx_min : int
        Index of the best loss (0 for loss_2, 1 for loss_3).
    """
    indx_min = np.argmin([dict_loss["loss_2"][-1], dict_loss["loss_3"][-1]])
    key = ["loss_2", "loss_3"][indx_min]
    indx_stop = 0
    if dict_loss[key][-1] > dict_loss[key][-2] + 0.0001:
        ct_stop = ct_stop + 1
    if ct_stop >= 1 and dict_loss["loss_2"][-1] < 0.20:
        indx_stop = 1
    return ct_stop, indx_stop, indx_min


def save_estimation(net_Prop, pHash, dHash):
    """Save raw and normalized cell proportion estimates to CSV.

    Parameters
    ----------
    net_Prop : DeconvProp
        The trained model.
    pHash : dict
        Parameter dictionary (needs data_out_dir, prefix).
    dHash : dict
        Data dictionary (needs sample, celltype, epoch_end).
    """
    cellprop = np.squeeze(np.array(net_Prop.conv1.weight.tolist()), 1)
    df   = pd.concat([pd.DataFrame(dHash["sample"]), pd.DataFrame(cellprop)], axis=1)
    x    = ["Sample"]
    x.extend(dHash["celltype"])
    df.columns = x
    df.to_csv(pHash["data_out_dir"] + "/" + "Estimation_" + pHash["prefix"] + "_epoch_" + str(dHash["epoch_end"]) + ".csv", index=False)
    ### output normalized proportion
    cellprop = divide_by_row_sum(cellprop)
    df   = pd.concat([pd.DataFrame(dHash["sample"]), pd.DataFrame(cellprop)], axis=1)
    x    = ["Sample"]
    x.extend(dHash["celltype"])
    df.columns = x
    df.to_csv(pHash["data_out_dir"] + "/" + "Estimation_" + pHash["prefix"] + "_epoch_" + str(dHash["epoch_end"]) + "_normalized.csv", index=False)


def trainProp(dHash, pHash):
    """
    Train the CNN model for cell proportion estimation.

    Uses a 3-phase training cycle (block=3):
      Phase 0: Tune CNN kernel weights (cell proportions)
      Phase 1: Tune reference combination layer (only for epoch < 30000)
      Phase 2: Tune cell type scaling layer

    Includes early stopping after epoch 20000 based on correlation loss trends.

    Parameters
    ----------
    dHash : dict
        Data dictionary containing bulk RNA-seq and reference data.
    pHash : dict
        Parameter dictionary with model configuration.
    """
    torch.manual_seed(1334)
    torch.cuda.manual_seed(1334)
    net_Prop = DeconvProp(dHash, pHash)
    net_Prop.train()
    block = 3
    target           = dHash["bulk"]   # (N_sample, N_feature)

    dHash["target_mean"]      = torch.mean(target, dim=0) + 0.0001
    dHash["target_mean_hi"]   = torch.quantile(dHash["target_mean"], 0.25).item()
    dHash["target_mean_adj"]  = 2 * torch.tanh(dHash["target_mean"] + 0.10)
    dHash["target_cv"]        = torch.var(target, dim=0) / dHash["target_mean"]
    dHash["th_target_cv"]     = torch.quantile(dHash["target_cv"], 0.75)

    dict_loss        = {"loss_1": [], "loss_2": [], "loss_3": []}
    ct_stop          = 0
    early_stop       = 0

    LR               = 0.02
    N = pHash["max_epoch_cellprop"]
    ll_kernel = []

    for epoch in range(0, N + 1):
        print("epoch = " + str(epoch))
        modd          = epoch % block
        x_predict, x_afterRef, x_afterScale = net_Prop(dHash["CSE_reformat"], 1, pHash["n_celltype"] * pHash["n_gene"])
        new_ref       = x_afterScale.reshape([pHash["n_gene"], pHash["n_celltype"]])
        indx_1, indx_2 = get_indx(x_predict, new_ref, target, pHash, dHash)

        if modd == 1 and epoch < 30000:       # tune reference layer
            r          = pearson_corrcoef(x_predict.t(), target.t())
            train_loss = loss_ref(net_Prop, pHash) + 0.1 * loss_epsilon_insensitive_2(r, torch.ones_like(r), 0.05)
            train_loss.backward()
            with torch.no_grad():
                net_Prop.refLayer.weight.sub_(net_Prop.refLayer.weight.grad * LR / 5)
        elif modd == 2:     # tune cell type layer
            r          = pearson_corrcoef(x_predict.t(), target.t())
            train_loss = loss_scale(net_Prop) + 0.1 * loss_epsilon_insensitive_2(r, torch.ones_like(r), 0.05)
            train_loss.backward()
            with torch.no_grad():
                net_Prop.celltypeScaleLayer.weight.sub_(net_Prop.celltypeScaleLayer.weight.grad * LR)
        else:               # tune CNN kernel weight
            r_1      = pearson_corrcoef(x_predict[:, indx_1], target[:, indx_1])
            r_2      = pearson_corrcoef(x_predict[:, indx_2].t(), target[:, indx_2].t())
            loss_1     = loss_epsilon_insensitive(x_predict / dHash["target_mean_adj"], target / dHash["target_mean_adj"], 0.05)
            loss_2     = loss_epsilon_insensitive_2(r_1, torch.ones_like(r_1), 0.05)
            loss_3     = loss_epsilon_insensitive_2(r_2, torch.ones_like(r_2), 0.05)
            train_loss = loss_prop(net_Prop) + loss_1 + 0.1 * loss_2 + 0.1 * loss_3
            train_loss.backward()
            with torch.no_grad():
                net_Prop.conv1.weight.sub_(net_Prop.conv1.weight.grad * LR)
                ll_kernel.append(net_Prop.conv1.weight.tolist())

            if epoch >= 20000 and early_stop == 0:   ## check early stop
                dict_loss["loss_1"].append(loss_1.item())
                dict_loss["loss_2"].append(loss_2.item())
                dict_loss["loss_3"].append(loss_3.item())
                x_predict, x_afterRef, x_afterScale = net_Prop(dHash["CSE_reformat"], 1, pHash["n_celltype"] * pHash["n_gene"])
                r_1      = pearson_corrcoef(x_predict[:, indx_1], target[:, indx_1])
                r_2      = pearson_corrcoef(x_predict[:, indx_2].t(), target[:, indx_2].t())
                loss_1  = loss_epsilon_insensitive(x_predict / dHash["target_mean_adj"], target / dHash["target_mean_adj"], 0.05)
                loss_2  = loss_epsilon_insensitive_2(r_1, torch.ones_like(r_1), 0.05)
                loss_3  = loss_epsilon_insensitive_2(r_2, torch.ones_like(r_2), 0.05)
                dict_loss["loss_1"].append(loss_1.item())
                dict_loss["loss_2"].append(loss_2.item())
                dict_loss["loss_3"].append(loss_3.item())
                ct_stop, early_stop, indx_min = check_stop(net_Prop, epoch, ct_stop, dict_loss, pHash, dHash)
                if early_stop == 1:
                    dHash["epoch_end"] = epoch
                    save_estimation(net_Prop, pHash, dHash)

        with torch.no_grad():
            net_Prop.refLayer.weight.grad.zero_()
            net_Prop.celltypeScaleLayer.weight.grad.zero_()
            net_Prop.conv1.weight.grad.zero_()

        if epoch % 10000 == 0:
            dHash["epoch_end"] = epoch
            save_estimation(net_Prop, pHash, dHash)


    if early_stop == 0:
        dHash["epoch_end"] = pHash["max_epoch_cellprop"]
        save_estimation(net_Prop, pHash, dHash)

    for name, param in net_Prop.named_parameters():
        print(name, param)


####################################################################################################################################################################