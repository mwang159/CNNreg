import torch
import pandas as pd
import numpy as np
from .data import divide_by_row_sum, reformat_ref
from .layers import DeconvProp_S1
from .losses import (loss_prop, loss_ref, loss_scale, loss_stretch, 
                     loss_epsilon_insensitive, loss_epsilon_insensitive_2,
                     spearmanr)
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef

# https://pytorch.org/docs/0.3.0/optim.html#per-parameter-options
# https://discuss.pytorch.org/t/parameters-with-requires-grad-false-are-updated-during-training/90096/9
# https://www.youtube.com/watch?v=DbeIqrwb_dE

#### initialize weigth with some randomness improves performance
#### use of optim.Adam worsen the performance a lot
#### block = 2 seems perform better than larger block size
#### tuning celltypeScale layer and cell prortion layer separately in S1 performs better than tuning them together. 
#### using StretchLayer_2 in S2 worsen the performance. Should stick to StretchLayer

def trainProp(dHash, pHash):
    """
    Train the Stage I CNN model for cell proportion estimation.
    
    Parameters
    ----------
    dHash : dict
        Data dictionary containing bulk RNA-seq and reference data.
    pHash : dict
        Parameter dictionary with model configuration.
    """
    ## Stage I:  scale expression for each cell type and adjust cell proportion
    
    torch.manual_seed(1334)
    torch.cuda.manual_seed(1334)
    net_Prop_S1 = DeconvProp_S1(dHash, pHash)
    net_Prop_S1.train()
    block = 4
    loss  = nn.L1Loss()      # nn.MSELoss() nn.HuberLoss() 
    target = dHash["bulk"]   # (N_sample, N_feature)
    target_mean  = torch.mean(target, 0) + 0.0001
    target_mean_adj  = 2.0*torch.tanh(target_mean+0.1)
    #condition2   = torch.logical_and(target_mean >  0.01, target_mean <= 0.5)
    #target_mean_adj  = torch.tensor(torch.where(condition1, torch.tensor(0.25).double().to(pHash["device"]), torch.where(condition2, torch.tensor(0.5).double().to(pHash["device"]), torch.tensor(1.0).double().to(pHash["device"]))), dtype=torch.float64, device=pHash["device"])
    LR = 0.02 #max(0.0001, min(0.001, 1-epoch/N)) ## cannot use bigger than 0.05
    N = pHash["max_epoch_cellprop"]
    ll_kernel = []
    loss_model  = []
    for epoch in range(0, N):
        print("epoch = " + str(epoch))
        modd       = epoch % block
        x_predict, x_afterRef, x_afterScale, x_afterStretch  = net_Prop_S1(dHash["CSE_reformat"], 1, pHash["n_celltype"]*pHash["n_gene"])
        new_ref    = x_afterStretch.reshape([pHash["n_gene"], pHash["n_celltype"]])
        gene_var   = torch.var(new_ref, dim=1)
        quantile_marker = torch.quantile(new_ref, 0.5, dim=0)
        ll = []
        for ii in range( pHash["n_celltype"] ):
            mask        = torch.ones(new_ref.shape[1], dtype=torch.bool)
            mask[ii]    = False  # Set the mask to False for the column you want to exclude
            #ll = ll + torch.where((new_ref[:,ii] >= quantile_marker[ii]) & (new_ref[:,ii] > torch.mean(new_ref[:,mask],dim=1)[0]))[0].tolist()
            ll = ll + torch.where((new_ref[:,ii] >= quantile_marker[ii]))[0].tolist()
        indx_marker     = list(set(ll)) 
        gene_cv    = gene_var/target_mean
        th         = torch.quantile(gene_cv, 0.75)
        indx_1     = torch.where(((gene_cv >= 1.0) | (gene_cv >= th)))[0]
        #indx_2     = list(set(indx_marker + torch.where((target_mean >= 0.5))[0].tolist()))
        indx_2     = list(set(indx_marker).intersection(set(torch.where((target_mean >= 0.05))[0].tolist())))
        indx_3     = torch.where((target_mean < 2.0))[0]
        if modd == 1:       # tune reference layer
            train_loss = loss_ref(net_Prop_S1, pHash) + (1-torch.mean(pearson_corrcoef(x_predict[:,indx_3].t(),  target[:,indx_3].t())))
            #train_loss = loss_ref(net_Prop_S1, pHash) + 1-torch.mean(spearmanr(x_predict[:,indx_3], target[:,indx_3]))
            #0.1*(1-torch.mean(pearson_corrcoef(x_predict[:,indx_3].t(), target[:,indx_3].t())))
            train_loss.backward()
            with torch.no_grad():
                net_Prop_S1.refLayer.weight.sub_(net_Prop_S1.refLayer.weight.grad*LR/pHash["n_ref"])

        elif modd == 2:
            train_loss = loss_scale(net_Prop_S1) + 0.1*(1-torch.mean(pearson_corrcoef(x_predict[:,indx_3].t(),  target[:,indx_3].t())))
            train_loss.backward()
            with torch.no_grad():
                net_Prop_S1.celltypeScaleLayer.weight.sub_(net_Prop_S1.celltypeScaleLayer.weight.grad*LR)

        elif modd == 3:
            train_loss = loss_stretch(net_Prop_S1, x_afterStretch, dHash, pHash) + 0.1*(1-torch.mean(pearson_corrcoef(x_predict[:,indx_3].t(),  target[:,indx_3].t())))
            train_loss.backward()
            with torch.no_grad():
                net_Prop_S1.stretchLayer.weight.sub_(net_Prop_S1.stretchLayer.weight.grad*LR*pHash["n_gene"]/100)

        else:      
            rho_1      = spearmanr(x_predict[:,indx_1].t(), target[:,indx_1].t())
            rho_2      = spearmanr(x_predict[:,indx_2], target[:,indx_2])
            train_loss = loss_prop(net_Prop_S1) + \
              loss_epsilon_insensitive(x_predict[:,indx_3]/target_mean_adj[indx_3], target[:,indx_3]/target_mean_adj[indx_3], 0.05) + \
              0.1*loss_epsilon_insensitive_2(rho_1, torch.ones_like(rho_1), 0.05) + \
              0.1*loss_epsilon_insensitive_2(rho_2, torch.ones_like(rho_2), 0.05)
            train_loss.backward()
            with torch.no_grad():
                net_Prop_S1.conv1.weight.sub_(net_Prop_S1.conv1.weight.grad*LR)
                ll_kernel.append(net_Prop_S1.conv1.weight.tolist())
                
        with torch.no_grad():
            net_Prop_S1.refLayer.weight.grad.zero_()
            net_Prop_S1.celltypeScaleLayer.weight.grad.zero_()
            net_Prop_S1.stretchLayer.weight.grad.zero_()
            net_Prop_S1.conv1.weight.grad.zero_()
        ## output estimation
        if epoch % 1000 == 0:
            print("celltypeScaleLayer weight")
            print(net_Prop_S1.celltypeScaleLayer.weight.tolist())
            cellprop = np.squeeze(np.array(net_Prop_S1.conv1.weight.tolist()), 1)
            cellprop = divide_by_row_sum(cellprop)
            df   = pd.concat([pd.DataFrame(dHash["sample"]), pd.DataFrame(cellprop)], axis=1)
            x    = ["Sample"]
            x.extend(dHash["celltype"])
            df.columns = x
            df.to_csv(pHash["data_out_dir"]+ "/" + "Prop_predicted_" + pHash["prefix"] + "_epoch_" + str(epoch) + ".csv", index=False)


    for name, param in net_Prop_S1.named_parameters():
        print(name, param)

    ### output estimated proportion from Stage I
    cellprop = np.squeeze(np.array(net_Prop_S1.conv1.weight.tolist()), 1)
    df   = pd.concat([pd.DataFrame(dHash["sample"]), pd.DataFrame(cellprop)], axis=1)
    x    = ["Sample"]
    x.extend(dHash["celltype"])
    df.columns = x
    df.to_csv(pHash["data_out_dir"]+ "/" + "Prop_predicted_" + pHash["prefix"] + ".csv", index=False)
    # save final model
    torch.save(net_Prop_S1.state_dict(), pHash["model_file"])


####################################################################################################################################################################