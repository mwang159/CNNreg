import torch
import torch.nn as nn
import numpy as np
import torchsort

class RefCombLayer(nn.Module):
    """Reference combination layer for CNN deconvolution."""
    def __init__(self, pHash, dHash, device=None):
        super().__init__()
        num = 1.0 / pHash["n_ref"]
        w = torch.empty(pHash["n_ref"] * pHash["n_celltype"])
        self.weight = nn.Parameter(torch.nn.init.uniform_(w, a=num-num/5, b=num+num/5).to(device))
        self.w_expand = torch.flatten(self.weight.expand([pHash["n_gene"], len(w)]))

    def forward(self, x):
        return torch.mul(x, self.w_expand)


class SliceSumLayer(nn.Module):
    """Sum slices of the reference layer output."""
    def __init__(self, pHash, dHash, device=None):
        super().__init__()
        self.slice_column = pHash["n_ref"]
        self.slice_row = pHash["n_celltype"] * pHash["n_gene"]

    def forward(self, x):
        x = x.reshape(self.slice_row, self.slice_column)
        return torch.sum(x, dim=1)


class CelltypeScaleLayer(nn.Module):
    """Scale expression for each cell type."""
    def __init__(self, pHash, dHash, device=None):
        super().__init__()
        w = torch.empty(pHash["n_celltype"])
        self.weight = nn.Parameter(torch.nn.init.normal_(w, 1.0, 0.1).to(device))
        self.idx = dHash["idx_feature_celltype"]

    def forward(self, x):
        z = x[self.idx[0]] * self.weight[0]
        for ii in range(1, len(self.idx)):
            y = x[self.idx[ii]] * self.weight[ii]
            z = torch.vstack((z,y))
        return torch.flatten(torch.t(z))


class StretchLayer(nn.Module):
    """Stretch layer to scale features per gene."""
    def __init__(self, pHash, dHash, device=None):
        super().__init__()
        w = torch.empty(pHash["n_gene"])
        self.weight = nn.Parameter(torch.nn.init.normal_(w, 1.0, 0.1).to(device))
        self.w_expand = torch.flatten(torch.t(self.weight.expand((pHash["n_celltype"], -1))))


    def forward(self, x):
        return torch.mul(x, self.w_expand)


class DeconvProp_S1(nn.Module):
    """Stage I CNN model for cell proportion estimation."""
    
    @staticmethod
    def ini_kernel_weight(pHash):
        """Initialize kernel weights for conv1 layer."""
        ## (out_channels, in_channels, kernel_size)
        arr = np.ones(pHash["n_celltype"])/pHash["n_celltype"]
        arr = np.tile(arr, (pHash["n_sample"], 1, 1))
        w = torch.empty((arr.shape[0], arr.shape[1], arr.shape[2]))
        weights = torch.nn.init.normal_(w, 1.0/pHash["n_celltype"], 0.25*1.0/pHash["n_celltype"]).to(pHash["device"])
        return nn.Parameter(weights)
    
    def __init__(self, dHash, pHash):
        super().__init__()
        k = pHash["n_gene"]*pHash["n_celltype"]
        self.refLayer = RefCombLayer(pHash, dHash, device=pHash["device"])
        self.sum = SliceSumLayer(pHash, dHash, device=pHash["device"])
        self.celltypeScaleLayer = CelltypeScaleLayer(pHash, dHash, device=pHash["device"])
        self.stretchLayer = StretchLayer(pHash, dHash, device=pHash["device"])
        self.conv1 = nn.Conv1d(1, pHash["n_sample"], kernel_size=pHash["n_kernel"],
                               stride=pHash["n_kernel"], device=pHash["device"], bias=False)
        self.conv1.weight = self.ini_kernel_weight(pHash)

    def forward(self, x, N_batch, N_feature):
        y = self.refLayer(x)
        y0 = self.sum(y)
        y1 = self.celltypeScaleLayer(y0)
        y2 = self.stretchLayer(y1)
        y = y2.view(N_batch, 1, N_feature)
        y = self.conv1(y).squeeze(0)
        return y, y0, y1, y2
