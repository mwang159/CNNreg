"""
cnn_deconv: A CNN-based regression package for cell type deconvolution of bulk RNA-seq using scRNA-seq reference.
"""
from .data import data_CSE, flatten_list, divide_by_row_sum, reformat_ref
from .layers import RefCombLayer, SliceSumLayer, CelltypeScaleLayer, StretchLayer, DeconvProp_S1
from .losses import (loss_spearmanr, loss_prop, loss_ref, loss_scale, 
                     loss_stretch, loss_epsilon_insensitive)
from .train import trainProp
