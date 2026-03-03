"""
cnn_deconv: A CNN-based regression package for cell type deconvolution of bulk RNA-seq using scRNA-seq reference.
"""
from .data import data_CSE, divide_by_row_sum, reformat_ref
from .layers import RefCombLayer, SliceSumLayer, CelltypeScaleLayer, DeconvProp
from .losses import loss_prop, loss_ref, loss_scale, loss_epsilon_insensitive, loss_epsilon_insensitive_2
from .train import trainProp, get_indx, check_stop, save_estimation
