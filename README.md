# CNNreg: CNN-based Cell Type Deconvolution

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning approach for cell type deconvolution of bulk RNA-seq data using single-cell RNA-seq reference data. CNNreg employs a custom CNN-based regression model to estimate cell type proportions in complex tissue samples.

We recommend using [CNNregR](https://github.com/mwang159/CNNregR) package in R for easier data handling and preprocessing. Alternatively, users can also preprocess the data by yourself and use the Python API or CLI provided in this package.

## Installation

### Option 1: Using pip (Recommended)

CNNreg requires PyTorch. Install PyTorch first according to your system configuration:

```bash
# For CUDA 12.6 (check your CUDA version: nvidia-smi)
pip3 install torch --index-url https://download.pytorch.org/whl/cu126

# For CUDA 11.8
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install CNNreg:

```bash
pip install CNNreg
```

### Option 2: From Source (Development)

```bash
git clone https://github.com/mwang159/CNNreg.git
cd CNNreg
pip install -e .
```

### Option 3: Using Conda Environment

```bash
# Create environment
conda create -n cnnreg_env python=3.10
conda activate cnnreg_env

# Install PyTorch (choose appropriate CUDA version)
pip3 install torch --index-url https://download.pytorch.org/whl/cu126

# Install CNNreg
pip install CNNreg
```

### Verify Installation

```bash
# Check CLI command
cnnreg --help

# Test in Python
python -c "import CNNreg; print('CNNreg installed successfully!')"
```

## Quick Start

### Command Line Interface

```bash
cnnreg \
    -bulk data/bulk.csv \
    -ref data/sc_ref.csv \
    -o output/ \
    -C 7 \
    -EP 50000 \
    -pre GBM_analysis
```

**Parameters:**
- `-bulk`: Path to bulk RNA-seq CSV file
- `-ref`: Path to reference cell type specific expression (CSE) CSV
- `-o`: Output directory
- `-C`: Number of cell types (kernel size)
- `-EP`: Maximum training epochs
- `-pre`: Output file prefix

**Note**: Training automatically generates cell proportion predictions for the input bulk samples. Predictions are saved at checkpoints (every 10,000 epochs) and at completion (or early stop). Both raw and row-sum-normalized proportions are saved.

### Python API

```python
import torch
import pandas as pd
from CNNreg.data import data_CSE, reformat_ref
from CNNreg.train import trainProp
from CNNreg.layers import DeconvProp

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df_bulk = pd.read_csv("bulk.csv")
bulk_data = torch.tensor(
    df_bulk.iloc[:, 1:].values.transpose(), 
    dtype=torch.float32
).to(device)

# Configure parameters
pHash = {
    "bulk": "bulk.csv",
    "reference": "sc_ref.csv",
    "data_out_dir": "output/",
    "max_epoch_cellprop": 50000,
    "prefix": "GBM",
    "device": device,
    "n_kernel": 7,
    "n_gene": df_bulk.shape[0],
    "n_sample": df_bulk.shape[1] - 1,
    "n_celltype": 7
}

# Initialize data dictionary
dHash = {
    "bulk": bulk_data,
    "sample": df_bulk.columns.values[1:],
    "celltype": ["AClike", "MESlike", "NPClike", "OPClike", "OL", "Myeloid", "T"],
    "CSE": data_CSE(pHash["reference"], device=device)
}

# Add required indices
k = pHash["n_gene"] * pHash["n_celltype"]
dHash["idx_feature_celltype"] = [
    [int(y+x) for y in range(0, k, pHash["n_celltype"])]
    for x in range(pHash["n_celltype"])
]
dHash["CSE_reformat"] = reformat_ref(dHash["CSE"].expr_cse)

# Train model
trainProp(dHash, pHash)
```

## Input Data Format

### Bulk RNA-seq Data

Rows are samples, columns are genes:

```csv
Sample,Gene1,Gene2,Gene3,...
Sample1,0.5,1.2,0.8,...
Sample2,1.1,0.9,1.5,...
Sample3,0.7,1.3,0.6,...
```

### Reference scRNA-seq Data

Cell Type Specific Expression profiles from scRNA-seq:

```csv
CellType,Gene1,Gene2,Gene3,...
AClike_ref1,0.3,0.8,0.5,...
AClike_ref2,0.4,0.7,0.6,...
MESlike_ref1,0.9,0.2,0.4,...
```

## Output Files

- `Estimation_{prefix}_epoch_{N}.csv`: Raw (unnormalized) cell proportion weights at checkpoint epochs (every 10,000) and at early stop / completion
- `Estimation_{prefix}_epoch_{N}_normalized.csv`: Row-sum-normalized cell proportions (sum to 1 per sample) at the same checkpoints

**Output format:**

```csv
Sample,celltype_1,celltype_2,celltype_3,...
Sample1,0.23,0.15,0.31,...
Sample2,0.19,0.28,0.22,...
```

## Model Architecture

CNNreg uses a custom 4-layer CNN pipeline specifically designed for biological deconvolution:

1. **RefCombLayer**: Combines multiple reference samples per cell type via learned weighted combination
2. **SliceSumLayer**: Aggregates weighted reference combinations into a single profile per cell type
3. **CelltypeScaleLayer**: Scales the expression profile of each cell type independently (constrained to [0.25, 4])
4. **Conv1D Layer**: Estimates cell proportions via 1D convolution (kernel size = number of cell types)

### Training Strategy

Training uses a **3-phase manual SGD cycle**:
- **Phase 0** (`epoch % 3 == 0`): Tune `conv1` kernel weights (cell proportions) using Pearson correlation + expression fit loss
- **Phase 1** (`epoch % 3 == 1`, only for `epoch < 30,000`): Tune `refLayer` weights (reference combination)
- **Phase 2** (`epoch % 3 == 2`): Tune `celltypeScaleLayer` weights (per-cell-type scaling)

**Early stopping** activates after epoch 20,000: training halts when correlation loss begins increasing and the model has already achieved Pearson r > 0.75 on high-variability genes.

## Citation

If you use CNNreg in your research, please cite:

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- Python Package: [CNNreg on PyPI](https://pypi.org/project/CNNreg/)
- CNNreg: [CNNreg on GitHub](https://github.com/mwang159/CNNreg)
- CNNregR: [CNNregR on GitHub](https://github.com/mwang159/CNNregR)
- Issues: [Report bugs](https://github.com/mwang159/CNNreg/issues)

## Contact

- **Email**: wang.xue@mayo.edu, liu.yuanhang@mayo.edu