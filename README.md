# CNNreg: CNN-based Cell Type Deconvolution

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning approach for cell type deconvolution of bulk RNA-seq data using single-cell RNA-seq reference data. CNNreg employs a custom CNN-based regression model to estimate cell type proportions in complex tissue samples.

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
cnnreg -M train \
    -bulk data/bulk.csv \
    -ref data/sc_ref.csv \
    -o output/ \
    -C 7 \
    -EP 50000 \
    -pre GBM_analysis
```

**Parameters:**
- `-M`: Mode - currently only `train` is implemented (evaluate/predict/explain coming in future versions)
- `-bulk`: Path to bulk RNA-seq CSV file
- `-ref`: Path to reference cell type specific expression (CSE) CSV
- `-o`: Output directory
- `-C`: Number of cell types (kernel size)
- `-EP`: Maximum training epochs
- `-pre`: Output file prefix

**Note**: Training mode automatically generates cell proportion predictions for the input bulk samples. Predictions are saved at checkpoints (every 1000 epochs) and at completion.

### Python API

```python
import torch
import pandas as pd
from CNNreg.data import data_CSE, reformat_ref
from CNNreg.train import trainProp
from CNNreg.layers import DeconvProp_S1

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
    "model_file": "output/model.pt",
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

- `Prop_predicted_{prefix}_epoch_{N}.csv`: Cell proportions at checkpoint epochs (every 1000)
- `Prop_predicted_{prefix}.csv`: Final estimated cell proportions
- `cellprop_model.pt`: Trained PyTorch model (state dict)

**Output format:**

```csv
Sample,celltype_1,celltype_2,celltype_3,...
Sample1,0.23,0.15,0.31,...
Sample2,0.19,0.28,0.22,...
```

## Model Architecture

CNNreg uses a custom 5-layer CNN pipeline specifically designed for biological deconvolution:

1. **RefCombLayer**: Combines multiple reference samples per cell type
2. **SliceSumLayer**: Aggregates reference combinations
3. **CelltypeScaleLayer**: Scales expression for each cell type independently
4. **StretchLayer**: Applies gene-specific scaling factors
5. **Conv1D Layer**: Estimates cell proportions via 1D convolution

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