import argparse
import torch
import pandas as pd
import numpy as np
from .train import trainProp
from .data import data_CSE, reformat_ref, divide_by_row_sum

def main():
    parser = argparse.ArgumentParser(
        description="CNN-based bulk RNA-seq deconvolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-bulk", dest="bulk", type=str, required=False)
    parser.add_argument("-ref", dest="reference", type=str, required=False)
    parser.add_argument("-pre", dest="prefix", type=str, default="Project")
    parser.add_argument("-o", dest="outDIR", type=str, required=True)
    parser.add_argument("-C", dest="kernelSize", type=int, default=10)
    parser.add_argument("-EP", dest="maxEpochCellProp", type=int, default=1000)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paraHash = {
        "bulk": args.bulk,
        "reference": args.reference,
        "data_out_dir": args.outDIR,
        "max_epoch_cellprop": args.maxEpochCellProp,
        "prefix": args.prefix,
        "device": device,
        "n_kernel": args.kernelSize
    }

    df_bulk = pd.read_csv(paraHash["bulk"])
    dataHash = {
        "bulk": torch.tensor(df_bulk.iloc[:, 1:].values.transpose(), dtype=torch.float32).to(device),
        "sample": df_bulk.columns.values[range(1, df_bulk.shape[1])],
        "celltype": [f"celltype_{i+1}" for i in range(paraHash["n_kernel"])],
        "CSE": data_CSE(paraHash["reference"], device=device)
    }
    paraHash["n_gene"] = df_bulk.shape[0]
    paraHash["n_sample"] = df_bulk.shape[1] - 1
    paraHash["n_celltype"] = paraHash["n_kernel"]
    paraHash["n_ref"] = dataHash["CSE"].expr_cse.shape[0]

    # Initialize index arrays for celltype and reference features
    k = paraHash["n_gene"] * paraHash["n_celltype"]
    dataHash["idx_feature_celltype"] = []
    for x in range(paraHash["n_celltype"]):
        dataHash["idx_feature_celltype"].append([int(y + x) for y in range(0, k, paraHash["n_celltype"])])

    dataHash["idx_feature_ref"] = []
    for x in range(paraHash["n_ref"]):
        dataHash["idx_feature_ref"].append([y for y in range(x * k, (x + 1) * k)])

    dataHash["CSE_reformat"] = reformat_ref(dataHash["CSE"].expr_cse)
    df = pd.DataFrame(np.array(dataHash["CSE_reformat"].cpu()))
    trainProp(dataHash, paraHash)

if __name__ == "__main__":
    main()
