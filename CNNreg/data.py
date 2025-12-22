import torch
import pandas as pd
import numpy as np

def flatten_list(nested_list):
    """Flatten a nested list recursively."""
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def divide_by_row_sum(array):
    """Normalize rows to sum to 1."""
    row_sums = np.sum(array, axis=1, keepdims=True)
    return array / row_sums

def reformat_ref(CSE_data):
    """Flatten the reference CSE tensor."""
    return torch.flatten(torch.t(CSE_data), start_dim=0)

class data_CSE(torch.utils.data.Dataset):
    """Dataset wrapper for Cell Type Specific Expression (CSE)."""
    def __init__(self, f_CSE, device):
        super().__init__()
        df_cse = pd.read_csv(f_CSE)
        sample = df_cse.iloc[:,0]
        expr_cse = df_cse.iloc[:,1:]
        self.expr_cse = torch.tensor(expr_cse.values, dtype=torch.float32).to(device)
        self.sample = sample

    def __len__(self):
        return self.expr_cse.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {'expr_cse': self.expr_cse[idx, :], 'sample': self.sample[idx]}

    def update_data(self, new_CSE):
        self.expr_cse = new_CSE
