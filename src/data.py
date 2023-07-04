import os

import pandas as pd
import numpy as np

import torch

from torch.utils.data import Dataset

class NpyDataset(Dataset):
    """Npy files"""

    def __init__(self, root_dir, metadata_file, subset, subset_col='subset', transform=None):
        """
        Arguments:
            
        """
        self.metadata = pd.read_csv(metadata_file)
        
        # Keep only rows for subset
        self.metadata = self.metadata[self.metadata[subset_col] == subset]
        
        self.subset = subset
        
        
        self.root_dir = root_dir
        self.transform = transform
        
        categories = self.metadata.genre.unique()
        self.cat_to_int = {c: i for i, c in enumerate(sorted(categories))}


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
            
        row = self.metadata.iloc[idx]
        sample = np.load(os.path.join(self.root_dir, row['path']))
        genre = row['genre']
        
        label = self.cat_to_int[genre]
        
        if self.transform:
            sample = self.transform(sample[:, :])

        return sample, label