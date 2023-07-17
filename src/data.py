import os

import pandas as pd
import numpy as np

import torch

from torch.utils.data import Dataset


class NpyDataset(Dataset):
    """Npy files"""

    def __init__(
        self,
        root_dir: str,
        metadata_file: str,
        subset: str,
        subset_col: str = "subset",
        subset_mode: str = "default",
        transform=None,
        original_only: bool = False,
    ):
        """
        Arguments:

        subset_mode: str
            Which subset to choose. If == 'default' returns those rows for
            which the value in column `subset_col` is equal to `subset`,
            else return the complement set. This is done to retrieve the
            training set for a k-fold split (it's easier to just point out
            which data to leave aside).

        original_only: bool
            Whether to return only the original version of the songs
            (no augmentations). Used for computing metrics

        """
        self.metadata = pd.read_csv(metadata_file)

        # Keep only rows for subset
        if subset_mode == "default":
            self.metadata = self.metadata[self.metadata[subset_col] == subset]
        elif subset_mode == "kfold-train":
            self.metadata = self.metadata[self.metadata[subset_col] != subset]
        else:
            raise ValueError(
                "Parameter `subset_mode` can be either 'default' or 'kfold-train'"
            )

        if original_only:
            self.metadata = self.metadata[
                self.metadata["path"].map(lambda x: "orig" in x)
            ]

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
        sample = np.load(os.path.join(self.root_dir, row["path"]))
        genre = row["genre"]

        label = self.cat_to_int[genre]

        if self.transform:
            sample = self.transform(sample[:, :])

        return sample, label
