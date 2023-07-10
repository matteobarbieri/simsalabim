import os

import pandas as pd
import numpy as np

import torch

from torch.utils.data import Dataset

import librosa


class WavDataset(Dataset):
    """
    Wav files
    Returns mel spectrogram computed on the flight
    """

    def __init__(
        self,
        root_dir,
        metadata_file,
        subset,
        subset_col="subset",
        n_fft=2048,
        n_mels=256,
        hop_length=512,
        audio_transform=None,
        transform=None,
    ):
        """
        Arguments:

        """

        self.metadata = pd.read_csv(metadata_file)

        # Keep only rows for subset
        self.metadata = self.metadata[self.metadata[subset_col] == subset]

        self.subset = subset

        self.root_dir = root_dir

        self.audio_transform = audio_transform
        self.transform = transform

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length

        categories = self.metadata.genre.unique()
        self.cat_to_int = {c: i for i, c in enumerate(sorted(categories))}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        # sample = np.load(os.path.join(self.root_dir, row["path"]))
        sample, sr = librosa.load(os.path.join(self.root_dir, row["path"]))
        genre = row["genre"]

        label = self.cat_to_int[genre]

        # Audio-specific augmentations
        if self.audio_transform is not None:
            sample = self.audio_transform(sample)

        # Create mel spectrogram on the flight
        S = librosa.feature.melspectrogram(
            y=sample,
            sr=sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
        )

        S_db_mel = librosa.amplitude_to_db(S, ref=np.max)

        if self.transform:
            X = self.transform(S_db_mel)

        return X, label
