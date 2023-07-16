__all__ = [
    "time_stretch_random",
    "shift_pitch_random",
    "reverb_random",
    "get_fixed_window",
]

import random
from typing import List

import numpy as np

import pedalboard

import librosa

reverb = pedalboard.Reverb()

# Parameters for mel spectrograms generation
hop_length = 512
n_fft = 2048

from librosa.effects import time_stretch


def time_stretch_random(y: np.ndarray, sr: int = -1, intensity: float = 1.0):
    # param sr is kept so that signature is the same for all functions
    # Stretch tie between -20% and +25%
    rate = 1 + intensity * random.uniform(-0.2, 0.25)

    # rate = 1.25 # for debugging purposes, to compute max stretch

    y_stretch = time_stretch(y, rate=rate)

    return y_stretch


from librosa.effects import pitch_shift


def shift_pitch_random(y: np.ndarray, sr: int, intensity: float = 1.0):
    # Shift between three semitones up or down
    #
    n_steps = max(0.5, intensity) * random.uniform(-1.5, 1.5)

    y_pitch = pitch_shift(y, sr=sr, n_steps=n_steps)

    return y_pitch


def reverb_random(y: np.ndarray, sr: int, intensity: float = 1.0):
    """
    Add some amount of reverb to the clip
    """

    # Reverb-specific parameters
    reverb.room_size = intensity * random.uniform(0, 0.6)
    reverb.wet_level = intensity * random.uniform(0.33, 0.5)

    y_reverb = reverb(y, sample_rate=sr)

    return y_reverb


def noise_random(y: np.ndarray, sr: int = -1, intensity: float = 1.0):
    """
    Add random  white noise to the audio sample
    """
    noise_factor = random.uniform(0, 0.01)
    # noise_factor = 0.01
    y_noise = y + np.random.randn(len(y)) * noise_factor
    return y_noise


def get_fixed_window(
    S_db_mel: np.ndarray, width: int = 1024, starting_point: int = None
):
    # Create a window exactly 1024 wide
    # This is needed because time stretching might make the audio shorter
    """
    starting_point: index at which to start the slice, included for reproducibility (otherwise randomly determined)
    """

    if starting_point is not None:
        start_j = starting_point
    else:
        max_starting_point = S_db_mel.shape[1] - width
        start_j = random.randint(0, max_starting_point)

    return S_db_mel[:, start_j : start_j + width]


def get_n_variations(
    audio_file: str, n: int, n_mels: int = 256, return_original: bool = True
) -> List[np.ndarray]:
    """
    Returns n augmented versions of the same audio clip

    return_original: Whether to also return the mel spectrogram of the original clip

    Return list of MEL spectrograms
    """
    # TODO make sure seed is set for reproducibility

    augmentations = []

    y, sr = librosa.load(audio_file)

    if return_original:
        # First save the spectrogram of the original version of the file
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length
        )

        S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
        S_db_mel = get_fixed_window(S_db_mel, starting_point=0)

        augmentations.append(S_db_mel)

    # (for my own) sanity check
    assert S_db_mel.shape[1] == 1024
    assert S_db_mel.shape[0] == n_mels

    # Now Create variations by adding pitch_shift and other things

    for i in range(1, n + 1):
        y_aug = (
            y  # This is just so that I can move stuff up and down without going crazy
        )

        # Control amount of augmentation to prevent distorting the sample too much
        intensities = np.random.uniform(size=(3,))
        intensities /= intensities.sum()

        y_aug = shift_pitch_random(y_aug, sr, intensity=intensities[0])
        y_aug = time_stretch_random(y_aug, sr, intensity=intensities[1])
        y_aug = reverb_random(y_aug, sr, intensity=intensities[2])
        y_aug = noise_random(y_aug, sr)

        # y_aug = shift_pitch_random(y_aug, sr)
        # y_aug = time_stretch_random(y_aug)

        S = librosa.feature.melspectrogram(
            y=y_aug, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length
        )

        S_db_mel = librosa.amplitude_to_db(S, ref=np.max)

        # Check the size
        assert S_db_mel.shape[1] >= 1024

        S_db_mel = get_fixed_window(S_db_mel)

        # (for my own) sanity check
        assert S_db_mel.shape[1] == 1024
        assert S_db_mel.shape[0] == n_mels

        # Must explicity convert to float 32 because somewhere in the augmentation process it creates float64 and it's a no-no for the inference with the model
        augmentations.append(S_db_mel.astype(np.float32))

        # out_file = f"{genre_folder}/{fname[:-4]}-aug-{i}.npy"

        # processed_files["path"].append(out_file)
        # processed_files["genre"].append(genre)
        # processed_files["subset"].append(subset)
        # processed_files["original"].append(fname)
    # 1/0
    return augmentations
