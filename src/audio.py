"""
Aux functions that have to do with audio manipulation,
such as creation of MEL spectrograms etc.
"""

import numpy as np
import librosa


def wav_to_dfmel(
    wav_path,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=256,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant",
    power=2.0,
) -> np.ndarray:
    y, sr = librosa.load(wav_path)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        power=power,
    )

    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)

    return S_db_mel
