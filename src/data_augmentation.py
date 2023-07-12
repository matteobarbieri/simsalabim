__all__ = ['time_stretch_random', 'shift_pitch_random', 'reverb_random', 'get_fixed_window']

import random

import numpy as np

import pedalboard
reverb = pedalboard.Reverb()

from librosa.effects import time_stretch


def time_stretch_random(y: np.ndarray, sr: int = -1, intensity: float = 1.0):
    # param sr is kept so that signature is the same for all functions
    # Stretch tie between -20% and +25%
    rate = 1 + intensity*random.uniform(-0.2, 0.25)
    
    # rate = 1.25 # for debugging purposes, to compute max stretch

    y_stretch = time_stretch(y, rate=rate)

    return y_stretch


from librosa.effects import pitch_shift


def shift_pitch_random(y: np.ndarray, sr: int, intensity: float = 1.0):
    # Shift between three semitones up or down
    # 
    n_steps = max(0.5, intensity)*random.uniform(-1.5, 1.5)

    y_pitch = pitch_shift(y, sr=sr, n_steps=n_steps)

    return y_pitch


def reverb_random(y: np.ndarray, sr: int, intensity: float = 1.0):
    """
    Add some amount of reverb to the clip
    """

    # Reverb-specific parameters
    reverb.room_size = intensity*random.uniform(0, 0.6)
    reverb.wet_level = intensity*random.uniform(0.33, 0.5)

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


def get_fixed_window(S_db_mel, width=1024):
    # Create a window exactly 1024 wide
    # This is needed because time stretching might make the audio shorter
    max_starting_point = S_db_mel.shape[1] - width
    start_j = random.randint(0, max_starting_point)

    return S_db_mel[:, start_j : start_j + width]