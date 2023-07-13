#!/usr/bin/env python
# coding: utf-8


# Misc
import os, sys
from glob import glob

sys.path.append("../src")

from data_augmentation import (
    time_stretch_random,
    shift_pitch_random,
    reverb_random,
    noise_random,
    get_fixed_window,
)

from sklearn.model_selection import train_test_split, StratifiedKFold

# import random


# In[4]:


# Dataframes and such
import pandas as pd
import numpy as np


# Audio stuff
import librosa


from tqdm import tqdm


def main():
    DATA_FOLDER = "../data/gtzan"

    # ## Sample audio files

    # In[10]:

    # Had to remove jazz file #54 because of corruption, apparently

    # In[11]:

    # Make a list of all the wav files in the dataset and store them in a variable
    audio_files = glob(f"{DATA_FOLDER}/*/*.wav")

    # ### MEL Spectrograms

    # In[13]:

    # Parameters for mel spectrograms generation
    hop_length = 512

    n_fft = 2048
    # n_mels = 256
    n_mels = int(sys.argv[1])

    # ## Dataset creation

    # In[14]:

    wav_files = {
        "path": [],
        "genre": [],
    }

    for af in tqdm(audio_files):
        af_arr = af.split("/")
        genre = af_arr[-2]
        fname = af_arr[-1]

        out_file = f"{genre}/{fname}"

        wav_files["path"].append(out_file)
        wav_files["genre"].append(genre)

    df = pd.DataFrame(wav_files)

    # ### Do the split first

    # In[15]:

    # 10% for test
    test_size = 0.1

    # In[16]:

    skf = StratifiedKFold()

    df_full = None

    for fold, (_, test_index) in enumerate(skf.split(df, df["genre"])):
        df_test = df.iloc[test_index, :]

        df_test["subset"] = f"fold_{fold}"
        df_full = pd.concat((df_full, df_test))

    df_full.head(n=10)

    # In[92]:

    VARIATIONS_PER_SONG = 10

    OUT_FOLDER = f"../data/gtzan_augmented_{n_mels}_x{VARIATIONS_PER_SONG}_v2"

    # In[18]:

    # In[56]:

    # In[94]:

    processed_files = {
        "path": [],
        "genre": [],
        "subset": [],
        "original": [],
        # "fold": [],
    }

    means = []
    stds = []

    for _, r in tqdm(df_full.iterrows(), total=len(df_full)):
        #     print(r)

        genre = r["genre"]
        fname = r["path"].split("/")[-1]
        subset = r["subset"]

        genre_folder = f"{OUT_FOLDER}/{genre}"

        os.makedirs(genre_folder, exist_ok=True)

        # Load audio file and create spectrogram
        y, sr = librosa.load(os.path.join(DATA_FOLDER, r["path"]))

        # First save the spectrogram of the original version of the file
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length
        )

        S_db_mel = librosa.amplitude_to_db(S, ref=np.max)

        S_db_mel = get_fixed_window(S_db_mel)

        # (for my own) sanity check
        assert S_db_mel.shape[1] == 1024
        assert S_db_mel.shape[0] == n_mels

        if subset != "test":
            means.append(S_db_mel.mean())
            stds.append(S_db_mel.std())

        out_file = f"{genre_folder}/{fname[:-4]}-orig.npy"

        processed_files["path"].append(out_file)
        processed_files["genre"].append(genre)
        processed_files["subset"].append(subset)
        processed_files["original"].append(fname)

        np.save(out_file, S_db_mel)

        # Now Create variations by adding pitch_shift and other things

        for i in range(1, VARIATIONS_PER_SONG + 1):
            y_aug = y  # This is just so that I can move stuff up and down without going crazy

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

            out_file = f"{genre_folder}/{fname[:-4]}-aug-{i}.npy"

            processed_files["path"].append(out_file)
            processed_files["genre"].append(genre)
            processed_files["subset"].append(subset)
            processed_files["original"].append(fname)

            np.save(out_file, S_db_mel)

    df_aug = pd.DataFrame(processed_files)
    df_aug["path"] = df_aug["path"].apply(lambda x: x[len(OUT_FOLDER) + 1 :])
    df_aug.to_csv(f"{OUT_FOLDER}/metadata.csv", index=False)

    # In[102]:

    np.mean(means)

    # In[104]:

    np.mean(stds)

    # In[ ]:


main()
