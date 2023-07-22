"""
Compute metrics for the entire dataset, averaging results for a k-fold cross validation split.
"""

import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + "/../src")

from utils import get_effnet_b1, get_dataset, inference_one
import numpy as np

from tqdm import tqdm

from glob import glob

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from typing import Tuple, List

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("audio_file", type=str, help="The audio file to be analyzed")

    # This must be changed since it will have to load k different models
    # parser.add_argument(
    #     "--checkpoint",
    #     type=str,
    #     help="Path to the checkpoint of the trained model",
    #     required=True,
    # )

    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset version to use",
    )

    parser.add_argument(
        "models_folder",
        type=str,
        help="Folder where the k trained models (one for each test fold) are found",
    )

    parser.add_argument(
        "--n-augmentations",
        type=int,
        default=0,
        help="Number of augmentations to be performed for inference",
    )

    parser.add_argument(
        "--audio-root-dir",
        type=str,
        help="Root folder where actual audio files are found",
        default="data/gtzan",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds used",
    )

    parser.add_argument(
        "--cm-filename",
        type=str,
        default="confusion_matrix.png",
        help="Name of the output file for the confusion matrix",
    )

    # parser.add_argument(
    #     "--decision-strategy",
    #     type=str,
    #     default="majority",
    #     help="How to decide on a label for multiple predictions",
    # )

    # TODO add this to allow non-cuda inference
    # parser.add_argument('--cuda', type=str, description="Path to the checkpoint of the trained model")

    return parser.parse_args()


def evaluate_fold(
    audio_root_dir: str, dataset: str, f: int, models_folder: str, n_augmentations: int
) -> Tuple[List, List]:
    """
    dataset: str
        The dataset version to use for the evaluation
    f: int
        The fold index to evaluate
    models_folder: str
        The folder where the 5 models are saved
    n_augmentations:
        The number of augmentations to compute for each
    """

    ds = get_dataset(dataset, subset=f"fold_{f}", original_only=True)

    # So I don't have to fish for the correct file name
    weights_path = glob(os.path.join(models_folder, f"*fold_{f}.ckpt"))[0]

    # TODO should be able to accept different model loaders
    model = get_effnet_b1(weights_path=weights_path)

    ys_true = []
    ys_hat = []

    # Use the metadata df directly, so that inference-level augmentations can be used
    for _, r in tqdm(ds.metadata.iterrows(), total=len(ds.metadata)):
        audio_file_path = os.path.join(audio_root_dir, r["genre"], r["original"])

        y_true = ds.cat_to_int[r["genre"]]
        y_hat = inference_one(audio_file_path, n_augmentations, model)

        # Append to a list
        ys_true.append(y_true)
        ys_hat.append(y_hat)

    return ys_true, ys_hat


def main() -> None:
    args = parse_args()

    ys_true = []
    ys_hat = []

    accuracies = []

    for f in range(args.k):
        yt_k, yh_k = evaluate_fold(
            args.audio_root_dir,
            args.dataset,
            f,
            args.models_folder,
            args.n_augmentations,
        )

        accuracies.append(accuracy_score(yt_k, yh_k))

        ys_true.extend(yt_k)
        ys_hat.extend(yh_k)

    print(f"Accuracy mean: {np.mean(accuracies)}")
    print(f"Accuracy std: {np.std(accuracies)}")

    labels = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]

    cm = confusion_matrix(ys_true, ys_hat)

    fig = plt.figure(figsize=(12, 12))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap="Blues", colorbar=False, xticks_rotation=45)

    fig.autofmt_xdate(bottom=0.2, rotation=30, ha="right")

    plt.tight_layout()

    plt.savefig(args.cm_filename)

    return


if __name__ == "__main__":
    main()
