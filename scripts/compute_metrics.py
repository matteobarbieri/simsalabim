"""
Compute metrics for the entire dataset, averaging results for a k-fold cross validation split.
"""

import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + "/../src")

from utils import get_model, get_dataset, inference_one

from tqdm import tqdm

from sklearn.metrics import accuracy_score

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

    # parser.add_argument(
    #     "--decision-strategy",
    #     type=str,
    #     default="majority",
    #     help="How to decide on a label for multiple predictions",
    # )

    # TODO add this to allow non-cuda inference
    # parser.add_argument('--cuda', type=str, description="Path to the checkpoint of the trained model")

    return parser.parse_args()


def evaluate_fold(audio_root_dir: str, dataset: str, f: int, n_augmentations: int):
    """
    dataset: str
        The dataset version to use for the evaluation
    f: int
        The fold index to evaluate
    """

    ds = get_dataset(dataset, subset=f"fold_{f}", original_only=True)

    # TODO XXX must change to retrieve the actual model trained on the rest of the data
    model = get_model(weights_path="model_checkpoints/deleteme_test_model.ckpt")

    ys_true = []
    ys_hat = []

    # Use the metadata df directly, so that inference-level augmentations can be used
    for _, r in tqdm(ds.metadata.iloc[:2,].iterrows(), total=len(ds.metadata)):
        audio_file_path = os.path.join(audio_root_dir, r["genre"], r["original"])

        y_true = ds.cat_to_int[r["genre"]]
        y_hat = inference_one(audio_file_path, n_augmentations, model)

        # Append to a list
        ys_true.append(y_true)
        ys_hat.append(y_hat)

    acc = accuracy_score(ys_true, ys_hat)

    # TODO produce other metrics
    print(f"Accuracy on fold {f}: {acc}")


def main() -> None:
    args = parse_args()

    for f in range(args.k):
        evaluate_fold(args.audio_root_dir, args.dataset, f, args.n_augmentations)
        break

    return

    # model = get_model(weights_path=args.checkpoint)

    cat_idx = inference_one(args.audio_file, args.n_augmentations, model)

    # Hardcode categories because I'm lazy and it works for now
    cat_to_int = {
        "blues": 0,
        "classical": 1,
        "country": 2,
        "disco": 3,
        "hiphop": 4,
        "jazz": 5,
        "metal": 6,
        "pop": 7,
        "reggae": 8,
        "rock": 9,
    }

    int_to_cat = {v: k for k, v in cat_to_int.items()}

    predicted_genre = int_to_cat[cat_idx]

    print(f"Predicted genre: {predicted_genre}")


if __name__ == "__main__":
    main()
