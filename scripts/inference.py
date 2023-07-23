"""
🦉🦉🦉
DON'T LOOK AT ME, I'M PROBABLY DEPRECATED!!!
🦉🦉🦉
"""
import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + "/../src")

from utils import get_effnet_b1, inference_one

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("audio_file", type=str, help="The audio file to be analyzed")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint of the trained model",
        required=True,
    )

    parser.add_argument(
        "--n-augmentations",
        type=int,
        default=0,
        help="Number of augmentations to be performed for inference",
    )
    parser.add_argument(
        "--decision-strategy",
        type=str,
        default="majority",
        help="How to decide on a label for multiple predictions",
    )
    # TODO add this to allow non-cuda infqerence
    # parser.add_argument('--cuda', type=str, description="Path to the checkpoint of the trained model")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = get_effnet_b1(weights_path=args.checkpoint)
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
