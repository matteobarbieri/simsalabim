import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + "/../src")

from utils import get_model, get_transforms

import torch

from audio import wav_to_dfmel

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
    # TODO add this to allow non-cuda inference
    # parser.add_argument('--cuda', type=str, description="Path to the checkpoint of the trained model")

    return parser.parse_args()


def main():
    args = parse_args()

    S_db_mel = wav_to_dfmel(args.audio_file)

    weights_path = args.checkpoint

    litnet = get_model(weights_path=weights_path)

    train_transforms = get_transforms()

    # TODO change this to allow non-cuda
    y_hat = litnet(train_transforms(S_db_mel).to("cuda").unsqueeze(0))

    cat_idx = torch.argmax(y_hat).item()

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
