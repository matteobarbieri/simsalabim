import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + "/../src")

from utils import get_model, get_transforms

import torch

from audio import wav_to_dfmel

import argparse

from data_augmentation import get_fixed_window, get_n_variations


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("audio_file", type=str, help="The audio file to be analyzed")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint of the trained model",
        required=True,
    )
    
    parser.add_argument('--n-augmentations', type=int, default=0, help="Number of augmentations to be performed for inference")
    parser.add_argument('--decision-strategy', type=str, default='majority', help="How to decide on a label for multiple predictions")
    # TODO add this to allow non-cuda infqerence
    # parser.add_argument('--cuda', type=str, description="Path to the checkpoint of the trained model")

    return parser.parse_args()


def main():
    args = parse_args()

    # The original audio file
    S_db_mel = wav_to_dfmel(args.audio_file)

    S_db_mel = get_fixed_window(S_db_mel)


    variation_db_mels = get_n_variations(args.audio_file, args.n_augmentations, return_original=True)

    train_transforms = get_transforms()

    input_tensors = torch.stack([train_transforms(x) for x in variation_db_mels]).to("cuda")

    litnet = get_model(weights_path=args.checkpoint)

    # TODO change this to allow non-cuda
    # y_hat = litnet(train_transforms(S_db_mel).to("cuda").unsqueeze(0))
    y_hat = litnet(input_tensors.to("cuda"))

    print(y_hat)

    cat_idx = torch.argmax(y_hat, dim=1)

    print(cat_idx)
    print(cat_idx.cpu())
    print(cat_idx.cpu().numpy())

    return



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
