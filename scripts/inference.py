import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + '/../src')

from utils import get_model, get_transforms

import torch

from audio import wav_to_dfmel

def main():

    S_db_mel = wav_to_dfmel(sys.argv[1])

    weights_path = 'lightning_logs/version_16/checkpoints/epoch=49-step=650.ckpt'

    litnet = get_model(weights_path=weights_path)

    train_transforms = get_transforms()

    y_hat = litnet(train_transforms(S_db_mel).to('cuda').unsqueeze(0))

    cat_idx = torch.argmax(y_hat).item()

    # Hardcode categories because I'm lazy and it works for now
    cat_to_int = {
        'blues': 0, 
        'classical': 1, 
        'country': 2, 
        'disco': 3, 
        'hiphop': 4, 
        'jazz': 5, 
        'metal': 6, 
        'pop': 7, 
        'reggae': 8, 
        'rock': 9
    }

    int_to_cat = {v: k for k, v in cat_to_int.items()}

    predicted_genre = int_to_cat[cat_idx]
    
    print(f"Predicted genre: {predicted_genre}")


if __name__ == '__main__':
    main()