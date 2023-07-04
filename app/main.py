from typing import Union

import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + '/../src')

from torchvision.models import resnet18

# import lightning.pytorch as pl

import torchvision.transforms as transforms

from models import LitResnet

from torch import nn
import torch

from audio import wav_to_dfmel

from fastapi import FastAPI, UploadFile

app = FastAPI()

def get_classifier():
    # Inference stuff
    net = resnet18()

    # Required layer change for 1-dim input
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(512 * 1, 10)

    litnet = LitResnet.load_from_checkpoint(
        'lightning_logs/version_16/checkpoints/epoch=49-step=650.ckpt', net=net)

    litnet.eval()

    return litnet

litnet = get_classifier()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    print(file.filename)

    S_db_mel = wav_to_dfmel(file.file)

    global litnet

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 1290), transforms.InterpolationMode.NEAREST),
        transforms.Normalize(mean=[-80.0], std=[80])
    ])

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

    return {"predicted_genre": predicted_genre}