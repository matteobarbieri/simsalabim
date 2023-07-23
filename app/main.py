import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + "/../src")

from utils import get_effnet_b1, get_transforms

import torch

from audio import wav_to_dfmel

from fastapi import FastAPI, UploadFile

app = FastAPI()

weights_path = "/workspace/model/model.ckpt"

litnet = get_effnet_b1(weights_path=weights_path)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    # print(file.filename)

    S_db_mel = wav_to_dfmel(file.file)

    global litnet

    train_transforms = get_transforms()

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

    return {"predicted_genre": predicted_genre}
