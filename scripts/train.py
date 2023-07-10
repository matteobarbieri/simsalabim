import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + "/../src")

from data import NpyDataset
from models import LitResnet
from utils import get_model, get_transforms

from torch import nn

from torchvision.models import resnet18

import lightning.pytorch as pl

from torch.utils.data import DataLoader


def main():
    train_transforms = get_transforms()

    dataset_train = NpyDataset(
        "data/gtzan_processed",
        "data/gtzan_processed/metadata.csv",
        "train",
        transform=train_transforms,
    )
    dataset_val = NpyDataset(
        "data/gtzan_processed",
        "data/gtzan_processed/metadata.csv",
        "test",
        transform=train_transforms,
    )

    train_loader = DataLoader(dataset_train, batch_size=64)
    val_loader = DataLoader(dataset_val, batch_size=64)

    # net = resnet18(pretrained=True)
    # # Required layer change for 1-dim input
    # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # net.fc = nn.Linear(512 * 1, 10)

    # # init the lit module
    # litnet = LitResnet(net)

    litnet = get_model(eval=False, pretrained=True)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(
        model=litnet, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    main()
