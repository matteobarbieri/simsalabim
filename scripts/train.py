import os, sys

# Add src folder in root repo to Python path
sys.path.append(os.path.dirname(__file__) + "/../src")

from utils import get_model, get_datasets

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


from torch.utils.data import DataLoader

import logging

logging.basicConfig(level=logging.INFO)

from sacred import Experiment

from sacred.observers import MongoObserver


ex = Experiment("GTZAN-test")

ex.observers.append(
    MongoObserver(url="mongodb://sample:password@localhost", db_name="db")
)


@ex.config
def my_config():
    lr: float = 2e-4  # Learning rate
    epochs: int = 3  # Number of epochs
    batch_size: int = 64  # Number of samples per batch
    tag: str = None


@ex.automain
def train(_run, lr, epochs, batch_size, tag):
    
    # Available options:
    # * 'gtzan_processed'
    # * 'gtzan_augmented_256_test'

    dataset_train, dataset_val = get_datasets('gtzan_processed')

    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=7)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=7)

    litnet = get_model(eval=False, pretrained=True, lr=lr, _run=_run)

    checkpoint_callback = ModelCheckpoint(
        dirpath="model_checkpoints",
        save_top_k=2,
        monitor="val.loss",
        filename=f"{tag}-" + "{epoch}",
    )

    early_stopping_callback = EarlyStopping(monitor="val.loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback,
        ],
    )

    trainer.fit(
        model=litnet, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
