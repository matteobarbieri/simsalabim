import os, sys

import pandas as pd
import numpy as np

import torch
from torch import optim, nn, utils, Tensor

from torchvision.models import resnet18

import lightning.pytorch as pl

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

# from tofunctional import _interpolation_modes_from_int, InterpolationMode

from sklearn.metrics import accuracy_score, confusion_matrix

class NpyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, metadata_file, subset, subset_col='subset', transform=None):
        """
        Arguments:
            
        """
        self.metadata = pd.read_csv(metadata_file)
        
        # Keep only rows for subset
        self.metadata = self.metadata[self.metadata[subset_col] == subset]
        
        self.subset = subset
        
        
        self.root_dir = root_dir
        self.transform = transform
        
        categories = self.metadata.genre.unique()
        self.cat_to_int = {c: i for i, c in enumerate(sorted(categories))}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
            
        row = self.metadata.iloc[idx]
        sample = np.load(os.path.join(self.root_dir, row['path']))
        genre = row['genre']
        
        label = self.cat_to_int[genre]
        
        if self.transform:
            sample = self.transform(sample[:, :])

        return sample, label


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 1290), transforms.InterpolationMode.NEAREST),
    transforms.Normalize(mean=[-80.0], std=[80])
])

dataset_train = NpyDataset("data/gtzan_processed", "data/gtzan_processed/metadata.csv", 'train', transform=train_transforms)
dataset_val = NpyDataset("data/gtzan_processed", "data/gtzan_processed/metadata.csv", 'test', transform=train_transforms)

train_loader = DataLoader(dataset_train, batch_size=64)
val_loader = DataLoader(dataset_val, batch_size=64)

net = resnet18(pretrained=True)
# Required layer change for 1-dim input
net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
net.fc = nn.Linear(512 * 1, 10)






# define the LightningModule
class LitResnet(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
#         self.loss = nn.CrossEntropyLoss()
        self.loss = nn.NLLLoss()
    
        self.val_y = []
        self.val_y_hat = []

        self.lr =2e-4
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        logits = self.net(x)
        
        predicted = torch.argmax(logits, axis=1)
        
        loss = self.loss(logits, y)
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)
        
        loss = self.loss(logits, y)
        
        y_hat = torch.argmax(logits, axis=1)
        
        self.val_y.append(y)
        self.val_y_hat.append(y_hat)
        
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        return loss
        
        
    def on_validation_epoch_end(self):
        
        y = torch.cat(self.val_y)
        y_hat = torch.cat(self.val_y_hat)
        
        acc = accuracy_score(y.cpu(), y_hat.cpu())
        print(confusion_matrix(y.cpu(), y_hat.cpu()))
        
        # do something with all preds
        
        self.val_y.clear()
        self.val_y_hat.clear()
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        return optimizer

# init the autoencoder
litnet = LitResnet(net)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model=litnet, train_dataloaders=train_loader, val_dataloaders=val_loader)