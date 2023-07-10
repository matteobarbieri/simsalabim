import torch
from torch import optim, nn

import lightning.pytorch as pl

from sklearn.metrics import accuracy_score, confusion_matrix


# define the LightningModule
class LitResnet(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        #         self.loss = nn.CrossEntropyLoss()
        self.loss = nn.NLLLoss()

        self.val_y = []
        self.val_y_hat = []

        self.lr = 2e-4

    def forward(self, x):
        return self.net(x)

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
