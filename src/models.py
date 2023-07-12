import torch
from torch import optim, nn

import lightning.pytorch as pl

from focal_loss import FocalLoss

from sklearn.metrics import accuracy_score, confusion_matrix


# define the LightningModule
class LitResnet(pl.LightningModule):
    def __init__(self, net, lr=2e-4, _run=None):
        super().__init__()
        self.net = net

        # self.loss = nn.CrossEntropyLoss(reduction="sum")
        # self.loss = FocalLoss(reduction="sum")
        self.loss = FocalLoss(size_average=False)

        # Keep track of how many samples are in the training and validation set (needed for comparing apples to apples)
        self.n_train = 0
        self.n_val = 0

        self.train_epoch_loss = 0
        self.val_epoch_loss = 0

        self.train_y = []
        self.train_y_hat = []

        self.val_y = []
        self.val_y_hat = []

        self.lr = lr
        self._run = _run

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)

        loss = self.loss(logits, y)

        # Used later for logging
        self.n_train += len(y)
        self.train_epoch_loss += loss

        y_hat = torch.argmax(logits, axis=1)

        self.train_y.append(y)
        self.train_y_hat.append(y_hat)

        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)

        loss = self.loss(logits, y)

        # Used later for logging
        self.n_val += len(y)
        self.val_epoch_loss += loss

        y_hat = torch.argmax(logits, axis=1)

        self.val_y.append(y)
        self.val_y_hat.append(y_hat)

        # Logging to TensorBoard (if installed) by default
        # Logging mostly for early stopping
        self.log("val.loss", loss)
        return loss

    def on_validation_epoch_end(self):
        # This is needed because of the first validation lap that lightning does for reasons
        if self.n_train > 0:
            train_y = torch.cat(self.train_y)
            train_y_hat = torch.cat(self.train_y_hat)

            val_y = torch.cat(self.val_y)
            val_y_hat = torch.cat(self.val_y_hat)

            train_acc = accuracy_score(train_y.cpu(), train_y_hat.cpu())
            val_acc = accuracy_score(val_y.cpu(), val_y_hat.cpu())

            print("Training:")
            print(confusion_matrix(train_y.cpu(), train_y_hat.cpu()))

            print("Validation:")
            print(confusion_matrix(val_y.cpu(), val_y_hat.cpu()))

            self._run.log_scalar(
                "train.loss",
                self.train_epoch_loss.item() / self.n_train,
                self.current_epoch,
            )

            self._run.log_scalar(
                "val.loss", self.val_epoch_loss.item() / self.n_val, self.current_epoch
            )

            self._run.log_scalar("train.accuracy", train_acc, self.current_epoch)
            self._run.log_scalar("val.accuracy", val_acc, self.current_epoch)

            print(80 * "=")

        # Reset counters and stuff
        self.train_y.clear()
        self.train_y_hat.clear()
        self.val_y.clear()
        self.val_y_hat.clear()

        self.n_train = 0
        self.n_val = 0

        self.train_epoch_loss = 0
        self.val_epoch_loss = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        return optimizer
