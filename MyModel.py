from pytorch_lightning import LightningModule
from torch import nn
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Define PyTorch model
        classes = 5
        features = 300 * 2
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, 1024),
            nn.Linear(1024, 256),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, classes),
        )
        self.accuracy = Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.accuracy(preds, y)
        self.log(f"train_acc", self.accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, print_str="val"):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log(f"{print_str}_loss", loss, prog_bar=True)
        self.log(f"{print_str}_acc", self.accuracy, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)




