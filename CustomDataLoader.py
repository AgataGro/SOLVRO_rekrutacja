import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningDataModule


class CustomDataLoader(LightningDataModule):
    def __init__(self):
        super().__init__()
        ytrain = np.load('y_train.npy')
        train_labels = np.array([])
        for i in range(0, 49000):
            train_labels = np.append(train_labels, np.argmax(ytrain[i]))

        yval = np.load('y_val.npy')
        val_labels = np.array([])
        for i in range(0, 21000):
            val_labels = np.append(val_labels, np.argmax(yval[i]))

        self.train_dataset = TensorDataset(torch.from_numpy(np.load('X_train.npy')).float(),
                                           torch.tensor(train_labels).long())
        self.val_dataset = TensorDataset(torch.from_numpy(np.load('X_val.npy')).float(),
                                         torch.tensor(val_labels).long())
        self.pred_dataset = torch.from_numpy(np.load('X_test.npy')).float()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=3, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=3, shuffle=False, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=1, shuffle=False, num_workers=0)
