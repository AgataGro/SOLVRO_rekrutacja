import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
import torch
from CustomDataLoader import CustomDataLoader
from MyModel import MyModel


# Start the Trainer
trainer = Trainer(
    max_epochs=100,
    default_root_dir="C:/Users/agata/PycharmProjects/SOLVRO_rekrutacja"
)
# Define the Model
model = MyModel()
dataloader = CustomDataLoader()
# Train the Model
trainer.fit(model, dataloader.train_dataloader(), dataloader.val_dataloader())
predictions = trainer.predict(model, dataloader.predict_dataloader())
predictions = torch.stack(predictions)
results = np.array([])
for i in range(0, 30000):
    results = np.append(results, torch.argmax(predictions[i]))
df = pd.DataFrame(results)
df.to_csv('predictions.csv')
