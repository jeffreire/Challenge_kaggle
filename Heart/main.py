import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import Model
from train import Trainer

dataset = pd.read_csv('Data/heart.csv')
INPUTS = dataset.iloc[:, :-1]
TARGETS = dataset.iloc[:, -1]

INPUTS = torch.tensor(np.array(INPUTS), dtype= torch.float)
TARGETS = torch.tensor(np.array(TARGETS), dtype= torch.float)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(INPUTS, TARGETS), batch_size= 10, shuffle= True)

classification = Model()
classification.load('C:/Users/jefferson.maria/Desktop/Challenge_kaggle/checkspoint.pth')

criterion = nn.BCELoss()

optimizer = torch.optim.Adamax(classification.parameters(), lr = 1e-4,  weight_decay= 1e-6)

# # definindo a epocha
ephoca= 100

# # Treinamento do modelo
Trainer(ephoca, INPUTS, TARGETS, train_loader, optimizer, classification, criterion)

