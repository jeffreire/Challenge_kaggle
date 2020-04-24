import torch
import pandas as pd
import numpy as np
from model import Model
from sklearn.metrics import accuracy_score
from torch.nn import functional as F

classification = Model()
classification.load('checkspoint.pth')

dataset = pd.read_csv('Data/heart.csv')
INPUTS = dataset.iloc[:, :-1]
TARGETS = dataset.iloc[:, -1]

INPUTS = torch.tensor(np.array(INPUTS), dtype= torch.float)
TARGETS = torch.tensor(np.array(TARGETS), dtype= torch.float)

classification.eval()
previsao = classification.forward(INPUTS).detach()
x = F.binary_cross_entropy(previsao, TARGETS)

print(accuracy_score(TARGETS.numpy(), (previsao > 0.5).numpy()))