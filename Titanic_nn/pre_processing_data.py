# importando as libs
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from main import Main_neural

# Atribuindo as variaves com os dados do dataset, camada de entrada e o resultado esperado, formato de pandas
INPUTS = pd.read_csv('Data/inputs_breast.csv')
TARGETS = pd.read_csv('Data/outputs_breast.csv')

# Convertendo os dados de pandas para tensor
INPUTS = torch.tensor(np.array(INPUTS), dtype= torch.float)
TARGETS = torch.tensor(np.array(TARGETS), dtype= torch.float)
BATCH_SIZE = 10
LR = 0.001

# Trasnformando em dados tensores
TRAIN_LOADER = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(INPUTS, TARGETS), batch_size = BATCH_SIZE, shuffle = True)

Main_neural(INPUTS, TARGETS, LR, BATCH_SIZE, TRAIN_LOADER)
