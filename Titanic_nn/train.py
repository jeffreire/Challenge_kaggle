# 1 - Importando as bibliotecas
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# FIaxando os valores aleatorios obtidos
np.random.seed(123)
torch.manual_seed(123)

# Classe que ira treinar a rede neural 
class Trainer:
    def __init__(self, epocha, inputs, targets, train_loader, optimizer, classification, criterion):

        for i in range(epocha):
            running_loss = 0

            for data in train_loader:
                # Separando as entradas e os targets do tensor train loader
                inputs, labels = data
                inputs = torch.tensor(inputs)
                optimizer.zero_grad()

                # Passando as entradas no modelo
                outputs = classification(inputs)
                # atribindo o resultado e o esperado na cross entropy para calcular o erro
                loss = criterion(outputs, labels)
                # implementando o backpropagation para atualizar os erros
                loss.backward()
                #Passando o proximo passo 
                optimizer.step()

                running_loss += loss.item()
            print('Ephoca %3d: loss %.5f' % (i + 1, running_loss/len(train_loader)))
