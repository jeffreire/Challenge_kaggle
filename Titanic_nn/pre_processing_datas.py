import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from sklearn.model_selection import GridSearchCV
from model import Model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from skorch import NeuralNetBinaryClassifier
from train import Trainer

class Processando:
    def __init__(self, data):
        self.dataset = data

    def drop_coluns(self, coluns):
        return self.dataset.drop(coluns, axis= 1, inplace= True)
    
    def _filling_numeric_columns(self, colum):
        mean = self.dataset[colum].mean()
        mean = math.floor(mean)
        self.dataset.update(self.dataset[colum].fillna(mean))

    def _generate_one_hot_vector_from_categorical_label(self, data, categorical_index):
        categorical_data = data.iloc[ :, categorical_index ]

        labelEncoder = LabelEncoder()
        categorical_data = labelEncoder.fit_transform( categorical_data )

        oneHotEncoder = OneHotEncoder()
        categorical_data = oneHotEncoder.fit_transform( categorical_data.reshape(-1,1) ).toarray()

        # dummy variable trap avoiding
        categorical_data = np.delete( categorical_data, np.s_[0], axis=1)

        return categorical_data
    
    def _generate_scalar_features(self, data, features_index):
        scalar_data = data.iloc[ :, features_index ]

        # normalization
        sc = StandardScaler()
        scalar_data = sc.fit_transform( scalar_data )

        return scalar_data

dataset = pd.read_csv('Data/Train.csv')
pre_processig = Processando(dataset)
pre_processig.drop_coluns(['Name', 'Ticket', 'Cabin'])

pre_processig._filling_numeric_columns(['Age'])
dataset = dataset.dropna()

INPUTS = dataset.iloc[:, [0,2,3,4,5,6,7,8] ]
TARGETS = dataset.iloc[:, 1]

SEX = pre_processig._generate_one_hot_vector_from_categorical_label(INPUTS, 2)
EMBARKED = pre_processig._generate_one_hot_vector_from_categorical_label(INPUTS, 7)
scalar_data = pre_processig._generate_scalar_features(INPUTS, [0,1,3,4,5,6])
INPUTS = np.concatenate((SEX, EMBARKED, scalar_data), axis= 1)

INPUTS =  np.array(INPUTS, dtype= 'float32')
TARGETS = np.array(TARGETS, dtype= 'float32')

INPUTS = torch.tensor(np.array(INPUTS), dtype= torch.float)
TARGETS = torch.tensor(np.array(TARGETS), dtype= torch.float)

# # Transformando os dados em tensores, concatenando os dois tensores me um so
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(INPUTS, TARGETS), batch_size= 320, shuffle= True)

# # Atribuindo o model a variavel
classification = Model()

# # # Carregando os pesos salvos e aplicando no modelo
# classification.load('C:/Users/jefferson.maria/Desktop/Challenge_kaggle/checkspoint.pth')

# # Atribuindo o criterio de calculo dos erros
criterion = nn.CrossEntropyLoss()

# # Optimizando, utilizando a class Adam passando por parametro o modelo, taxa de aprendizaggem e a taxa que sera calculada os pesos
optimizer = torch.optim.Adam(classification.parameters(), lr = 1e-4, weight_decay = 1e-6)

# # definindo a epocha
ephoca= 1000

# # Treinamento do modelo
Trainer(ephoca, INPUTS, TARGETS, train_loader, optimizer, classification, criterion)

classification.checkpoint('checkspoint.pth')