import pandas as pd
import numpy as np
import math
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pre_processing_datas import Processando
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from model import Model

classification = Model()
classification.load('C:/Users/jefferson.maria/Desktop/Challenge_kaggle/checkspoint.pth')

dataset = pd.read_csv('Data/test.csv')
dt = pd.read_csv('Data/train.csv')
processando = Processando(dataset)
processando.drop_coluns(['Name', 'Ticket', 'Cabin'])

processando._filling_numeric_columns('Age')
processando._filling_numeric_columns('Fare')

INPUTS = dataset.iloc[:, [0,1,2,3,4,5,6,7] ]

SEX = processando._generate_one_hot_vector_from_categorical_label(INPUTS, 2)
EMBARKED = processando._generate_one_hot_vector_from_categorical_label(INPUTS, 7)
scalar_data = processando._generate_scalar_features(INPUTS, [0,1,3,4,5,6])

INPUTS = np.concatenate((SEX, EMBARKED, scalar_data), axis= 1)
TARGETS = dt.iloc[:, 1]

INPUTS =  np.array(INPUTS, dtype= 'float32')
TARGETS = np.array(TARGETS, dtype= 'float32')


INPUTS = torch.tensor(np.array(INPUTS), dtype= torch.float)
TARGETS = torch.tensor(np.array(TARGETS), dtype= torch.float)

classification.eval()
previsao = classification.predict(INPUTS)

print(previsao)
print(accuracy_score(TARGETS.numpy(), (previsao > 0.5).numpy()))