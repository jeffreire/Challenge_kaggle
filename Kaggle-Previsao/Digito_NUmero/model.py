import pandas as pd
import numpy as np
from torch import nn
import torch

class Modelo( nn.Module ):
    def __init__( self ):
        super.__init__()
        
        self.layer_input = nn.Linear( 785, 350 )
        self.function_activation = nn.ReLU()
        self.dropout = nn.Dropout( 0.2 )
        self.layer_hidden = nn.Linear( 350, 110 )
        self.function_activation_1 = nn.ReLU()
        self.layer_output = nn.











torch.manual_seed( 123 )

train = pd.read_csv('Kaggle-Previsao/Digito_NUmero/train.csv')
test = pd.read_csv('Kaggle-Previsao/Digito_NUmero/test.csv')