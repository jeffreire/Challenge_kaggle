import pandas as pd
import numpy as np
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_input = nn.Linear(9, 6)
        nn.init.uniform(self.f_input.weight)
        self.f_h1 = nn.Linear(6, 6)
        nn.init.uniform(self.f_h1.weight)
        self.f_h2 = nn.Linear(6, 1)
        self.f_acvation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.f_output = nn.Sigmoid()

    def forward(self, x):
        x = self.f_input(x)
        x = self.f_acvation(x)
        x = self.dropout(x)
        x = self.f_h1(x)
        x = self.f_acvation(x)
        x = self.dropout(x)
        x = self.f_h2(x)
        x = self.f_output(x)
        return x
    
    def load(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
        



