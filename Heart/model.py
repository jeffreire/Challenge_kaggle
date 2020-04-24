import torch
import torch.nn as nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_input = nn.Linear(13, 7)
        self.dropout = nn.Dropout(0.2)
        self.f_h1 = nn.Linear(7, 7)
        self.f_out = nn.Linear(7, 1)

        nn.init.xavier_uniform_(self.f_input.weight)
        nn.init.xavier_uniform_(self.f_h1.weight)
        nn.init.xavier_uniform_(self.f_out.weight)

    def forward(self, x):
        x = F.relu(self.f_input(x))
        x = F.relu(self.f_h1(x))
        prob = F.sigmoid(self.f_out(x))
        return prob

    def load(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)