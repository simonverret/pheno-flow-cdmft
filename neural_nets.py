import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeToOneNN(nn.Module):
    def __init__(self, hsize=16):
        super().__init__()
        self.linear1 = nn.Linear(3,hsize)
        self.linear2 = nn.Linear(hsize,1)

        ## initialization of weights
        nn.init.xavier_uniform(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)
        nn.init.xavier_uniform(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)

    def forward(self, kx, ky, w):
        cat = torch.stack([kx,ky,w], dim=-1)
        out = F.relu(self.linear1(cat))
        out = F.relu(self.linear2(out))
        out = out.squeeze(-1)
        return out




