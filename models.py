import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

class Sender(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Sender, self).__init__()
        self.linear1 = torch.nn.Linear(D_in,D_out)
    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred

class Receiver(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Receiver, self).__init__()
        self.linear1 = nn.Linear(D_in, D_out)
    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred