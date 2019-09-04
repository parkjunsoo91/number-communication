import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

from nac import NAC
from nalu import NALU

device = torch.device("cuda:0")

step = 0.1
x = [e*step for e in range(100)] # 20 * 10
y = [e for e in x]
x_test = [e*step for e in range(200)]
y_test = [e for e in x_test]

print(x)
print(y)

x = torch.tensor(x).to(device, torch.float)
y = torch.tensor(y).to(device, torch.float)
x_test = torch.tensor(x_test).to(device, torch.float)
y_test = torch.tensor(y_test).to(device, torch.float)


hidden_dim = 8

seq1 = nn.Sequential()
model = nn.Sequential(
    #NAC(1,1,hidden_dim,1)
    #NALU(1,1,hidden_dim,1),

    nn.Linear(1,hidden_dim),
    nn.SELU(),
    nn.Linear(hidden_dim,hidden_dim),
    nn.SELU(),
    nn.Linear(hidden_dim,1)
)
model.cuda()

criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for t in range(10000):
    y_pred = model(x.unsqueeze(-1)).squeeze(-1)
    # print(y_pred.size())
    # print(y.size())
    #exit()
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if t % 10 != 0:
        continue

    with torch.no_grad():
        y_pred = model(x_test.unsqueeze(-1)).squeeze(-1)
        loss = criterion(y_pred, y_test)
        print("epoch", t, "loss", loss.item())
        
        #for i in range(len(x)):
        #    print("{:.1f} , {}".format(x[i].item(), y_pred[i].item()))
        
        plt.clf()
        plt.title("epoch {}".format(t))
        plt.plot(x_test.cpu().numpy(), y_pred.cpu().numpy())
        plt.plot(x_test.cpu().numpy(), y_test.cpu().numpy())
        plt.pause(0.001)
