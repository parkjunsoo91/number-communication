#pylint:disable=E1101

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import argparse
from models import MyModel
from math_dataset import MyDataset


def main():
    _i, _j, _k = 2,3,3
    dataset = MyDataset(_i,_j,_k)

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    #batch, input, hidden, output
    N, D_in, H, D_out = 10, _i+_j+_k, 16, _i*_j*_k
    msg_len = 10

    x, y = dataset.get_frame()
    x = torch.tensor(x, dtype=dtype, device=device)
    #x = torch.cat((x,x,x,x,x),0)
    y = torch.tensor(y, dtype=torch.long, device=device).squeeze()
    #y = torch.cat((y,y,y,y,y),0)
    print(x.size(), y.size())
    #x = torch.zeros(N, D_in, device=device, dtype=dtype)
    #y = torch.zeros(N, device=device, dtype=dtype)

    model = MyModel(D_in, H, D_out)
    #model = torch.nn.Linear(D_in, D_out)

    loss_fn = torch.nn.CrossEntropyLoss(reduce=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for t in range(10001):
        if True: #reinforce
            y_pred = model(x)
            probs = F.softmax(y_pred, dim=1)
            m = Categorical(probs)
            action = m.sample()
            reward = torch.eq(action, y).to(torch.float)
            reward = (reward - reward.mean())
            loss = -m.log_prob(action) * reward
            model.zero_grad()
            loss.sum().backward()
            #loss.backward(loss)
            optimizer.step()
        
        elif True:
            y_pred = model(x)
        
        else: # supervised
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if t % 100 == 0:
            with torch.no_grad():
                y_pred = model(x)
                eq = torch.eq(torch.argmax(y_pred, dim=1), y)
                print("t: {}, acc: {}/{} = {}".format(t, torch.sum(eq).item(), eq.numel(), torch.sum(eq).item() / eq.numel()))


        torch.save({'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, "checkpoints.tar")

if __name__ == "__main__":
    main()
        



    # model3 = MyModel(D_in, H, D_out)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # checkpoint = torch.load("checkpoints.tar")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # print(model.state_dict())
    # print(optimizer.state_dict())

    # PATH = "model.pt"
    # torch.save(model.state_dict(), PATH)

    # model2 = MyModel(D_in, H, D_out)
    # model.load_state_dict(torch.load(PATH))
    # model.eval() # for dropout and BN