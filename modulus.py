import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

device = torch.device("cuda:0")

step = 0.2
a = [step*e for e in range(50)]
b = [2]
x = [[ea, eb] for eb in b for ea in a] # 20 * 10
y = [e[0]%e[1] for e in x]

print("x:", x)
print("y:", y)

x = torch.tensor(x).to(device, torch.float)
y = torch.tensor(y).to(device, torch.float)

hidden_dim = 100

model = nn.Sequential(
    nn.Linear(2,hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim,hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim,1)
)
model.cuda()

criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for t in range(1000):
    y_pred = model(x).squeeze(-1)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if t % 10 != 0:
        continue
    print(t, loss.item())
    #with torch.no_grad():
        #y_pred = model(x)
        
        #for i in range(len(x)):
        #    print("{}={}, {}".format(x[i], y[i].item(), y_pred[i].item()))
        
with torch.no_grad():
    y_pred = model(x)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    y_reshaped = y.reshape(len(b), len(a)).cpu().numpy()

    im = ax1.imshow(y_reshaped)
    ax1.set_xticks(np.arange(len(a)))
    ax1.set_yticks(np.arange(len(b)))
    ax1.set_xticklabels(a)
    ax1.set_yticklabels(b)
    ax1.set_xlabel("a")
    ax1.set_ylabel("b")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(a)):
        for j in range(len(b)):
            text = ax1.text(i, j, round(y_reshaped[j,i].item(),1), ha="center", va="center", color="w")
    ax1.set_title("true label")

    y_reshaped = y_pred.reshape(len(b), len(a)).cpu().numpy()
    
    print("y_reshaped", y_reshaped)
    im = ax2.imshow(y_reshaped)
    ax2.set_xticks(np.arange(len(a)))
    ax2.set_yticks(np.arange(len(b)))
    ax2.set_xticklabels(a)
    ax2.set_yticklabels(b)
    ax2.set_xlabel("a")
    ax2.set_ylabel("b")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(a)):
        for j in range(len(b)):
            text = ax2.text(i, j, round(y_reshaped[j,i].item(),1), ha="center", va="center", color="w")
    ax2.set_title("predicted")

    #fig.set_title("epoch {}".format(t))
    fig.tight_layout()
    #plt.title("epoch {}".format(t))
    #plt.plot(x[:,0].cpu().numpy(), y_pred.cpu().numpy(), 'b.')
    
    #plt.pause(0.001)
    plt.show()



