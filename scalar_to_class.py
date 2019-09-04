import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import random

n=2
modulo = 10

x = [e*1 for e in range(10*n)]
#x = random.sample(x, 10*n)
y = [e%modulo for e in x]
#x_test = [e*0.1 for e in range(100*n)]
x_test = [e*1 for e in range(10*n*3)]
y_test = [e%modulo for e in x_test]

x = torch.tensor(x).unsqueeze(1).to(torch.float)
y = torch.tensor(y).to(torch.long)
x_test = torch.tensor(x_test).unsqueeze(1).to(torch.float)
y_test = torch.tensor(y_test).to(torch.long)
print(x.size())
print(y.size())
print(x_test.size())
print(y_test.size())


class ModuloModel(nn.Module):
    def __init__(self):
        super(ModuloModel, self).__init__()
        
        self.hidden_dim = 1
        self.gen_hidden_dim = 20

        self.expander = nn.Linear(1,self.hidden_dim)
        self.generator = nn.Sequential(
                        nn.Linear(self.hidden_dim, self.gen_hidden_dim),
                        nn.Tanh(),
                        nn.Linear(self.gen_hidden_dim,10))
    def forward(self, x):
        
        expan = self.expander(x)
        print(expan)
        mod = torch.fmod(expan, modulo)
        word = self.generator(mod)
        return word

hidden_dim = 10
simple = nn.Sequential(
    #nn.Linear(1,10)
    nn.Linear(1,hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim,10),
)
#model = simple
model = ModuloModel()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for t in range(10000):
    # indices = random.sample(range(10*n), 10*n)
    # train_loss = 0
    # for i in indices:
    #     y_score = model(x[i].unsqueeze(-1))
    #     loss = criterion(y_score, y[i].unsqueeze(-1))
    #     train_loss += loss
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    
    y_score = model(x)
    train_loss = criterion(y_score, y)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if t % 10 != 0:
        continue
    with torch.no_grad():
        score = model(x_test)
        loss = criterion(score, y_test)
        pred = torch.argmax(score, dim=1)

        correct = torch.eq(pred, y_test).sum().to(torch.float)
        total = len(y_test)
        acc = correct/total
        print("epoch {}, trainloss {:.3f} testloss {:.3f}, acc {:.3f}".format(t, train_loss.item(), loss.item(), acc))
        #print("epoch", t, "loss", loss.item())
        
        #for i in range(len(x_test)):
        #    print("{:.1f} , {}".format(x_test[i].item(), pred[i].item()))
        #print(x_test.size(), pred.size())
        plt.clf()
        #plt.title("epoch {}".format(t))
        plt.plot(x_test.numpy(), pred.numpy())
        plt.plot(x.tolist(), y.tolist(),  'ro')
        plt.pause(0.001)
        #plt.show()


