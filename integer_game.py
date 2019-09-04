import argparse
import random
import time
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


random.seed(0)
torch.manual_seed(0)
dtype = torch.float
device = torch.device("cpu")

#task definition:
#input: 1 integer
#prediction: value of integer (regression task)


class Sender_Receiver(nn.Module):
    def __init__(self):
        super(Sender_Receiver, self).__init__()
        self.sender_hidden_size = 100
        self.vocab_size = 20
        self.max_length = 2
        self.receiver_hidden_size = 100
        self.tau = 5

        self.linear1 = nn.Linear(1, self.sender_hidden_size)
        self.sender_grucell = nn.GRUCell(self.vocab_size, self.sender_hidden_size)
        
        self.linear2 = nn.Linear(self.sender_hidden_size, self.vocab_size)

        self.receiver_grucell = nn.GRUCell(self.vocab_size, self.receiver_hidden_size)
        self.linear3 = nn.Linear(self.receiver_hidden_size, 1)
        
        self.eval_mode = False
    def set_train(self):
        self.train()
        self.eval_mode = False
    def set_eval(self):
        self.eval()
        self.eval_mode = True

    def forward(self, x):
        #x size: batch_size * 1
        sender_hidden = self.linear1(x)
        receiver_hidden = self.init_hidden()
        symbol = self.init_symbol()
        symbol_sequence = []
        for i in range(self.max_length):
            #sender_side
            sender_hidden = self.sender_grucell(symbol, sender_hidden)
            logprob = F.log_softmax(self.linear2(sender_hidden), dim=1)
            if self.eval_mode:
                symbol = torch.eye(logprob.size()[1])[torch.argmax(logprob, dim=1)]
            else:
                symbol = F.gumbel_softmax(logprob, hard = True, tau = self.tau)
            #record symbols
            symbol_sequence.append(torch.argmax(symbol).item())
            #receiver side
            receiver_hidden = self.receiver_grucell(symbol, receiver_hidden)
        prediction = self.linear3(receiver_hidden)
        return prediction, symbol_sequence
            
    def init_symbol(self):
        #symbol shape(batch, input_size)
        return torch.zeros(1, self.vocab_size)
    def init_hidden(self):
        return torch.zeros(1, self.receiver_hidden_size)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

model = Sender_Receiver()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

n_iters = 10000
print_every = 50
plot_every = 50
test_every = 50
all_losses = []
total_loss = 0
start = time.time()

for iter in range(1, n_iters + 1):
    batch_loss = 0
    for batch in range(100):
        #r = random.random()*10
        r = random.paretovariate(1)
        x = torch.tensor([r]).unsqueeze(-1)
        y = torch.tensor([r]).unsqueeze(-1)
        prediction, symbol_sequence = model(x)
        loss = criterion(prediction, y)
        batch_loss += loss
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(loss)
    
    if iter % test_every == 0:
        model.set_eval()
        tests = [float(i) for i in range(10)]
        test_x = [torch.tensor([[t]]) for t in tests]
        test_y = test_x
        for x in test_x:
            prediction, symbol_sequence = model(x)
            print("{:.2f}, {:.2f}, {}".format(x.item(), prediction.item(), symbol_sequence))

        
        model.set_train()

