import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from math_dataset import MyDataset

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
args = parser.parse_args()
PATH = "guess_game.pt"

class SingleLayer(nn.Module):
    def __init__(self, D_in, D_out):
        super(SingleLayer, self).__init__()
        self.linear1 = nn.Linear(D_in,D_out)
    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class TwoLayer(nn.Module):
    def __init__(self, D_in, H, D_out, drop=0.1):
        super(TwoLayer, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(p=drop)
    def forward(self, x):
        hidden = F.tanh(self.linear1(x))
        drop = self.dropout(hidden)
        y_pred = self.linear2(drop)
        return y_pred


def train():
    #data size
    _i, _j, _k = 3,3,3
    #batch, input, hidden, output
    N, D_in, H, D_out = 1, _i+_j+_k, _i*_j*_k, _i*_j*_k
    signal_size = 40
    
    dtype = torch.float
    device = torch.device("cpu")
    #device = torch.device("cuda:0")

    dataset = MyDataset(_i,_j,_k)
    x, y = dataset.get_frame()
    x = torch.tensor(x, dtype=dtype, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device).squeeze()
    print(x.size(), y.size())

    #sender = SingleLayer(D_in, signal_size)
    #sender = TwoLayer(D_in, H, signal_size)
    sender = TwoLayer(D_in, H, signal_size, drop=0.4).to(device)
    #receiver = SingleLayer(signal_size, D_out)
    #receiver = TwoLayer(signal_size, H, D_out)
    receiver = TwoLayer(signal_size, H, D_out, drop=0.4).to(device)

    optimizer = optim.Adam([{'params': sender.parameters()}, \
                            {'params': receiver.parameters()}], \
                             lr = 1e-2)
    epoch = 0
    if args.resume:
        checkpoint = torch.load(PATH)
        sender.load_state_dict(checkpoint['sender_state_dict'])
        receiver.load_state_dict(checkpoint['receiver_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    for episode in range(epoch, epoch + 100001):
        signal_score = sender(x)
        signal_probs = F.softmax(signal_score, dim=1)
        distr_signal = Categorical(signal_probs)
        signal = distr_signal.sample()
        signal_vector = torch.eye(signal_size)[signal]
        signal_vector = signal_vector.to(device)
        
        action_score = receiver(signal_vector)
        action_probs = F.softmax(action_score, dim=1)
        distr_action = Categorical(action_probs)
        action = distr_action.sample()

        reward = torch.eq(action, y).to(torch.float)
        reward = (reward - reward.mean())
        
        sender_loss = -distr_signal.log_prob(signal) * reward
        receiver_loss = -distr_action.log_prob(action) * reward

        sender.zero_grad()
        receiver.zero_grad()

        sender_loss.sum().backward()
        receiver_loss.sum().backward()
        
        optimizer.step()

        if episode % 1000 == 0:
            with torch.no_grad():
                sender.eval()
                receiver.eval()
                score_signal = sender(x)
                signal = torch.argmax(score_signal, dim=1)
                signal_vector = torch.eye(signal_size)[signal]
                score_action = receiver(signal_vector)
                action = torch.argmax(score_action, dim=1)
                eq = torch.eq(action, y)

                print("t: {}, acc: {}/{} = {}".format(episode, torch.sum(eq).item(), eq.numel(), torch.sum(eq).item() / eq.numel()))
                for i in range(y.numel()):
                    print("({:2d},{:2d}),'{:2d}'".format(y[i].item(), action[i].item(), signal[i].item()))
                sender.train()
                receiver.train()
            
            torch.save({'epoch': episode,
                        'sender_state_dict': sender.state_dict(),
                        'receiver_state_dict': receiver.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)

if __name__ == "__main__":
    train()
