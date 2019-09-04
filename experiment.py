import argparse
import csv
import logging
import numpy as np
from time import strftime, localtime
from collections import Counter
import math
import random

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.distributions import Categorical

from agents import Sender_Receiver
from agents import *
from analysis import draw

from nac import NAC
from nalu import NALU
from mlp import MLP

parser = argparse.ArgumentParser()

parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
parser.add_argument('--cuda', action="store_true", help='use GPUs')
parser.add_argument('--draw_saved', action="store_true", help='only draw learning history')

# DATA
parser.add_argument('-i', '--input', choices=['discrete', 'continuous', 'combined'])
parser.add_argument('--data', choices=['contiguous', 'power', 'uniform'], help='data distribution')
parser.add_argument('--max', type=int, default=10, help='from 0 to max number')
parser.add_argument('--num', type=int, default=10, help='number of numbers to learn')
parser.add_argument('--split', choices=['random', 'end'])

# LOSS
parser.add_argument('--regression', type=float, default=0.1, help='regression loss weight')
parser.add_argument('--classification', type=float, default=0.01, help='cross entropy loss weight')
parser.add_argument('--length_reward', type=float, default=1, help='length reward(loss) multiplier')
        # rich-get-richer vocabulary
        # apply higher weight for small numbers? (need to be formalized)

# LANGUAGE:
parser.add_argument('--vocab', type=int, default=2, help='vocab size')
parser.add_argument('--seq', type=int, default=4, help='sequence length')

# MODEL CLASS:
parser.add_argument('--model', choices=['single', 'two', 'SR'])

# MODEL PARAMETERS:
parser.add_argument('--tau', type=float, default=5.0, help='gumbel temperature')
parser.add_argument('--hidden', type=int, default=512, help='hidden size')
parser.add_argument('--rnn', choices=['rnn', 'gru', 'lstm', 'nac', 'nalu'])
parser.add_argument('--reverse', action="store_true", help='reverse receiver rnn')

# TRAINING:
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=10000, help='epoch up to')
        # curriculum?

# ANALYSIS:
parser.add_argument('--visualize', action='store_true', help='real-time loss visualization')

###############
# for each experiment save:
#     savefile name: args
#     loss history?
#     model parameters, epoch, optimizer (this becomes obsolete if model is changed... so only for resuming)
#     a file that shows number-vocab mappings + value prediction

# a program to show the results by loading saved files    
#     show loss history graph
#     show numbers -> vocabulary results

#     


args = parser.parse_args()
options = vars(args)
for o in options:
    print(o, options[o])
PATH = "results/tasked_game.pt"

logging.basicConfig(filename='log.log', level=logging.DEBUG)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
device = torch.device("cuda:0")

class MyDataset(Dataset):
    def __init__(self, x_list, y_list):
        #we are given x_list in python list format, and create x, y pairs
        print(x_list)
        self.x = torch.tensor(x_list).to(torch.long).to(torch.float)
        self.y = torch.tensor(y_list).to(torch.long).to(torch.float)        
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        #should return numpy array
        return {'x': self.x[idx], 'y': self.y[idx]}


def make_data_list_identity():
    x_train = [e for e in range(100)]

    x_valid_inter = random.sample(x_train, 10)
    for e in x_valid_inter:
        x_train.remove(e)
    x_valid_extra = random.sample(range(100,1000), 10)

    x_test_inter = random.sample(x_train, 10)
    for e in x_test_inter:
        x_train.remove(e)
    x_test_extra = random.sample(range(100,1000), 10)

    inputs = []
    targets = []
    for x_set in [x_train, x_valid_inter, x_valid_extra, x_test_inter, x_test_extra]:
        inputs.append([[e] for e in x_set])
    for x_set in inputs:
        targets.append([e[0] for e in x_set])
    
    return inputs, targets


def make_data_list_binaryop(op):
    unary = False
    if op in ['ide', 'pow', 'sqrt']:
        unary = True
    
    n = 1000
    x_train = [e for e in range(n)]
    #x_train = random.sample(x_train, 1000)
    x_valid_inter = random.sample(x_train, n//10)
    for e in x_valid_inter:
        x_train.remove(e)
    x_valid_extra = random.sample(range(n, n*10), n//10)

    x_test_inter = random.sample(x_train, n//10)
    for e in x_test_inter:
        x_train.remove(e)
    x_test_extra = random.sample(range(n, n*10), n//10)

    inputs = []
    targets = []

    
    for x_list in [x_train, x_valid_inter, x_valid_extra, x_test_inter, x_test_extra]:
        if unary:
            inputs.append([[e] for e in x_list])
        else:
            inputs.append([[e//10, e%10 + 1] for e in x_list])


    for x_list in inputs:
        if op == 'add':
            targets.append([e[0]+e[1] for e in x_list])
        elif op == 'sub':
            targets.append([e[0]-e[1] for e in x_list])
        elif op == 'mul':
            targets.append([e[0]*e[1] for e in x_list])
        elif op == 'div':
            targets.append([e[0]/e[1] for e in x_list])
        elif op == 'mod':
            targets.append([e[0]%e[1] for e in x_list])
        elif op == 'ide':
            targets.append([e[0] for e in x_list])
        elif op == 'pow':
            targets.append([math.pow(e[0],2) for e in x_list])
        elif op == 'sqrt':
            targets.append([math.sqrt(e[0]) for e in x_list])
    return inputs, targets

def make_dataset(op):
    #inputs, targets = make_data_list_identity()
    inputs, targets = make_data_list_binaryop(op)

    dataset_train = MyDataset(inputs[0], targets[0])
    dataset_valid_inter = MyDataset(inputs[1], targets[1])
    dataset_valid_extra = MyDataset(inputs[2], targets[2])
    dataset_test_inter = MyDataset(inputs[3], targets[3])
    dataset_test_extra = MyDataset(inputs[4], targets[4])
    return dataset_train, dataset_valid_inter, dataset_valid_extra, dataset_test_inter, dataset_test_extra



def eval_model(model, dataset, criterion):
    inputs = []
    preds = []
    targets = []
    dataloader = DataLoader(dataset, batch_size = 100)
    vocabs = []
    vocab_exists = False

    with torch.no_grad():
        model.eval()    
        for i_batch, sample_batched in enumerate(dataloader):
            scalar_output = model(sample_batched['x'].to(device, torch.float))
            inputs.append(sample_batched['x'].to(device, torch.float))
            preds.append(scalar_output.squeeze())
            targets.append(sample_batched['y'].to(device, torch.float))
            if hasattr(model, 'words'):
                vocab_list = torch.stack(model.words) #(seqlen, batchsize)
                vocabs.append(torch.transpose(vocab_list, 0,1))
                vocab_exists = True
        inputs = torch.cat(inputs, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        if vocab_exists:
            vocabs = torch.cat(vocabs)
    
    valid_loss = criterion(preds, targets)
    for i in range(len(preds)):
        if len(vocabs) ==0:
            print("{}, {:.1f} ,{:.1f}".format(inputs[i].tolist(), targets[i].item(), preds[i].item()))
        else:
            print("{}, {:.1f} ,{:.1f}, {}".format(inputs[i].tolist(), targets[i].item(), preds[i].item(), vocabs[i].tolist()))
    return valid_loss

def train_model(model, train_dataloader, valid_dataset_inter, valid_dataset_extra, criterion, opt, patience):

    batch_size = 100
    #train_dataloader = DataLoader(train_dataset, batch_size = batch_size)
    
    checkpoint = 'best_model.sav'
    running_patience = patience
    running_batch = 0
    running_loss = 0
    min_loss = float('inf')
    optimizer = opt(model.parameters(), lr=args.lr)
    num_epochs = args.epoch
    every = 10
    for epoch in range(1, num_epochs + 1):
        #training
        
        model.train()
        for i_batch, sample_batched in enumerate(train_dataloader):
            scalar_output = model(sample_batched['x'].to(device, torch.float))
            loss = criterion(scalar_output.squeeze(), sample_batched['y'].to(device, torch.float))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_batch += 1
            if epoch % every == 0:
                print('epoch {},  train loss {:.3f}  '.format(epoch, loss.item()))


        # validation
        if epoch % every != 0:
            continue
        valid_loss_inter = eval_model(model, valid_dataset_inter, criterion)
        valid_loss_extra = eval_model(model, valid_dataset_extra, criterion)
        print('valid_loss inter: {:.3f}, extra: {:.3f}'.format(valid_loss_inter.item(), valid_loss_extra.item()))
        valid_loss = valid_loss_inter + valid_loss_extra
        #early stopping
        if valid_loss < min_loss:
            min_loss = valid_loss
            running_patience = patience
            with open(checkpoint, 'wb') as f:
                torch.save(model.state_dict(), f)
        else:
            running_patience -= 1
        if running_patience == 0:
            break
    
    model.load_state_dict(torch.load(checkpoint))


def reconstruction_experiment():

    op = 'div'
    train_dataset, valid_dataset_inter, valid_dataset_extra, test_dataset_inter, test_dataset_extra = make_dataset(op) #torch.Dataset class
    train_dataloader = DataLoader(train_dataset, batch_size = 100)

    IN_DIM=2
    if op in ['ide', 'pow', 'sqrt']:
        IN_DIM = 1
    NUM_LAYERS = 2
    HIDDEN_DIM = 2
    models = [
            
            #MLP(num_layers=1,in_dim=IN_DIM,hidden_dim=1,out_dim=1,activation='none',),
            # NAC(num_layers=1,in_dim=IN_DIM,hidden_dim=HIDDEN_DIM,out_dim=1,),
            NALU(num_layers=1,in_dim=IN_DIM,hidden_dim=HIDDEN_DIM,out_dim=1),

            # MLP(num_layers=2,in_dim=IN_DIM,hidden_dim=HIDDEN_DIM,out_dim=1,activation='relu6',),
            # MLP(num_layers=2,in_dim=IN_DIM,hidden_dim=HIDDEN_DIM,out_dim=1,activation='none',),
            # NAC(num_layers=2,in_dim=IN_DIM,hidden_dim=HIDDEN_DIM,out_dim=1,),
            #NALU(num_layers=2,in_dim=IN_DIM,hidden_dim=HIDDEN_DIM,out_dim=1),
            # Sender_Receiver(rnn='rnn').cuda(),
            # Sender_Receiver(rnn='gru').cuda(),
            # Sender_Receiver(rnn='nac').cuda(),
            # Sender_Receiver(rnn='nalu').cuda() ,
            # Gumbel_Agent().cuda(),
            #Multi_Gumbel_Agent(in_dim=IN_DIM).cuda()
            #Mod_Agent(in_dim=IN_DIM, order=20).cuda()
            #Numeral_Machine().cuda()
              ]
    for m in models:
        m.cuda()
    criterion = nn.MSELoss()
    
    optimizer = optim.RMSprop
    #optimizer = optim.Adam

    for model in models:
        #print("model rnn name: ", model.rnn)
        
        train_model(model, train_dataloader, valid_dataset_inter, valid_dataset_extra, criterion, optimizer, patience=500)
        loss_inter = eval_model(model, test_dataset_inter, criterion)
        loss_extra = eval_model(model, test_dataset_extra, criterion)
        #print("test loss", loss.item())
        f=open("results.txt", 'a')
        f.write("\n{:.3f}, {:.3f}".format(loss_inter.item(), loss_extra.item()))
        f.close()
        print("evaluation loss: {:.3f}, {:.3f}".format(loss_inter.item(), loss_extra.item()))
        #write everything to result section? #assuming that process wasn't interrupted... #well we will do per-model observation anyways and do the controlled experiment as last step
    #record_results()

def wave(n = 1):
    #xs = np.linspace(0, 50, 200)
    xs = range(1000)
    wavs = []
    mods = []
    for L in range (100,101):
        for x in xs:
            mod = smooth_mod(x+0.5, L, n)
            print("{:.1f} mod {} = {:.1f} ({:.1f})".format(x, L, x%L, mod))
            #Ls.append(L)
            wavs.append(mod)
            mods.append(x%L)
    
    plt.plot(xs, wavs)
    plt.plot(xs, mods)
    plt.show()

if __name__ == '__main__':

    reconstruction_experiment()

