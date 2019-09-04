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
from communication_dataset import Num2NumDataset
from communication_dataset import *
from analysis import draw

from nac import NAC
from nalu import NALU
from mlp import MLP

parser = argparse.ArgumentParser()

parser.add_argument('--load', type=str)
parser.add_argument('--save', type=str)

parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
parser.add_argument('--load_saved', action="store_true", help='evaluate saved model')

# DATA
parser.add_argument('--soft', action="store_true", help='use soft split')
parser.add_argument('--range', type=int, default=1000, help='training range')

# LANGUAGE:
parser.add_argument('--seq', type=int, default=0, help='sequence length')
parser.add_argument('--base', type=int, default=10, help='numeral system base number')

parser.add_argument('--model', type=str)

# MODEL PARAMETERS:
parser.add_argument('--hidden', type=int, default=512, help='hidden size')
parser.add_argument('--reverse', action="store_true", help='reverse receiver rnn')

# TRAINING:
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=100000, help='epoch up to')
parser.add_argument('--patience', type=int, default=100000, help='patience')
parser.add_argument('--batch', type=int, default=100, help='batch size')


args = parser.parse_args()
options = vars(args)
for o in options:
    print(o, options[o])

#logging.basicConfig(filename='log.log', level=logging.DEBUG)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
device = torch.device("cuda:0")

VOCAB = ['0','1','2','3','4','5','6','7','8','9','SOS','EOS','PAD']
SOS = VOCAB.index('SOS')
EOS = VOCAB.index('EOS')
PAD = VOCAB.index('PAD')

def run_model(model, dataset, details):
    inputs = []
    outputs = []
    targets = []
    hiddens = []

    dataloader = DataLoader(dataset, batch_size = 100)

    with torch.no_grad():
        model.eval()
        for i_batch, sample_batched in enumerate(dataloader):
            vocab_outputs = model(sample_batched['inputs'].to(device), 
                                  sample_batched['teacher_onehots'].to(device),
                                  sample_batched['teacher_digits'].to(device),
                                  teacher=False)
            vocab_outputs = torch.stack(vocab_outputs, dim=1) #result: (batch size * seqlen * vocabdim)

            inputs.append(sample_batched['inputs'].to(device))
            outputs.append(vocab_outputs)
            targets.append(sample_batched['digits'].to(device))
            h = torch.cat(model.hiddens, dim=1)
            hiddens.append(h)

        inputs = torch.cat(inputs, dim=0)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        hiddens = torch.cat(hiddens, dim=0)
    if details:
        torch.set_printoptions(precision= 1)
        for i in range(len(inputs)):
            print("{}, out:{}, t:{}, {}".format(inputs[i].item(), outputs[i].argmax(dim=1).tolist(), 
                                            targets[i].tolist(), 
                                            [round(e,1) for e in hiddens[i].tolist()]
                                            ))
    return inputs, outputs, targets

def eval_model(model, dataset, criterion):
    inputs , outputs, targets = run_model(model, dataset, False)

    output_digits = torch.argmax(outputs, dim=2)

    #for i in range(len(output_digits)):
    #    print(targets[i].tolist(), output_digits[i].tolist())

    total = 0
    correct = 0
    interpreted_list = []
    for i in range(targets.size()[0]):
        interpreted_num = 0
        stopped = False
        for j in range(targets.size()[1]):
            t = targets[i,j]
            o = output_digits[i,j]
            if t != PAD:
                if output_digits[i,j] == t :
                    correct += 1
                total += 1
            if not stopped:
                if o == EOS:
                    stopped = True
                elif o == PAD:
                    stopped = True
                elif o == SOS:
                    stopped = True
                else:
                    interpreted_num += o * pow(10,j)
        interpreted_list.append(interpreted_num)
    
    
    digitAcc = correct / total

    inputs = inputs.squeeze(-1)
    num_correct = 0
    num_total = len(inputs)
    for i in range(len(inputs)):
        #print(inputs[i].item(), interpreted_list[i])
        if int(inputs[i]) == int(interpreted_list[i]):
            num_correct += 1
    numberAcc = num_correct / num_total
    
    
    with torch.no_grad():
        interpreted_list = torch.tensor(interpreted_list).to(device, torch.float)

        numMSE = nn.MSELoss()
        numericMSE = numMSE(interpreted_list, inputs).item()

        numAbs = nn.L1Loss()
        numericAbs = numAbs(interpreted_list, inputs).item()

        aligned = targets.size()[0]*targets.size()[1]
    
        outputs_flat = outputs.reshape(aligned, -1)
        targets_flat = torch.flatten(targets)
    
        
        criterion = nn.CrossEntropyLoss(ignore_index=args.base+2)
        avgCE = nn.CrossEntropyLoss(reduction='mean', ignore_index=args.base+2)
        sumCE = nn.CrossEntropyLoss(reduction='sum', ignore_index=args.base+2)

    #fieldnames = ['testRange', 'averageCELoss', 'sumCELoss', 'digit-wise accuracy', 'number-wise accuracy', 'numerical MSELoss', 'numerical ABSLoss']
    results = {}
    results['avgCELoss'] = avgCE(outputs_flat, targets_flat).item()
    results['sumCELoss'] = sumCE(outputs_flat, targets_flat).item()
    results['digitAcc'] = digitAcc
    results['numberAcc'] = numberAcc
    results['numericMSE'] = numericMSE
    results['numericAbs'] = numericAbs


    return results


#def train_model(model, train_dataloader, valid_dataset_inter, valid_dataset_extra, criterion, opt, max_patience):
# validation dataset contains whatever we have combined to create validation dataset. we don't need to separate them out.
def train_model(model, train_dataset, valid_dataset, 
                criterion, opt, lr, max_patience, max_epoch, batch_size, save_path):
    
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

    patience = max_patience

    min_loss = float('inf')
    best_epoch = 0
    best_model_state_dict = model.state_dict()
    optimizer = opt(model.parameters(), lr=lr)

    for epoch in range(1, max_epoch + 1):        
        model.train()
        for i_batch, sample_batched in enumerate(train_dataloader):

            vocab_outputs = model(sample_batched['inputs'].to(device), 
                                  sample_batched['teacher_onehots'].to(device),
                                  sample_batched['teacher_digits'].to(device),
                                  teacher=True)
            vocab_outputs = torch.stack(vocab_outputs, dim=1) #result: (batch size * seqlen * vocabdim)
            vocab_outputs = torch.transpose(vocab_outputs, 1,2) #result: (N * C * d1)

            #loss = criterion(vocab_outputs, sample_batched['digits'].to(device, torch.long))
            loss = criterion(vocab_outputs, sample_batched['digits'].to(device, torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        # if epoch%10!=0:
        #     continue
        inputs, outputs, targets = run_model(model, train_dataset, True)
        outputs = torch.transpose(outputs, 1,2)
        train_loss = criterion(outputs, targets)

        inputs, outputs, targets = run_model(model, valid_dataset, False)
        outputs = torch.transpose(outputs, 1,2)
        valid_loss = criterion(outputs, targets)


    #train_loss = eval_model(model, train_dataset, criterion)['avgCELoss']
    #valid_loss = eval_model(model, valid_dataset, criterion)['avgCELoss']
        print('epoch {}, train_loss {:.3f}, valid_loss {:.3f}, min_loss {:.3f}, patience:{}'.format(epoch, train_loss, valid_loss, min_loss, patience))

    #if min loss achieved update best record
        if valid_loss < min_loss:
            min_loss = valid_loss
            best_epoch = epoch
            best_model_state_dict = model.state_dict()
            patience = max_patience
        else:
            patience -= 1

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_model_state_dict': best_model_state_dict,
                    'best_epoch': best_epoch,
                    }, save_path)

        if patience == 0:
            break
    print("training ended and saved at", save_path)



def numeral_generation():
    #MAX_LEN = int(math.log(args.data*10, args.base) + 2)
    #if args.seq != 0:
    #    MAX_LEN = args.seq

    assert args.save or args.load, "either save or load"
    
    if args.load:
        LOAD_PATH = "saved/N2L_" + args.load + ".sav"
        SAVE_NAME = args.load
    if args.save:
        SAVE_PATH = "saved/N2L_" + args.save + ".sav"
        SAVE_NAME = args.save
    
    MAX_LEN = 6
    if args.base ==  2:
        MAX_LEN = 18
    BASE = args.base
    HIDDEN_DIM = 100

    BATCH_SIZE = args.batch
    LR = args.lr if args.lr else 0.001
    PATIENCE = 1000
    MAX_EPOCH = 10000
    RANGE = args.range
    SOFT = args.soft
    train_list, valid_list, test_list = dataset_split(10, soft=True)
    #train_list, valid_list, test_list = dataset_split(RANGE, soft=SOFT)
    print(len(train_list), len(valid_list), len(test_list))
    train_dataset = Num2NumDataset(train_list, BASE, MAX_LEN, reverse=True)
    valid_dataset = Num2NumDataset(valid_list, BASE, MAX_LEN, reverse=True)
    test_dataset = Num2NumDataset(test_list, BASE, MAX_LEN, reverse=True)

    #models don't need to care about base. they only care about vocab size

    if args.model == 'lstm':
        model = NumToLang_LSTM(hidden_dim=HIDDEN_DIM, vocab_size=len(VOCAB), max_len=MAX_LEN)
        print("fuck")
    if args.model == 'modulus':
        model = NumToLang_Modulus(hidden_dim=HIDDEN_DIM, vocab_size=len(VOCAB), max_len=MAX_LEN, modulus = BASE)
    if args.model == 'modulus2':
        model = NumToLang_NALU(hidden_dim=HIDDEN_DIM, vocab_size=len(VOCAB), max_len=MAX_LEN)
    if args.model == 'extra':
        model = NumToLang_Extra(hidden_dim=HIDDEN_DIM, vocab_size=len(VOCAB), max_len=MAX_LEN)
    model.cuda()

    if args.load:
        checkpoint = torch.load(LOAD_PATH)
        model.load_state_dict(checkpoint['best_model_state_dict'])
        best_epoch = checkpoint['best_epoch']
        last_epoch = checkpoint['epoch'] 
        print("model_name", LOAD_PATH)
        print("best_epoch", best_epoch)
        print("epoch", last_epoch)      


    #criterion = nn.CrossEntropyLoss(ignore_index=args.base+1)
    criterion = nn.NLLLoss(ignore_index=PAD)
    #criterion = nn.NLLLoss()
    optimizer = optim.Adam

    if args.save:
        train_model(model, train_dataset, valid_dataset, 
                    criterion = criterion, 
                    opt = optimizer,
                    lr = LR,
                    max_patience = PATIENCE,
                    max_epoch = MAX_EPOCH,
                    batch_size = BATCH_SIZE,
                    save_path = SAVE_PATH)

    # test outputs
    inputs, outputs, targets = run_model(model, test_dataset, False)
    
    # result analysis part
    test_ranges = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    test_datasets = {}
    for r in test_ranges:
        test_datasets[r] = Num2NumDataset([e for e in range(r)], BASE, MAX_LEN, reverse=True)

    results = {}
    for test_range in test_datasets:
        dataset = test_datasets[test_range]
        results[test_range] = eval_model(model, dataset, criterion)
        results[test_range]['testRange'] = test_range
        fieldnames = results[test_range].keys()

        #each testrange result contains: testRange, average CELoss, sum CELoss, digit-wise accuracy, number-wise accuracy, numerical MSELoss, numerical loss
    
    with open('saved/N2L_' + SAVE_NAME + '_results.csv', 'w', newline='') as csvfile:
        #fieldnames = ['testRange', 'averageCELoss', 'sumCELoss', 'digit-wise accuracy', 'number-wise accuracy', 'numerical MSELoss', 'numerical ABSLoss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for test_range in results:
            writer.writerow(results[test_range])

        print(fieldnames)
        for test_range in results:
            print(results[test_range])


    # generation part
    generation_dataset = Num2NumDataset([e for e in range(10000)], BASE, MAX_LEN, reverse=True)    
    inputs, outputs, targets = run_model(model, generation_dataset, False)
    output_digits = torch.argmax(outputs, dim=2)

    with open('saved/N2L_' + SAVE_NAME + '_generated.csv', 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        for i in range(len(inputs)):
            row = [inputs[i].item()] + targets[i].tolist() + output_digits[i].tolist()
            writer.writerow(row)
    #print("evaluation loss: {:.3f}, {:.3f}".format(loss_inter.item(), loss_extra.item()))

if __name__ == '__main__':
    numeral_generation()

