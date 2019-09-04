import math
import random
import argparse


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

from communication_dataset import dataset_split
from communication_dataset import *
from agents import *


parser = argparse.ArgumentParser()
parser.add_argument('--reverse', action="store_true", help="reverse digit")
parser.add_argument('--name', type=str)
parser.add_argument('--train', action="store_true")
#parser.add_argument('--load_saved', action="store_true", help="load saved and test only")

args = parser.parse_args()

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = torch.device("cuda:0")

VOCAB = ['0','1','2','3','4','5','6','7','8','9','SOS','EOS','PAD']


# class Num2NumDataset(Dataset):
#     def __init__(self, data_array, base, max_len, reverse=False, english=False):
#         BASE = base
#         MAX_LEN = max_len
#         VOCAB_SIZE = BASE+2

#         #data_array accepts 1-D number list
#         inputs = [[e] for e in data_array]
#         outputs = [e for e in data_array]

#         #words assuming base-10 language
#         if english:
#             digits = [self.get_digits_english(e[0], BASE, MAX_LEN, reverse)[0] for e in inputs]
#             lengths = [self.get_digits_english(e[0], BASE, MAX_LEN, reverse)[1] for e in inputs]
#         else:
#             digits = [self.get_digits(e[0], BASE, MAX_LEN, reverse)[0] for e in inputs]
#             lengths = [self.get_digits(e[0], BASE, MAX_LEN, reverse)[1] for e in inputs]
#         onehots = [[np.eye(VOCAB_SIZE)[d] for d in e] for e in digits]

#         self.inputs = torch.tensor(inputs).to(torch.float)
#         self.outputs = torch.tensor(outputs).to(torch.float)
#         self.onehots = torch.tensor(onehots).to(torch.long).to(torch.float)
#         self.lengths = lengths
#         self.digits = torch.tensor(digits).to(torch.long)
    
#     def __len__(self):
#         return len(self.inputs)
#     def __getitem__(self, idx):
#         return {'inputs': self.inputs[idx], 'targets':self.outputs[idx], 'onehots':self.onehots[idx], 'digits': self.digits[idx], 'lengths': self.lengths[idx]}

#     def get_digits(self, n, base, max_len, reverse):
#         digit_list = []
#         while n > 0:
#             digit_list.insert(0, n%base)
#             n = n // base
#         if len(digit_list) == 0:
#             digit_list.append(0)
#         if reverse:
#             digit_list.reverse()

#         #record sequence lengths
#         length = len(digit_list)
        
#         #add EOS token (=base)
#         if length < max_len:
#             digit_list.append(base)

#         #add padding tokens (=base+1)
#         while len(digit_list) < max_len:
#             #digit_list.append(base+1)
#             digit_list.insert(0, base+1)
#         return digit_list, length
    
#     def get_digits_english(self, n, base, max_len, reverse):
#         n_init = n
#         digit_list = []
#         tens = ['error',12,13]
#         tens_idx = 0
#         while n > 0:
#             last_digit = n%base
#             if last_digit == 0:
#                 pass
#             else:
#                 digit_list.insert(0, last_digit)
#                 if tens_idx != 0:
#                     digit_list.insert(1, tens[tens_idx])
#             n = n // base
#             tens_idx += 1
#         if len(digit_list) == 0:
#             digit_list.append(0)

#         if reverse:
#             digit_list.reverse()

#         #record sequence lengths
#         length = len(digit_list)
        
#         #add EOS token (=base)
#         if length < max_len:
#             digit_list.append(base)

#         #add padding tokens (=base+1)
#         while len(digit_list) < max_len:
#             digit_list.append(base+1)

#         return digit_list, length

def run_model(model, dataset, details=False):
    dataloader = DataLoader(dataset, batch_size = 200)
    model.eval()
    with torch.no_grad():
        outputs = []
        targets = []
        digits = []
        for i_batch, sample_batched in enumerate(dataloader):
            receiver_outputs = model(sample_batched['onehots'].to(device))
            receiver_outputs = receiver_outputs.squeeze()

            outputs.append(receiver_outputs)
            targets.append(sample_batched['targets'].to(device, torch.float))
            digits.append(sample_batched['digits'].to(device))
        
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        digits = torch.cat(digits, dim=0)

    for i in range(len(outputs)):
        print("{:.3f}, {:.3f}, {}".format(targets[i].item(), outputs[i].item(), digits[i].tolist()))
    return outputs, targets, digits


def train_model(receiver_model, train_dataset, valid_dataset,
                criterion, opt, lr, max_patience, max_epoch, batch_size, model_path):
    
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

    optimizer = opt([{'params': receiver_model.parameters()}], lr=lr) 

    
    patience = max_patience
    min_loss = float('inf')

    for epoch in range(1, max_epoch + 1):
        receiver_model.train()
        for i_batch, sample_batched in enumerate(train_dataloader):
            #print(sample_batched['onehots'], sample_batched['onehots'].size())
            receiver_outputs = receiver_model(sample_batched['onehots'].to(device))
            receiver_outputs = receiver_outputs.squeeze()
            #print(receiver_outputs.size())
            #print(sample_batched['targets'].size())
            
            loss = criterion(receiver_outputs, sample_batched['targets'].to(device, torch.float))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #validation
        train_outputs, train_targets, _ = run_model(receiver_model, train_dataset)

        train_loss = criterion(train_outputs, train_targets).item()
        #print("diff {} meandiff {}".format(diff, meandiff))
        #for i in range(len(train_inputs)):
        #    print("input {}, output {:.2f}".format(train_inputs[i].item(), train_outputs[i].item()))

        valid_outputs, valid_targets, _ = run_model(receiver_model, valid_dataset)
        
        # for i in range(len(valid_inputs)):
        #    print("input {}, output {:.2f}".format(valid_inputs[i].item(), valid_outputs[i].item()))

        #valid_loss = mseloss(inputs, outputs)
        valid_loss = criterion(valid_outputs, valid_targets).item()

        if valid_loss < min_loss:
            min_loss = valid_loss
            patience = max_patience
            torch.save({'receiver_state_dict': receiver_model.state_dict()
                }, model_path)
        
        else:
            patience -= 1 

        print("epoch {}, train_loss {:.3f}, valid_loss {:.3f}, patience {}".format(epoch, train_loss, valid_loss, patience))
        
        if patience == 0:
            break


#model train policy: either 1) load pretrained sender 2) nothing sender 3) pretrained & fixed sender
#sender model choice: load by name
#receiver model choice: choose from list


def numeral_communication():

    MAX_LEN = 6 #up to 99,999e , which is range(100,000)
    BASE = 10
    REVERSE = True
    TRAIN_RANGE=200
    VALID_RANGE=300
    LR = 0.001
    PATIENCE = 5000
    MAX_EPOCH = 10000
    BATCH_SIZE = 5

    HIDDEN_DIM = 100
    LOAD_PRETRAINED = True
    NAME = args.name
    SAVE_PATH = "saved/REC_" + NAME +".sav"

    #train_list, valid_list, test_list = dataset_split(1000, soft=True)
    #train_list, valid_list, test_list = dataset_split(1000, soft=False)
    train_list = list(range(3000))
    valid_list = random.sample(range(3000),100)
    test_list = list(range(3000,4000))
    train_dataset = Num2NumDataset(train_list, BASE, MAX_LEN, reverse=REVERSE)
    valid_dataset = Num2NumDataset(valid_list, BASE, MAX_LEN, reverse=REVERSE)

    #receiver_model = SampleReceiver(type=2) 
    receiver_model = BaselineReceiver()
    receiver_model.cuda()

    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    if args.train:
        optimizer = optim.Adam

        train_model(receiver_model, train_dataset, valid_dataset, 
                    criterion = criterion, 
                    opt = optimizer,
                    lr = LR,
                    max_patience = PATIENCE,
                    max_epoch = MAX_EPOCH,
                    batch_size = BATCH_SIZE,
                    model_path = SAVE_PATH)
    
    print('loading checkpoint...')
    checkpoint = torch.load(SAVE_PATH)
    receiver_model.load_state_dict(checkpoint['receiver_state_dict'])
    
    print('loading test dataset')
    test_dataset = Num2NumDataset(test_list, BASE, MAX_LEN, reverse=REVERSE)
    outputs, targets, digits = run_model(receiver_model, test_dataset, details=True)
    test_loss = criterion(outputs, targets).item()
    
    outputs, targets, _ = run_model(receiver_model, train_dataset)
    train_loss = criterion(outputs, targets).item()
    outputs, targets, _ = run_model(receiver_model, valid_dataset)
    valid_loss = criterion(outputs, targets).item()
    print("train {:.3f}, valid {:.3f}, test {:.3f}".format(train_loss, valid_loss, test_loss))

if __name__ == '__main__':
    numeral_communication()
    