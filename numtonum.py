import math
import random
import argparse
import csv

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from communication_dataset import Num2NumDataset
from communication_dataset import *

from agents import *


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help="model class to use")
parser.add_argument('--load', type=str, help="filename of model to load")
parser.add_argument('--saveas', type=str, help="save result and model as")
parser.add_argument('--train', action="store_true", help="whether to train anew")

parser.add_argument('--reverse', action="store_true", help="reverse digit")

args = parser.parse_args()
options = vars(args)
for o in options:
    print(o, options[o])

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = torch.device("cuda:0")

VOCAB = ['0','1','2','3','4','5','6','7','8','9','SOS','EOS','PAD']



def run_model(sender_model, receiver_model, dataset, details=False):
    dataloader = DataLoader(dataset, batch_size = 200)
    sender_model.eval()
    receiver_model.eval()
    with torch.no_grad():
        outputs = []
        targets = []
        words = []
        for i_batch, sample_batched in enumerate(dataloader):
            word_probs = sender_model(sample_batched['inputs'].to(device, torch.float),
                                         sample_batched['teacher_onehots'].to(device),
                                         sample_batched['teacher_digits'].to(device),
                                         teacher=False)
            word_probs = torch.stack(word_probs, dim=1) #batch*seq*dim
            word_onehots = F.gumbel_softmax(word_probs, tau=1, hard=True)
            #word_onehots = word_probs
            receiver_outputs = receiver_model(word_onehots) #batch*1
            receiver_outputs = receiver_outputs.squeeze() #batch

            
            outputs.append(receiver_outputs)
            targets.append(sample_batched['targets'].to(device, torch.float))
            words.append(word_onehots.argmax(dim=2))

        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        words = torch.cat(words, dim=0)
        
    if details:
        for i in range(len(outputs)):
            print("{}, {}, {}".format(round(targets[i].item(), 1), 
                                        round(outputs[i].item(), 1), 
                                        words[i].tolist()))

    return outputs, targets, words


def train_model(sender_model, receiver_model, train_dataset, valid_dataset,
                criterion, opt, lr, max_patience, max_epoch, batch_size, model_path):
    
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    if False:
        optimizer = opt([ #{'params': sender_model.parameters(), 'lr':lr},
                        {'params': receiver_model.parameters()}], lr=lr) 
    else:
        optimizer = opt([{'params': sender_model.parameters(), 'lr':lr},
                        {'params': receiver_model.parameters()}], lr=lr)

    #mseloss = nn.MSELoss()
    #maeloss = nn.L1Loss()
    maeloss = nn.MSELoss()

    patience = max_patience
    min_loss = float('inf')

    for epoch in range(1, max_epoch + 1):
        sender_model.train()
        receiver_model.train()
        for i_batch, sample_batched in enumerate(train_dataloader):
            word_probs = sender_model(sample_batched['inputs'].to(device, torch.float), 
                                         sample_batched['teacher_onehots'].to(device),
                                         sample_batched['teacher_digits'].to(device),
                                         teacher=False)
            # seq*(batch*dim)
            word_probs = torch.stack(word_probs, dim=1) #batch*seq*dim
            word_onehots = F.gumbel_softmax(word_probs, tau=1, hard=True)
            #word_onehots = word_probs
            receiver_output = receiver_model(word_onehots) #batch*1
            receiver_output = receiver_output.squeeze() #batch

            #loss = mseloss(receiver_output, sample_batched['targets'].to(device, torch.float))
            loss = maeloss(receiver_output, sample_batched['targets'].to(device, torch.float))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #validation
        train_outputs, train_targets, words = run_model(sender_model, receiver_model, train_dataset)
        train_loss = maeloss(train_outputs, train_targets).item()

        outputs, targets, _ = run_model(sender_model, receiver_model, valid_dataset, details=True)
        #valid_loss = mseloss(outputs, targets)
        valid_loss = maeloss(outputs, targets).item()
        
        #for i in range(len(outputs)):
        #    print("target {}, output {:.2f}".format(targets[i].item(), outputs[i].item()))

        if valid_loss < min_loss:
            min_loss = valid_loss
            patience = max_patience
            torch.save({'sender_state_dict': sender_model.state_dict(),
                'receiver_state_dict': receiver_model.state_dict()
                }, model_path)
        
        else:
            patience -= 1 

        print("epoch {}, train_loss {:.3f}, valid_loss {:.3f}, patience {}".format(epoch, train_loss, valid_loss, patience))

        if patience == 0:
            break




def numeral_communication():

    MAX_LEN = 6 #up to 99,999e , which is range(100,000)
    BASE = 10
    REVERSE = True
    TRAIN_RANGE=200
    VALID_RANGE=350
    LR = 0.01
    PATIENCE = 1000 * 1000
    MAX_EPOCH = 10000
    BATCH_SIZE = 50

    HIDDEN_DIM = 100
    NAME = args.saveas
    SAVE_PATH = "saved/S-R_" + NAME + ".sav"

    #train_list, valid_list, test_list = dataset_split(100, True) #8:1:1, 1000
    #train_list, valid_list, test_list = dataset_split(1000, REVERSE)
    #train_list, valid_list, test_list = dataset_split(1000, 'ultimate') #8:1:1, 1000
    #train_list, valid_list, test_list = dataset_split(1000, False) #hard split used for sender
    #train_list, valid_list, test_list = dataset_split(200, 'veryhard')
    train_list, valid_list, test_list = dataset_split(1000, 'work')
    print(train_list)
    print(valid_list)
    print(test_list)
    train_dataset = Num2NumDataset(train_list, BASE, MAX_LEN, reverse=REVERSE)
    valid_dataset = Num2NumDataset(valid_list, BASE, MAX_LEN, reverse=REVERSE)
    test_dataset = Num2NumDataset(test_list, BASE, MAX_LEN, reverse=REVERSE)
    
    # whole = Num2NumDataset(list(range(1000)), BASE, MAX_LEN, reverse=REVERSE)
    # extra_1 = Num2NumDataset(list(range(1000,2000)), BASE, MAX_LEN, reverse=REVERSE)
    # extra_2 = Num2NumDataset(list(range(2000,5000)), BASE, MAX_LEN, reverse=REVERSE)
    # extra_5 = Num2NumDataset(list(range(5000,10000)), BASE, MAX_LEN, reverse=REVERSE)
    # extra_10 = Num2NumDataset(list(range(10000,20000)), BASE, MAX_LEN, reverse=REVERSE)
    # extra_20 = Num2NumDataset(list(range(20000,50000)), BASE, MAX_LEN, reverse=REVERSE)
    # extra_50 = Num2NumDataset(list(range(50000,100000)), BASE, MAX_LEN, reverse=REVERSE)

    # model type selection
    if args.model == 'lstm':
        sender_model = NumToLang_LSTM(hidden_dim=HIDDEN_DIM, vocab_size=13, max_len=MAX_LEN)
    elif args.model == 'mod':
        sender_model = NumToLang_Modulus(hidden_dim=HIDDEN_DIM, vocab_size=13, max_len=MAX_LEN, modulus=10)
    else:
        print("specify --model")
    receiver_model = BaselineReceiver()

    # load pretrained model
    if args.load:
        SENDER_FILENAME = "N2L_" + args.load
        model_path = 'saved/'+ SENDER_FILENAME + '.sav'
        checkpoint = torch.load(model_path)
        sender_model.load_state_dict(checkpoint['best_model_state_dict'])

    sender_model.cuda()
    receiver_model.cuda()

    # train model
    if args.train:
        print("training...")
        criterion = nn.CrossEntropyLoss(ignore_index = BASE+1)
        optimizer = optim.Adam

        train_model(sender_model, receiver_model, train_dataset, valid_dataset, 
                    criterion = criterion, 
                    opt = optimizer,
                    lr = LR,
                    max_patience = PATIENCE,
                    max_epoch = MAX_EPOCH,
                    batch_size = BATCH_SIZE,
                    model_path = SAVE_PATH)
    
    print("loading S-R checkpoint...")
    checkpoint = torch.load(SAVE_PATH)
    sender_model.load_state_dict(checkpoint['sender_state_dict'])
    receiver_model.load_state_dict(checkpoint['receiver_state_dict'])
    
    maeloss = nn.L1Loss()

    outputs, targets, digits = run_model(sender_model, receiver_model, train_dataset)
    train_loss = maeloss(outputs, targets).item()
    print("dataset len", len(train_dataset))
    print(count_unique_sentences(digits))
    print(count_vocab_used(digits))

    outputs, targets, digits = run_model(sender_model, receiver_model, valid_dataset)
    valid_loss = maeloss(outputs, targets).item()
    print("dataset len", len(valid_dataset))
    print(count_unique_sentences(digits))
    print(count_vocab_used(digits))

    outputs, targets, digits = run_model(sender_model, receiver_model, test_dataset)
    test_loss = maeloss(outputs, targets).item()
    print("dataset len", len(test_dataset))
    print(count_unique_sentences(digits))
    print(count_vocab_used(digits))


    #for i in range(len(outputs)):
    #    print("{:.3f}, {:.3f}, {}".format(targets[i].item(), outputs[i].item(), digits[i].tolist()))
    print("train {:.3f}, valid {:.3f}, test {:.3f}".format(train_loss, valid_loss, test_loss))
    

    # generation part
    print("creating generation dataset...")
    generation_dataset = Num2NumDataset([e for e in range(1000)], BASE, MAX_LEN, reverse=REVERSE)    
    outputs, targets, digits = run_model(sender_model, receiver_model, generation_dataset)
    print(1000)
    print(count_unique_sentences(digits))
    #print(count_vocab_used(digits))

    generation_dataset = Num2NumDataset([e for e in range(10000)], BASE, MAX_LEN, reverse=REVERSE)    
    outputs, targets, digits = run_model(sender_model, receiver_model, generation_dataset)
    print(10000)
    print(count_unique_sentences(digits))
    #print(count_vocab_used(digits))

    with open('saved/S-R_' + NAME + '_generated.csv', 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        for i in range(len(targets)):
            row = [targets[i].item()] + [outputs[i].item()] + digits[i].tolist()
            writer.writerow(row)
    #print("evaluation loss: {:.3f}, {:.3f}".format(loss_inter.item(), loss_extra.item()))

def count_unique_sentences(digits):
    sentences = []
    for i in range(len(digits)):
        if digits[i].tolist() in sentences:
            pass
        else:
            sentences.append(digits[i].tolist())
    return len(sentences)

def count_vocab_used(digits):
    words = {}
    for i in range(len(digits)):
        sentence = digits[i].tolist()
        for c in sentence:
            if c in words:
                words[c] += 1
            else:
                words[c] = 1
    return words

            

if __name__ == '__main__':
    numeral_communication()