import argparse
import csv
import logging
import numpy as np
from time import strftime, localtime
from collections import Counter
import math

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
#print(args)

logging.basicConfig(filename='log.log', level=logging.DEBUG)

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

if args.draw_saved:
    #epoch_history_t = 
    #draw(epoch_history_t, training_history, epoch_history_v, validation_history)
    exit()

########## data section #############
# this depends on the task that we want to do:  

class MyDataset(Dataset):
    def __init__(self, x_list, hot_size):
        self.continuous_data(x_list)
        self.discrete_data(x_list, hot_size)
            
    def continuous_data(self, x_list):
        #we are given x_list in python list format, and create x, y pairs
        self.x = torch.tensor(x_list).to(torch.long).to(torch.float)
        self.y = torch.tensor(x_list).to(torch.long).to(torch.float)
        
    def discrete_data(self, x_list, hot_size):
        #for now let's assume there is only contiguous input data.
        self.x_hot = torch.eye(hot_size)[x_list].to(torch.long)

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        #should return numpy array
        return {'x': self.x[idx], 'y': self.y[idx], 'x_hot': self.x_hot[idx]}


##### train data and test data section #######
# what is going to be trained and what is going to be tested??
NUM_DATA = args.num

if args.data == None:
    args.data = 'contiguous'
if args.data == 'contiguous':
    raw_data = [i for i in range(NUM_DATA)]
elif args.data == 'power':
    raw_data = [math.pow(1.2,i) for i in range(NUM_DATA)]
elif args.data == 'uniform':
    raw_data = np.random.uniform(0, args.max, NUM_DATA)
else:
    print("input data not specified")

BATCH_SIZE = args.num
if args.split == None:
    whole_dataset = MyDataset(raw_data, args.max)
    #sampler = WeightedRandomSampler([1/(i+1) for i in range(len(raw_data))], len(raw_data)*2)
    #train_dataloader = DataLoader(whole_dataset, sampler=sampler, batch_size = BATCH_SIZE)
    train_dataloader = DataLoader(whole_dataset, batch_size = BATCH_SIZE)
    valid_dataloader = DataLoader(whole_dataset, batch_size = BATCH_SIZE)
    test_dataloader = DataLoader(whole_dataset, batch_size=BATCH_SIZE)
elif args.split == 'random':
    whole_dataset = MyDataset(raw_data, len(raw_data))
    train_dataset, test_dataset = random_split(whole_dataset, [NUM_DATA-(NUM_DATA//10), NUM_DATA//10])
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
    valid_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
elif args.split == 'end':
    train_dataset = MyDataset(raw_data[0 : NUM_DATA - NUM_DATA//10], len(raw_data))
    test_dataset = MyDataset(raw_data[NUM_DATA - NUM_DATA//10: NUM_DATA], len(raw_data))
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
    valid_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

##### model initialization #####
if args.model == None:
    args.model = 'SR'
if args.model == 'SR':
    model = Sender_Receiver(args)
elif args.model == 'single':
    model = Simple_Linear(args)
elif args.model == 'two':
    model = TwoLayer(args)
else:
    print('model is none')
    model = Sender_Receiver(args)
print("model type : {}".format(args.model))


if args.resume:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

# move model to GPU before constructing optimizer
# https://pytorch.org/docs/stable/optim.html
if args.cuda:
    model.cuda()
    device = torch.device("cuda:0")
else:
    model.cpu()
    device = torch.device("cpu")

#model = model.double()


##### train section ########
LEARNING_RATE = args.lr
#optimizer = optim.Adam([{'params':model.rnncell2.parameters(), 'lr':LEARNING_RATE}], lr=LEARNING_RATE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
if False:
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
current_epoch = 0
end_epoch = args.epoch
if args.resume:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']


def train_one_epoch(model, optimizer, dataloader):
    #train for one epoch
    model.train()
    #mse_loss = nn.MSELoss(reduction='sum')
    #ce_loss = nn.CrossEntropyLoss(reduction='sum')
    
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    epoch_mse = 0
    epoch_ce = 0
    epoch_mse2 = 0
    epoch_length_loss = 0
    SGD = False
    prob_list = []
    reward_list = []
    
    #print("new epoch")
    reg_outputs = []
    class_outputs = []
    # a batch contains batch_size data pairs
    # an episode is about one data pair
    # a RL loss is done on an aggregated rewards over multiple episodes
    # param update is done on batch

    #assume datasize is 10, batch size is 5, update every batch, 
    #feed in as batch
    #for different length, 
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['x'])
        reg_output, class_output = model(sample_batched['x'].unsqueeze(-1).to(device, torch.float),
                                         sample_batched['x_hot'].to(device, torch.float))
        reg_outputs.append(reg_output.squeeze(1))
        class_outputs.append(class_output)
        batch_mse = mse_loss(reg_output.squeeze(1), sample_batched['y'].to(device, torch.float))
        batch_ce = ce_loss(class_output, sample_batched['y'].to(device, torch.long))


        
        model.zero_grad()
        loss = batch_mse * args.regression + batch_ce * args.classification
        loss.backward()
        optimizer.step()
        
        continue

    return 0

    #     batch_ce = args.classification * ce_loss(class_output, sample_batched['y'].to(device, torch.long))
        
    #     epoch_mse += batch_mse
    #     epoch_ce += batch_ce


    #     vocab_batch = torch.stack(model.words)
    #     vocab_batch = torch.transpose(vocab_batch, 0,1)
    #     prob_batch = torch.stack(model.saved_probs)
    #     prob_batch = torch.transpose(prob_batch, 0,1)

    #     for b in range(len(vocab_batch)):
    #         #weight = 1.0 / (1 + sample_batched['x'][b])
    #         #weight = math.exp(sample_batched['x'][b] * -1 -2)
    #         #print(sample_batched['x'][v], weight)
    #         weight = 1
            
    #         for w in range(len(vocab_batch[b])):
    #             if vocab_batch[b,w] == 0:
    #                 if w == 0:
    #                     reward = -1
    #                 else:
    #                     reward = 1
    #             else:
    #                 reward = -0.1
    #             reward_list.append(reward * weight * 5)
    #             prob_list.append(prob_batch[b,w])

    #     #vocabs.append(torch.transpose(vocab_list, 0,1))
    #     #length_loss = -model.saved_probs[i] * reward of choosing that words_input

    #     # if SGD:
    #     #     model.zero_grad()
    #     #     loss = batch_mse + batch_ce
    #     #     loss.backward()
    #     #     optimizer.step()
    # reward_list = torch.tensor(reward_list).to(device, torch.float)
    # reward_list = (reward_list - torch.mean(reward_list)) / (torch.std(reward_list) + 0.001)
    # prob_list = torch.stack(prob_list)
    # # normalize weight?
    # length_loss = torch.mul(reward_list, prob_list).sum() * -1 * 0.05 * args.length_reward
    # length_loss = 0

    # reg_outputs = torch.cat(reg_outputs, 0)
    # class_outputs = torch.cat(class_outputs, 0)
    # #print(reg_outputs)

    # if not SGD:
    #     model.zero_grad()
    #     #epoch_mse_mean = epoch_mse / len(dataloader)
    #     #epoch_ce_mean = epoch_ce / len(dataloader)
    #     loss = epoch_mse + epoch_ce + length_loss
    #     loss = epoch_mse + epoch_ce
    #     loss.backward()
    #     optimizer.step()
    # #print("mean MSE loss = {:.3f}, mean CE loss = {:.3f}".format(epoch_mse_mean, epoch_ce_mean))
    # return loss.item()


def evaluate_model(model, dataloader):
    model.eval()
    # return test result
    mse_loss = nn.MSELoss(reduction='sum')
    ce_loss = nn.CrossEntropyLoss(reduction='sum')
    loss = 0

    x = []
    vocabs = []
    reg_outputs = []
    class_outputs = []
    equality = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            reg_output, class_output = model(sample_batched['x'].unsqueeze(-1).to(device, torch.float),
                                         sample_batched['x_hot'].to(device, torch.float))
            loss += args.regression * mse_loss(reg_output.squeeze(1), sample_batched['y'].to(device, torch.float))
            loss += args.classification * ce_loss(class_output, sample_batched['y'].to(device, torch.long))
            
            x.append(sample_batched['x'])
            if args.model == "SR":
                vocab_list = torch.stack(model.words) #(seqlen, batchsize)
                vocabs.append(torch.transpose(vocab_list, 0,1))
            reg_outputs.append(reg_output)
            class_predicted = torch.argmax(class_output, dim=1)
            equality.append(torch.eq(class_predicted, sample_batched['y'].to(device, torch.long)))
            class_outputs.append(class_predicted)
            
    x = torch.cat(x)
    vocabs = torch.cat(vocabs)
    reg_outputs = torch.cat(reg_outputs).squeeze(1)
    class_outputs = torch.cat(class_outputs)
    equality = torch.cat(equality)
    correct = equality.sum()
    total = equality.numel()
    accuracy = (correct.float() / total).item()

    if args.model == "SR":
        #print("vocabs!!")
        #print(vocabs)
        #print(torch.cat(vocabs, dim=1))
        #vocabs = torch.transpose(torch.cat(vocabs, dim=1), 0, 1) 
        pass
    
    print("evaluation result") 
    print("OX | input|class|regress | words")
    for i, e in enumerate(x):
        if args.model == "SR":
            print("({correct}) {cont:5.1f} -> {cla:2}, {reg:5.2f}, {words}".format(
                cont=x[i].item(), cla=class_outputs[i].item(), reg=reg_outputs[i].item(), 
                words=vocabs[i].tolist(), correct=equality[i].item()))
        else:
            pass
    print("accuracy: {} / {} = {:.3f}".format(correct, total, accuracy))
    return loss.item(), accuracy

def test_receiver(mode):
    model.eval()
    if mode == 'all':  
        theoretical = int(math.pow(args.vocab, args.seq))
        words = [torch.eye(args.vocab)[to_human(e, args.vocab, args.seq)] for e in range(theoretical)]
        words = torch.stack(words)
        #print(words)
        words_input = torch.transpose(words, 0,1)
        #print(words)
        reg_output, class_output = model.receiver_test(words_input.to(device))
        #print(reg_output, class_output)
        print("receiver training result")
        #print(torch.max(torch.softmax(class_output[0], dim=0)))
        for i in range(theoretical):
            print("{} ({:2}) -> {:2} ({:3.1f}), {:3.1f}".format(
                    to_human(i, args.vocab, args.seq),
                    i, 
                    torch.max(class_output[i],dim=0)[1].item(), 
                    torch.max(torch.softmax(class_output[i],dim=0)).item(),
                    reg_output[i][0].item()))



def to_human(n, base, length):
    ret = []
    for i in range(length):
        ret.insert(0,n%base)
        n=n//base
    return ret

def save_model(model, optimizer, epoch):
    PATH = "results/tasked_game.pt"
    torch.save(model.state_dict(), PATH)
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, PATH)

test_every = 10
epoch_history_t = []
training_history = []

epoch_history_v = []
validation_history = []
validation_accuracy_history=[]

epoch_history_test = []
test_history = []
max_valid_accuracy = 0
max_acc_epoch = 0
while True:
    current_epoch += 1
    training_loss = train_one_epoch(model, optimizer, train_dataloader)
    
    if current_epoch % test_every == 0:
        epoch_history_t.append(current_epoch)
        training_history.append(training_loss)
    
    if current_epoch % test_every == 0:
    # evaluate model
        validation_loss, validation_accuracy = evaluate_model(model, valid_dataloader)
        epoch_history_v.append(current_epoch)
        validation_history.append(validation_loss)
        validation_accuracy_history.append(validation_accuracy)
        print("epoch {}, train_loss {:3.3f}, validation_loss {:3.3f}".format(current_epoch, training_loss, validation_loss))
        #for param in model.parameters():
        #    print(param.data)
        if validation_accuracy > max_valid_accuracy:
            max_valid_accuracy = validation_accuracy
            max_acc_epoch = current_epoch
        print("max validation acc: {:.3f} at epoch {}".format(max_valid_accuracy, max_acc_epoch))
        save_model(model, optimizer, current_epoch)
        
    #only when we have unseen test set
    if current_epoch % test_every == 0 and args.split != None:
        test_loss, test_accuracy = evaluate_model(model, test_dataloader)
        epoch_history_test.append(current_epoch)
        test_history.append(test_loss)
        print("test loss {}".format(test_loss))
    
    #test receiver's schemes
    if current_epoch % test_every == 0 and False:
        test_receiver('all')

    if args.visualize and current_epoch % test_every == 0:
        draw(epoch_history_t, training_history, epoch_history_v, validation_history)

    if max_valid_accuracy == 1:
        break

    if current_epoch >= end_epoch:
        break


#save
