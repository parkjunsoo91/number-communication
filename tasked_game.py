import argparse
import csv
import logging
import numpy as np
from time import strftime, localtime
from collections import Counter

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import Categorical


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
parser.add_argument('--cuda', action="store_true", help='use gpu')
parser.add_argument('--max', type=int, default=100, help='numbers to learn')
parser.add_argument('--curri', action="store_true", help='curriculum or not')
parser.add_argument('--seq', type=int, default=2, help='max seq length')
parser.add_argument('--vocab', type=int, default=10, help='vocab size')
parser.add_argument('--epoch', type=int, default=100001, help='epochs to run')
parser.add_argument('--tau', type=float, default=5, help='gumbel temperature')
parser.add_argument('--joint', action="store_true", help='use MSE joint loss')
parser.add_argument('--weighted', action="store_true", help='use gibbs weighted loss')
parser.add_argument('--mweight', type=float, default=0.001, help='MSE loss weight param')
parser.add_argument('--blweight', type=float, default=0.01, help='blank loss weight param')
parser.add_argument('--lr',  type=float, default=0.001, help='Learning Rate')
parser.add_argument('--hidden', type=int, default=512, help='rnn hidden size')
parser.add_argument('--out_hidden', type=int, default=500, help='output hidden size')
parser.add_argument('--tasks', type=int, default=1, help='task')
parser.add_argument('--chapter', type=int, default=0, help='learn up to, if 0, =max')



args = parser.parse_args()
PATH = "tasked_game.pt"
#print(args)

logging.basicConfig(filename='log.log', level=logging.DEBUG)

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class TaskedDataset(Dataset):
    
    def trash_init(self, y_max, x_max, tasks=['ide']):
        datasize = 1000
        x1_float = np.random.uniform(low=0.0, high=x_max, size=datasize)
        x2_float = np.zeros(datasize)
        t = np.zeros(datasize).astype(int)
        x1 = x1_float.astype(int)
        x2 = x2_float.astype(int)
        y = np.around(x1_float).astype(int)


        self.t = torch.tensor(t).to(torch.long)
        self.x1 = torch.tensor(x1).to(torch.long)
        self.x2 = torch.tensor(x2).to(torch.long)
        self.y = torch.tensor(y).to(torch.long)
        self.t_hot = torch.zeros(datasize,1)
        self.x1_hot = torch.eye(y_max)[self.x1]
        self.x2_hot = torch.eye(y_max)[self.x2]
        self.x1_float = x1_float
        self.x2_float = x2_float   
        
        #base notation given in script argument, for purpose of controlling the message
        self.x1_human = torch.tensor([to_human(e, args.vocab, args.seq) for e in x1])
        self.x2_human = torch.tensor([to_human(e, args.vocab, args.seq) for e in x2])
        self.x1_human_weight = (self.x1<=25) #| (self.x1==30) | (self.x1==40) | (self.x1==50) | (self.x1==60) | (self.x1==70) | (self.x1==80) | (self.x1==90)
        #self.x1_human_weight = (self.x1!=25) & (self.x1!=37) & (self.x1!=52) & (self.x1!=74) & (self.x1!=83) & (self.x1!=90) 
        self.x2_human_weight = self.x1_human_weight


    def __init__(self, y_max, x_max, tasks=['ide', 'fir', 'sec', 'big', 'sma', 'add', 'sub']):
        t = []
        x1 = []
        x2 = []
        y = []
        taskset = ['ide', 'fir', 'sec', 'big', 'sma', 'add', 'sub']
        #for this experiement's sake
        tasks=['ide']
        taskset = ['ide']
        for taskname in tasks:
            for i in range(x_max):
                for j in range(x_max):
                    if taskname == "ide":
                        if j != 0:
                            continue
                        t.append(taskset.index(taskname))
                        x1.append(i)
                        x2.append(j)
                        y.append(i)
                    elif taskname == 'fir':
                        t.append(taskset.index(taskname))
                        x1.append(i)
                        x2.append(j)
                        y.append(i)
                    elif taskname == 'sec':
                        t.append(taskset.index(taskname))
                        x1.append(i)
                        x2.append(j)
                        y.append(j)
                    elif taskname == 'big':
                        t.append(taskset.index(taskname))
                        x1.append(i)
                        x2.append(j)
                        y.append(max(i,j))
                    elif taskname == 'sma':
                        t.append(taskset.index(taskname))
                        x1.append(i)
                        x2.append(j)
                        y.append(min(i,j))
                    elif taskname == 'add':
                        if i + j < y_max:
                            t.append(taskset.index(taskname))
                            x1.append(i)
                            x2.append(j)
                            y.append(i+j)
                    elif taskname == 'sub':
                        if i - j >= 0:
                            t.append(taskset.index(taskname))
                            x1.append(i)
                            x2.append(j)
                            y.append(i-j)
        self.t = torch.tensor(t)
        self.x1 = torch.tensor(x1)
        self.x2 = torch.tensor(x2)
        self.y = torch.tensor(y)
        self.t_hot = torch.eye(len(taskset))[self.t]
        self.x1_hot = torch.eye(y_max)[self.x1]
        self.x2_hot = torch.eye(y_max)[self.x2]
        self.x1_float = torch.tensor([float(x) for x in x1])
        self.x2_float = torch.tensor([float(x) for x in x2])
        #base notation given in script argument, for purpose of controlling the message
        self.x1_human = torch.tensor([to_human(e, args.vocab, args.seq) for e in x1])
        self.x2_human = torch.tensor([to_human(e, args.vocab, args.seq) for e in x2])
        self.x1_human_weight = (self.x1<=25) #| (self.x1==30) | (self.x1==40) | (self.x1==50) | (self.x1==60) | (self.x1==70) | (self.x1==80) | (self.x1==90)
        #self.x1_human_weight = (self.x1!=25) & (self.x1!=37) & (self.x1!=52) & (self.x1!=74) & (self.x1!=83) & (self.x1!=90) 
        self.x2_human_weight = self.x1_human_weight

    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return {'t':self.t[idx], 'x1': self.x1[idx], 'x2': self.x2[idx], 'y':self.y[idx], 
                't_hot':self.t_hot[idx], 'x1_hot': self.x1_hot[idx], 'x2_hot': self.x2_hot[idx],
                'x1_float':self.x1_float[idx], 'x2_float':self.x2_float[idx], 
                'x1_human': self.x1_human[idx], 'x2_human':self.x2_human[idx], 'x1_human_weight':self.x1_human_weight[idx], 'x2_human_weight':self.x2_human_weight[idx] }

def to_human(n, base, length):
    ret = []
    for i in range(length):
        ret.insert(0,n % base)
        #ret.append(n%base)
        n = n // base
    return ret

class Sender_Receiver(nn.Module):
    def __init__(self, params):
        super(Sender_Receiver, self).__init__()
        for p in params:
            setattr(self, p, params[p])

        #self.i_h = nn.Linear(self.input_size, self.sender_hidden_size)
        self.i_h = nn.Linear(1, self.sender_hidden_size) #case of continous input only

        #sender regression
        self.s_r = nn.Linear(self.sender_hidden_size, 1)

        self.sender_grucell = nn.GRUCell(self.vocab_size, self.sender_hidden_size) #gru
        #self.sender_lstmcell = nn.LSTMCell(self.vocab_size, self.sender_hidden_size) #lstm

        # self.i2h = nn.Linear(self.vocab_size + self.sender_hidden_size, self.sender_hidden_size)
        # self.i2o = nn.Linear(self.vocab_size + self.sender_hidden_size, self.output_size)

        self.h_w = nn.Linear(self.sender_hidden_size, self.vocab_size)

        self.receiver_gru = nn.GRU(self.vocab_size, self.receiver_hidden_size, 1)
        self.h_o = nn.Linear(self.receiver_hidden_size, self.output_size)
        self.c_ho = nn.Linear(self.task_size + self.receiver_hidden_size * 2, self.output_hidden_size)
        self.ho_o = nn.Linear(self.output_hidden_size, self.output_size)
        self.ho_r = nn.Linear(self.output_hidden_size, 1)

        self.set_train()

    def set_train(self):
        self.train()
        self.eval_mode = False
    def set_eval(self):
        self.eval()
        self.eval_mode = True

    def input_to_sentence(self, x_hot):
        batch_size = x_hot.size()[0]
        h = self.i_h(x_hot)
        sr = self.s_r(h)
        #c = self.i_h(x_hot)
        c = torch.zeros_like(h) #if lstm
        input_word = torch.zeros(batch_size, self.vocab_size, device=x_hot.device)
        output_words = torch.zeros(self.max_seq_len, batch_size, self.vocab_size, device=x_hot.device)
        output_scores = torch.zeros(self.max_seq_len, batch_size, self.vocab_size, device=x_hot.device)
        for t in range(self.max_seq_len):
            h = self.sender_grucell(input_word, h)
            #h, c = self.sender_lstmcell(input_word, (h, c)) #if lstm
            output_score = self.h_w(h)
            output_scores[t] = F.log_softmax(output_score, dim=1)
            if self.eval_mode:
                output_word = torch.eye(output_score.size()[1])[torch.argmax(output_score, dim=1)].to(device=x_hot.device)
            else:
                output_word = F.gumbel_softmax(output_score, hard = True, tau = self.tau)
            output_words[t] = output_word
            #input_word = output_word.detach()  #what if we don't detach?
            input_word = output_word
        return output_words, output_scores, sr

    def sentence_to_hidden(self, sentence):
        batch_size = sentence.size()[1]
        h_0 = torch.zeros(1, batch_size, self.receiver_hidden_size, device=sentence.device)
        #output(seq_len, batch, num_directions*hidden_size)
        #h_n(layers*directions, batch, hidden_size)
        output, h_n = self.receiver_gru(sentence, h_0)
        h_n = h_n.view(batch_size, self.receiver_hidden_size)
        return output, h_n

    def forward(self, x1_hot, x2_hot, t_hot, x1_float, x2_float):
        #combined input
        #x1_input = torch.cat((x1_hot, x1_float.unsqueeze(-1)), dim=1)
        #x2_input = torch.cat((x2_hot, x2_float.unsqueeze(-1)), dim=1)
        
        #continuous input only
        x1_input = x1_float.unsqueeze(-1)
        x2_input = x1_float.unsqueeze(-1)
        
        #discrete input only
        #x1_input = x1_hot
        #x2_input = x2_hot

        sentence1, score1, sender_reg1 = self.input_to_sentence(x1_input)
        sentence2, score2, sender_reg2 = self.input_to_sentence(x2_input)
        #sentence: size(seq, batch, vocab)
        self.sent1 = sentence1
        self.sent2 = sentence2
        self.score1 = score1
        self.score2 = score2
        self.sender_reg1 = sender_reg1
        self.sender_reg2 = sender_reg2
        self.words1 = torch.transpose(torch.argmax(sentence1, dim=2), 0, 1)
        self.words2 = torch.transpose(torch.argmax(sentence2, dim=2), 0, 1)

        _, h1 = self.sentence_to_hidden(sentence1)
        _, h2 = self.sentence_to_hidden(sentence2)
        
        guess1_score = self.h_o(h1)
        guess2_score = self.h_o(h2)
        combined = torch.cat((t_hot, h1, h2), dim=1)
        output_hidden = torch.relu(self.c_ho(combined))
        output_score = self.ho_o(output_hidden)
        output_regress = self.ho_r(output_hidden)

        return output_score, guess1_score, guess2_score, output_regress, sender_reg1, sender_reg2

def train():

    dtype = torch.float
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda:0")
    cpu = torch.device('cpu')

    tasks = ['ide', 'fir', 'sec', 'big', 'sma', 'add', 'sub']
    tasks = tasks[0:args.tasks]
    tasks = ['ide']
    print(tasks)
    MAX_NUM = args.max
    MAX_SEQ_LEN = args.seq
    VOCAB = args.vocab
    params = {}
    params['input_size'] = MAX_NUM +1 -1 #for discrete input only
    params['sender_hidden_size'] = args.hidden
    params['vocab_size'] = VOCAB #+ 1
    params['max_seq_len'] = MAX_SEQ_LEN
    params['receiver_hidden_size'] = args.hidden
    params['output_hidden_size'] = args.out_hidden
    params['output_size'] = MAX_NUM
    params['tau'] = args.tau
    #arams['task_size'] = len(['ide', 'fir', 'sec', 'big', 'sma', 'add', 'sub'])
    params['task_size'] = 1
    model = Sender_Receiver(params).to(device)
    if args.cuda:
        model.cuda()

    # dataset = TaskedDataset(MAX_NUM, args.curri) # or this  part
    # dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    #data_weight = torch.tensor([1/(n+1) for n in range(MAX_NUM)], device=device)

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    epoch = 0
    if args.resume:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict']),
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    print(args)
    logging.debug(args)
    end = False
    if args.curri:
        #chapters = [100]
        chapters = [20, 30, 40, 50, 60, 70, 80, 90, 95, 96,97,98,99,100]
        #chapters = [i for i in range(20,101)]
        if args.vocab == 2:
            if args.seq == 5:
                chapters = [3,4,5,6,7,8,10,12,16,20,24,28,32]
            if args.seq == 6:
                chapters = [3,4,5,6,7,8,10,12,16,20,24,28,32,40,48,56,64]
            if args.seq == 7:
                chapters = [3,4,5,6,7,8,10,12,16,20,24,28,32,40,48,56,64,80,96,112,128]
        if args.vocab == 3:
            if args.seq == 3:
                chapters = [3, 6, 9, 18, 27]
            if args.seq == 4:
                chapters = [3, 6, 9, 18, 27, 54, 91]
        if args.vocab == 4:
            if args.seq == 3:
                chapters = [3,4,5,6,7,8,10,12,16,20,24,28,32,40,48,56,64]
    else:
        chapters = [args.max]
    #chapters = [91]

    if args.chapter != 0:
        chapters = [args.chapter]

                    
    epochs = []
    max_accuracy = 0
    max_acc_epoch = 0
    for chapter in chapters: #max learning
        #prepare chapter data
        dataset = TaskedDataset(MAX_NUM, chapter, tasks)
        test_dataset = TaskedDataset(MAX_NUM, chapter, ['ide'])

        #do a test-train split
        train_size = int(1 * len(dataset))
        test_size = len(dataset) - train_size
        #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        # but for now we do only train set
        #training data
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
        

        #test data for visualization
        dataloader_test = DataLoader(test_dataset, batch_size=chapter, shuffle=False)
        
        #data weight in Gibb's distribution
        if False:
            data_weight = torch.tensor([1/(n+1) for n in range(MAX_NUM)], device=device)
        else:
            data_weight = None
        
        #loss terms
        CELoss = nn.CrossEntropyLoss()
        MSELoss = nn.MSELoss()
        CELoss_human = nn.CrossEntropyLoss(reduction='none')
        epochs.append(epoch)

        while True:
            #train
            epochs[-1] += 1
            epoch_loss = 0
            for i_batch, sample_batched in enumerate(dataloader):
                output_score, guess1_score, guess2_score, output_regress, sender_reg1, sender_reg2 = model(
                                                                sample_batched['x1_hot'].to(device=device, dtype=dtype),
                                                                sample_batched['x2_hot'].to(device=device, dtype=dtype),
                                                                sample_batched['t_hot'].to(device=device, dtype=dtype),
                                                                sample_batched['x1_float'].to(device=device, dtype=dtype),
                                                                sample_batched['x2_float'].to(device=device, dtype=dtype))
                #print(output_score.size())
                #print(sample_batched['y'].size())
                celoss = CELoss(output_score, sample_batched['y'].to(device))
                batch_loss = 0
                batch_loss = celoss * 1.0 * 0.01 

                #loss of receiver's guess of the 2 inputs
                if False:
                    celoss_x1 = CELoss(guess1_score, sample_batched['x1'].to(device))
                    celoss_x2 = CELoss(guess2_score, sample_batched['x2'].to(device))
                    batch_loss += (celoss_x1 + celoss_x2) * 0.0


                # add reward for overusing vocab
                if False:
                    sentence1 = model.sent1
                    sentence2 = model.sent2
                    #word_freq = torch.add(sentence1.sum(dim=0).sum(dim=0), sentence2.sum(dim=0).sum(dim=0))
                    word_freq = sentence1.sum(dim=0).sum(dim=0)
                    total = word_freq.sum()
                    alpha = 0.1
                    word_log_prob = torch.log(word_freq + 0.01 / (alpha + total - 1))
                    #all_words = torch.reshape(torch.cat((sentence1, sentence2), dim=1), (-1, word_freq.size()[0]))
                    all_words = torch.reshape(sentence1, (-1, word_freq.size()[0]))
                    reward = torch.dot(all_words.sum(dim=0), word_log_prob) / all_words.size()[0]
                    #batch_loss -= reward * 0.001 

                #MSE loss of sender's notion of input
                if False:
                    mseloss = 0
                    mseloss_sender = MSELoss(torch.cat((sender_reg1.squeeze(), sender_reg2.squeeze())), torch.cat((sample_batched['x1'], sample_batched['x2'])).to(device=device, dtype=dtype))
                    batch_loss += mseloss_sender * 0.01
                
                #MSE loss of receiver's guess
                if args.joint:
                    mseloss = MSELoss(output_regress.squeeze(), sample_batched['y'].to(device=device, dtype=dtype))
                    batch_loss += mseloss * 0.1

                epoch_loss += batch_loss
                model.zero_grad()
                #torch.cuda.synchronize()
                batch_loss.backward()
                optimizer.step()

            if epochs[-1] % 100 != 0:
                continue

            with torch.no_grad():
                model.set_eval()
                equality = torch.tensor([], dtype=torch.uint8)
                words1 = torch.tensor([], dtype=torch.long)
                words2 = torch.tensor([], dtype=torch.long)
                x1 = torch.tensor([], dtype=torch.long)
                x2 = torch.tensor([], dtype=torch.long)
                y = torch.tensor([], dtype=torch.long)
                t = torch.tensor([], dtype=torch.long)
                pred = torch.tensor([], dtype=torch.long)
                w1pred = torch.tensor([], dtype=torch.long)
                w2pred = torch.tensor([], dtype=torch.long)
                regress = torch.tensor([], dtype=dtype)
                sender_reg1 = torch.tensor([], dtype=dtype)
                sender_reg2 = torch.tensor([], dtype=dtype)
                for i_batch, sample_batched in enumerate(dataloader):
                    output_score, sw1, sw2, reg, sr1, sr2 = model(sample_batched['x1_hot'].to(device=device, dtype=dtype),
                                                   sample_batched['x2_hot'].to(device=device, dtype=dtype),
                                                   sample_batched['t_hot'].to(device=device, dtype=dtype),
                                                   sample_batched['x1_float'].to(device=device, dtype=dtype),
                                                   sample_batched['x2_float'].to(device=device, dtype=dtype))
                    predicted = torch.argmax(output_score, dim=1).to(device=cpu)
                    w1predicted = torch.argmax(sw1, dim=1).to(device=cpu)
                    w2predicted = torch.argmax(sw2, dim=1).to(device=cpu)
                    target = sample_batched['y']
                    equality_batch = torch.eq(predicted, target)
                    equality = torch.cat((equality, equality_batch), dim=0)
                    words1_batch = model.words1.to(device=cpu)
                    words2_batch = model.words2.to(device=cpu)
                    words1 = torch.cat((words1, words1_batch), dim=0)
                    words2 = torch.cat((words2, words2_batch), dim=0)
                    t = torch.cat((t, sample_batched['t']))
                    x1 = torch.cat((x1, sample_batched['x1']))
                    x2 = torch.cat((x2, sample_batched['x2']))
                    y = torch.cat((y, sample_batched['y']))
                    pred = torch.cat((pred, predicted), dim=0)
                    w1pred = torch.cat((w1pred, w1predicted), dim=0)
                    w2pred = torch.cat((w2pred, w2predicted), dim=0)
                    regress = torch.cat((regress, reg.to(device=cpu)), dim=0)
                    sender_reg1 = torch.cat((sender_reg1, sr1.to(device=cpu)), dim=0)
                    sender_reg2 = torch.cat((sender_reg2, sr2.to(device=cpu)), dim=0)
                model.set_train()
                #get statistics
                correct = equality.sum()
                total = equality.numel()
                accuracy = float(correct) / total 
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_acc_epoch = epochs


                # map:{number input : [ch1, ch2, ch3], ...}
                # translate:{vocab1: translate, ...}
                charmap = {}
                frequentist_translator = {}
                frequentist_counter = Counter()
                for i in range(len(y)):
                    charmap[x1[i].item()] = [e.item() for e in words1[i]]
                    for e in words1[i]:
                        frequentist_counter[e.item()] += 1
                chars = [e for e in range(args.vocab)]
                char_counter = 0
                for e in frequentist_counter.most_common():
                    frequentist_translator[e[0]] = chars[char_counter]
                    char_counter+=1

                #print(charmap)

                if True:
                    for i in range(len(y)):
                        print("({c}) {t} ({x1:2}, {x2:2}) = {y:2} -> {p:2} ({reg:5.1f}) {x1:2}={w1}({sr1:5.1f})-{w1p:2}, {x2:2}={w2}({sr2:5.1f})-{w2p:2}".format(
                                                                                        t=tasks[t[i]], x1=x1[i], x2=x2[i], y=y[i], p=pred[i], c=equality[i],
                                                                                        w1=[e.item() for e in words1[i]], w2=[e.item() for e in words2[i]], 
                                                                                        w1p=w1pred[i], w2p=w2pred[i], reg=regress[i].item(),
                                                                                        sr1= sender_reg1[i].item(), sr2=sender_reg2[i].item()))
                
                for k in sorted(charmap):
                    print("{}, {}".format(k, [frequentist_translator[e] for e in charmap[k]]))
                
                print("chapter {}, epoch {}, loss {:.5f}".format(chapter, epochs[-1], epoch_loss))
                print("accuracy: {} / {} = {:.3f}".format(correct, total, accuracy))
                print("max acc: {}, epoch {}".format(max_accuracy, max_acc_epoch))
                logging.debug("{} chapter{}, epoch {}, loss:{:.5f}, acc: {}/{} = {:.3%}".format(strftime("%H:%M", localtime()), chapter, epochs[-1], epoch_loss, correct, total, accuracy))

                if accuracy == 1.00:
                    break

                #test set
                model.set_eval()
                equality = torch.tensor([], dtype=torch.uint8)
                words1 = torch.tensor([], dtype=torch.long)
                words2 = torch.tensor([], dtype=torch.long)
                x1 = torch.tensor([], dtype=torch.long)
                x2 = torch.tensor([], dtype=torch.long)
                y = torch.tensor([], dtype=torch.long)
                t = torch.tensor([], dtype=torch.long)
                pred = torch.tensor([], dtype=torch.long)
                w1pred = torch.tensor([], dtype=torch.long)
                w2pred = torch.tensor([], dtype=torch.long)
                regress = torch.tensor([], dtype=dtype)
                sender_reg1 = torch.tensor([], dtype=dtype)
                sender_reg2 = torch.tensor([], dtype=dtype)
                for i_batch, sample_batched in enumerate(dataloader_test):
                    output_score, sw1, sw2, reg, sr1, sr2 = model(sample_batched['x1_hot'].to(device=device, dtype=dtype),
                                                   sample_batched['x2_hot'].to(device=device, dtype=dtype),
                                                   sample_batched['t_hot'].to(device=device, dtype=dtype),
                                                   sample_batched['x1_float'].to(device=device, dtype=dtype),
                                                   sample_batched['x2_float'].to(device=device, dtype=dtype))
                    predicted = torch.argmax(output_score, dim=1).to(device=cpu)
                    w1predicted = torch.argmax(sw1, dim=1).to(device=cpu)
                    w2predicted = torch.argmax(sw2, dim=1).to(device=cpu)
                    target = sample_batched['y']
                    equality_batch = torch.eq(predicted, target)
                    equality = torch.cat((equality, equality_batch), dim=0)
                    words1_batch = model.words1.to(device=cpu)
                    words2_batch = model.words2.to(device=cpu)
                    words1 = torch.cat((words1, words1_batch), dim=0)
                    words2 = torch.cat((words2, words2_batch), dim=0)
                    t = torch.cat((t, sample_batched['t']))
                    x1 = torch.cat((x1, sample_batched['x1']))
                    x2 = torch.cat((x2, sample_batched['x2']))
                    y = torch.cat((y, sample_batched['y']))
                    pred = torch.cat((pred, predicted), dim=0)
                    w1pred = torch.cat((w1pred, w1predicted), dim=0)
                    w2pred = torch.cat((w2pred, w2predicted), dim=0)
                    regress = torch.cat((regress, reg.to(device=cpu)), dim=0)
                    sender_reg1 = torch.cat((sender_reg1, sr1.to(device=cpu)), dim=0)
                    sender_reg2 = torch.cat((sender_reg2, sr2.to(device=cpu)), dim=0)
                model.set_train()
                #get statistics
                correct = equality.sum()
                total = equality.numel()
                if total != 0:
                    accuracy = float(correct) / total
                else:
                    accuracy = 0

                charmap = {}
                for i in range(len(y)):
                    charmap[x1[i].item()] = [e.item() for e in words1[i]]
                for k in sorted(charmap):
                    print("{}, {}".format(k, [frequentist_translator[e] for e in charmap[k]]))

                
                #print them all    
                if True:
                    for i in range(len(y)):
                        print("({c}) {t} ({x1:2}, {x2:2}) = {y:2} -> {p:2} ({reg:5.1f}) {x1:2}={w1}({sr1:5.1f})-{w1p:2}, {x2:2}={w2}({sr2:5.1f})-{w2p:2}".format(
                                                                                        t=tasks[t[i]], x1=x1[i], x2=x2[i], y=y[i], p=pred[i], c=equality[i],
                                                                                        w1=[e.item() for e in words1[i]], w2=[e.item() for e in words2[i]], 
                                                                                        w1p=w1pred[i], w2p=w2pred[i], reg=regress[i].item(),
                                                                                        sr1= sender_reg1[i].item(), sr2=sender_reg2[i].item()))
                print("chapter {}, epoch {}, loss {:.5f}".format(chapter, epochs[-1], epoch_loss))
                print("accuracy: {} / {} = {:.3f}".format(correct, total, accuracy))

                #proceed to next chapter
                if accuracy >= 1.00 and chapter != chapters[-1]:
                    break
                if epochs[-1] > args.epoch + 1:
                    break
                if chapter == chapters[-1] and accuracy > 0.9999:
                    break



    #at the end

        # if epoch % 500 == 0:
        #     with torch.no_grad():
        #         model.set_eval()
        #         with open('results.csv', 'w') as csvfile:
        #             fieldnames = ['op', 'arg1', 'arg2', 'ans', 'pred', 'estim' ,'cor', 'msg0', 'msg1','msg2','msg3','msg4','msg5','msg6','msg7','msg8','msg9']
        #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
        #             writer.writeheader()

        #             equality = torch.tensor([], dtype=torch.uint8)
        #             words_cat = torch.tensor([], dtype=torch.long)
        #             for i_batch, sample_batched in enumerate(dataloader):
        #                 output_score, output_value = model(sample_batched['x_hot'].to(device=device, dtype=dtype), args.cuda)
        #                 predicted = torch.argmax(torch.softmax(output_score.to(device=cpu), dim=1), dim=1)
        #                 predicted_value = output_value.squeeze().to(device=cpu)
        #                 equal = torch.eq(predicted, sample_batched['y'])
        #                 equality = torch.cat((equality, equal))
        #                 words = torch.transpose(model.words.to(device=cpu), 0, 1)
        #                 words_cat = torch.cat((words_cat, torch.transpose(model.words.to(device=cpu), 0, 1)))
        #                 for i in range(len(output_score)):
        #                     writedict = {'op': "", 'arg1': sample_batched['x'][i].item(), 'arg2': "",
        #                                 'ans': sample_batched['y'][i].item(), 'pred': predicted[i].item(), 'estim': predicted_value[i].item(), 'cor': equal[i].item()}
        #                     for w in range(MAX_SEQ_LEN):
        #                         writedict['msg'+str(w)] = words[i][w].item()
        #                     writer.writerow(writedict)
        #                     print("{} {} {:.2f} '{}'".format(writedict['arg1'], writedict['pred'], predicted_value[i], [writedict['msg'+str(w)] for w in range(MAX_SEQ_LEN)]))
        #             correct = torch.sum(equality).item()
        #             total = equality.numel()
        #             print("acc: {}/{} = {:.2%}".format(correct, total, correct/total))
        #             logging.debug("epoch {} acc: {}/{} = {:.2%}, loss:{:.4f}".format(epoch, correct, total, correct/total, loss_sum))
        #             if correct == total:
        #                 end = True

        #         model.set_train()
            torch.save({'epoch': epochs[-1],
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)
            if end == True:
                break

if __name__ == "__main__":
    train()