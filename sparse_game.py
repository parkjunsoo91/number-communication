import argparse
import csv
import logging
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical



parser = argparse.ArgumentParser()
parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
parser.add_argument('--cuda', action="store_true", help='use gpu')
parser.add_argument('--max', type=int, default=100, help='numbers to learn')
parser.add_argument('--curri', type=int, default=10, help='curriculum part')
parser.add_argument('--seq', type=int, default=2, help='max seq length')
parser.add_argument('--vocab', type=int, default=10, help='vocab size')
parser.add_argument('--epoch', type=int, default=100001, help='epochs to run')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--tau', type=float, default=5, help='gumbel temperature')
parser.add_argument('--joint', action="store_true", help='use MSE joint loss')
parser.add_argument('--mweight', type=float, default=0.001, help='MSE loss weight param')
parser.add_argument('--blweight', type=float, default=0.01, help='blank loss weight param')
parser.add_argument('--lr',  type=float, default=0.001, help='Learning Rate')
parser.add_argument('--hidden', type=int, default=512, help='rnn hidden size')


args = parser.parse_args()
PATH = "sparse_game.pt"

logging.basicConfig(filename='log.log', level=logging.DEBUG)

#1hot number representation with task and answer
#

class ArithmeticDataset(Dataset):
    def __init__(self, max, curriculum):
        self.x = torch.tensor([e for e in range(max)])
        self.x_hot = torch.eye(max)[self.x]
        self.y = torch.tensor([e for e in self.x])
        tasks = ['identity', 'addition', 'subtraction']
        self.t_hot = torch.eye(len(tasks))
        
        self.curriculum = curriculum

    def __len__(self):
        return self.curriculum #len(self.y)
    def __getitem__(self, idx):
        return {'x_hot': self.x_hot[idx], 'x': self.x[idx], 'y': self.y[idx]}

class CurriculumSampler(DataLoader):
    def __init__(self, data_source):
        self.data_source = data_source
    def __iter__(self):
        return iter(range(len(self.data_source)))


class ST_gumbel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hard=False, tau=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        y_soft = F.gumbel_softmax(input, hard=hard, tau=tau)
        ctx.save_for_backward(y_soft)
        if hard:
            _, k = y_soft.max(-1)
            # this bit is based on
            # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
            y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
            # this cool bit of code achieves two things:
            # - makes the output value exactly one-hot (since we add then
            #   subtract y_soft value)
            # - makes the gradient equal to y_soft gradient (since we strip
            #   all other gradients)
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class Sender_Receiver(nn.Module):
    def __init__(self, params):
        super(Sender_Receiver, self).__init__()
        
        for p in params:
            setattr(self, p, params[p])
        #input 1-hot -> hidden repr
        self.i_h = nn.Linear(self.input_size, self.sender_hidden_size)
        #feed empty word, then output words into rnn input
        self.wh_h = nn.GRUCell(self.vocab_size, self.sender_hidden_size)
        self.h_w = nn.Linear(self.sender_hidden_size, self.vocab_size)

        self.w_h = nn.GRU(self.vocab_size, self.receiver_hidden_size, 1)
        self.h_o = nn.Linear(self.receiver_hidden_size, self.output_size)
        self.h_v = nn.Linear(self.receiver_hidden_size, 1)
        
        self.drop = nn.Dropout(self.dropout)

        self.set_train()

    def set_train(self):
        self.train()
        self.hard = True
    def set_eval(self):
        self.eval()
        self.hard = True

    def forward(self, x, cuda):
        dev = torch.device("cpu")
        if cuda:
            dev = torch.device("cuda:0")
        batch_size = x.size()[0]

        #input -> hidden
        sender_hidden = self.drop(self.i_h(x))
        #hidden + 0word -> hidden -> output_word
        #hidden + output_word -> hidden
        input_word = torch.zeros(batch_size, self.vocab_size, device=x.device)
        output_words = torch.zeros(self.max_seq_len, batch_size, self.vocab_size, device=x.device)
        for t in range(self.max_seq_len):
            sender_hidden = self.wh_h(input_word, sender_hidden)

            # y_soft = F.gumbel_softmax(self.h_w(sender_hidden), hard=False, tau=self.tau)
            # _, k = y_soft.max(-1)
            # y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
            # y = y_hard - y_soft.detach() + y_soft
            
            # output_word = y
            

            # distr_word = Categorical(y_soft)
            # distr_word.log_prob(k)

            output_word = F.gumbel_softmax(self.h_w(sender_hidden), hard=True, tau=self.tau)

            #probability of sampling the argmax term?

            output_words[t] = output_word
            input_word = output_word.detach()
        self.words_gumbel = output_words #words as 1-hot
        #0'th word occurence as negative loss
        self.words = torch.argmax(output_words, dim=2) #words as number id

        # #zero padding
        # for b in range(batch_size):
        #     for i in range(1,self.max_seq_len):
        #         if self.words[i-1,b] == 0:
        #             output_words[i,b] = torch.zeros(self.vocab_size, device = dev)
        #             self.words[i,b] = 0

        #words-> rnn hidden

        _, h_n = self.w_h(output_words, torch.zeros(1, batch_size, self.receiver_hidden_size, device=x.device))
        #h_n shape:(num_layers * num_directions, batch, hidden_size)
        h_n = h_n.view(batch_size, self.receiver_hidden_size)
        h_n = self.drop(h_n)
        score = self.h_o(h_n)
        value = self.h_v(h_n)
        return score, value

def train():

    dtype = torch.float
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda:0")
    cpu = torch.device('cpu')

    MAX_NUM = args.max
    MAX_SEQ_LEN = args.seq
    VOCAB = args.vocab
    params = {}
    params['batch_size'] = MAX_NUM
    params['input_size'] = MAX_NUM
    params['sender_hidden_size'] = args.hidden
    params['vocab_size'] = VOCAB #+ 1
    params['max_seq_len'] = MAX_SEQ_LEN
    params['receiver_hidden_size'] = args.hidden
    params['output_size'] = MAX_NUM
    params['dropout'] = args.dropout
    params['tau'] = args.tau
    
    model = Sender_Receiver(params).to(device)

    dataset = ArithmeticDataset(MAX_NUM, args.curri) # or this  part
    dataloader = DataLoader(dataset, batch_size=10)
    
    data_weight = torch.tensor([1/(n+1) for n in range(MAX_NUM)], device=device)
    CELoss = nn.CrossEntropyLoss(weight=data_weight)
    MSELoss = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    epoch = 0
    if args.resume:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict']),
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    print(args)
    logging.debug(args)
    end = False
    for epoch in range(epoch, epoch + args.epoch):
        loss_sum = 0
        for i_batch, sample_batched in enumerate(dataloader): #modify this part as to provide curriculum learning
            output_score, output_value = model(sample_batched['x_hot'].to(device=device, dtype=dtype), args.cuda)
            celoss = CELoss(output_score, sample_batched['y'].to(device=device, dtype=torch.long))
            loss = celoss
            mseloss = 0
            if args.joint:
                mseloss_elementwise = MSELoss(output_value.squeeze(), sample_batched['y'].to(device=device, dtype=dtype))
                mseloss = torch.sum(torch.mul(mseloss_elementwise, data_weight)) * args.mweight
                loss = celoss + mseloss
                
                sss = torch.sum(model.words_gumbel[:,:,0], dim=0)
                blankloss = 1 / (torch.sum(torch.mul(sss, data_weight)) + 1) 
                
                loss += blankloss * args.blweight
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
            if epoch % 500 == 0:
                print("MSEloss {}, CEloss {}".format(mseloss, celoss))
        if epoch % 500 == 0:
            print("epoch {}, loss {}".format(epoch, loss_sum))

        if epoch % 500 == 0:
            with torch.no_grad():
                model.set_eval()
                with open('results.csv', 'w') as csvfile:
                    fieldnames = ['op', 'arg1', 'arg2', 'ans', 'pred', 'estim' ,'cor', 'msg0', 'msg1','msg2','msg3','msg4','msg5','msg6','msg7','msg8','msg9']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
                    writer.writeheader()

                    equality = torch.tensor([], dtype=torch.uint8)
                    words_cat = torch.tensor([], dtype=torch.long)
                    for i_batch, sample_batched in enumerate(dataloader):
                        output_score, output_value = model(sample_batched['x_hot'].to(device=device, dtype=dtype), args.cuda)
                        predicted = torch.argmax(torch.softmax(output_score.to(device=cpu), dim=1), dim=1)
                        predicted_value = output_value.squeeze().to(device=cpu)
                        equal = torch.eq(predicted, sample_batched['y'])
                        equality = torch.cat((equality, equal))
                        words = torch.transpose(model.words.to(device=cpu), 0, 1)
                        words_cat = torch.cat((words_cat, torch.transpose(model.words.to(device=cpu), 0, 1)))
                        for i in range(len(output_score)):
                            writedict = {'op': "", 'arg1': sample_batched['x'][i].item(), 'arg2': "",
                                        'ans': sample_batched['y'][i].item(), 'pred': predicted[i].item(), 'estim': predicted_value[i].item(), 'cor': equal[i].item()}
                            for w in range(MAX_SEQ_LEN):
                                writedict['msg'+str(w)] = words[i][w].item()
                            writer.writerow(writedict)
                            print("{} {} {:.2f} '{}'".format(writedict['arg1'], writedict['pred'], predicted_value[i], [writedict['msg'+str(w)] for w in range(MAX_SEQ_LEN)]))
                    correct = torch.sum(equality).item()
                    total = equality.numel()
                    print("acc: {}/{} = {:.2%}".format(correct, total, correct/total))
                    logging.debug("epoch {} acc: {}/{} = {:.2%}, loss:{:.4f}".format(epoch, correct, total, correct/total, loss_sum))
                    if correct == total:
                        end = True

                model.set_train()
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)
            if end == True:
                break

if __name__ == "__main__":
    train()