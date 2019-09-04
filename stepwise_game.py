import argparse
import csv
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
parser.add_argument('--cuda', action="store_true", help='use gpu')
args = parser.parse_args()
PATH = "stepwise_game.pt"
def int_to_binary(d, n):
    fs = '{0:0' + str(n) + 'b}'
    b_str = fs.format(d)
    return [int(c) for c in b_str]
def binary_to_int(b):
    sum = 0
    for i in range(len(b)):
        sum *= 2
        sum += b[i]
    return sum
def generate_data(num_digits, operations = ['add', 'sub', 'mul', 'fir', 'sec', 'ide', 'inc']): #operation(8), arg1(8), arg2(8), answer(8)
            
    task_size = len(operations)
    
    max_number = pow(2,num_digits)
    train_data = []
    test_data = []
    for t in operations:
        if t == 'add':
            data = []
            taskvec = [0] * len(operations)
            taskvec[operations.index(t)] = 1
            for i in range(max_number):
                for j in range(max_number):
                    if i + j < max_number:
                        a = int_to_binary(i, num_digits)
                        b = int_to_binary(j, num_digits)
                        c = int_to_binary(i+j, num_digits)
                        data.append(taskvec + a + b + c)
            x_train, x_test = train_test_split(data, test_size=0.1)
            train_data += x_train
            test_data += x_test
            
        if t == 'sub':
            data = []
            taskvec = [0] * len(operations)
            taskvec[operations.index(t)] = 1
            for i in range(max_number):
                for j in range(max_number):
                    if i - j >= 0:
                        a = int_to_binary(i, num_digits)
                        b = int_to_binary(j, num_digits)
                        c = int_to_binary(i-j, num_digits)
                        data.append(taskvec + a + b + c)
            x_train, x_test = train_test_split(data, test_size=0.1)
            train_data += x_train
            test_data += x_test
            
        if t == 'mul':
            data = []
            taskvec = [0] * len(operations)
            taskvec[operations.index(t)] = 1
            for i in range(max_number):
                for j in range(max_number):
                    if i * j < max_number:
                        a = int_to_binary(i, num_digits)
                        b = int_to_binary(j, num_digits)
                        c = int_to_binary(i*j, num_digits)
                        data.append(taskvec + a + b + c)
            x_train, x_test = train_test_split(data, test_size=0.1)
            train_data += x_train
            test_data += x_test
        if t == 'fir':
            data = []
            taskvec = [0] * len(operations)
            taskvec[operations.index(t)] = 1
            for i in range(max_number):
                for j in range(max_number):
                    a = int_to_binary(i, num_digits)
                    b = int_to_binary(j, num_digits)
                    c = int_to_binary(i, num_digits)
                    data.append(taskvec + a + b + c)
            x_train, x_test = train_test_split(data, test_size=0.1)
            train_data += x_train
            test_data += x_test
        if t == 'sec':
            data = []
            taskvec = [0] * len(operations)
            taskvec[operations.index(t)] = 1
            for i in range(max_number):
                for j in range(max_number):
                    a = int_to_binary(i, num_digits)
                    b = int_to_binary(j, num_digits)
                    c = int_to_binary(j, num_digits)
                    data.append(taskvec + a + b + c)
            x_train, x_test = train_test_split(data, test_size=0.1)
            train_data += x_train
            test_data += x_test
        if t == 'ide':
            data = []
            taskvec = [0] * len(operations)
            taskvec[operations.index(t)] = 1
            for i in range(max_number):
                for j in range(1):
                    a = int_to_binary(i, num_digits)
                    b = [0] * num_digits
                    c = int_to_binary(i, num_digits)
                    data.append(taskvec + a + b + c)
            x_train, x_test = train_test_split(data, test_size=0.1)
            train_data += x_train
            test_data += x_test
        if t == 'inc':
            data = []
            taskvec = [0] * len(operations)
            taskvec[operations.index(t)] = 1
            for i in range(max_number - 1):
                for j in range(1):
                    a = int_to_binary(i, num_digits)
                    b = [0] * num_digits
                    c = int_to_binary(i+1, num_digits)
                    data.append(taskvec + a + b + c)
            x_train, x_test = train_test_split(data, test_size=0.1)
            train_data += x_train
            test_data += x_test
    train_data = np.array(train_data)
    #np.random.shuffle(train_data)
    test_data = np.array(test_data)
    return {'train':{'x':train_data[:,:task_size+num_digits*2], 'y':train_data[:,task_size+num_digits*2:]},
            'test':{'x':test_data[:,:task_size+num_digits*2], 'y':test_data[:,task_size+num_digits*2:]}}

class ArithmeticDataset(Dataset):
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}

class Gumbel_Shared_MultiTurn(nn.Module):
    def __init__(self, params):
        super(Gumbel_Shared_MultiTurn, self).__init__()
        
        self.params = params
        
        self.signal_rnn = nn.GRUCell(params['signal_size'], params['signal_hidden_size'])
        self.signal_hidden = torch.zeros((params['num_agents'], params['batch_size'], params['signal_hidden_size']), requires_grad=True)

        self.state_rnn = rnn.GRUCell(params['state_input_size'], params['state_hidden_size'])
        self.state_hidden = torch.zeros((params['batch_size'], params['state_hidden_size']), requires_grad=True)

        self.combine_linear = nn.Linear(params['signal_hidden_size'] + params['state_hidden_size'] + params['goal_size'], params['combine_hidden_size'])
        self.speaker = nn.Linear(params['combine_hidden_size'], params['signal_size'])
        self.actor = nn.Linear(params['combine_hidden_size'], params['action_size'])

        self.dropout = nn.Dropout(p=drop)
        self.signals = []

        self.set_train()
        self.reset()
    
    def reset(self):
        self.signal_hidden = torch.zeros((params['num_agents'], params['batch_size'], params['signal_hidden_size']), requires_grad=True)
        self.state_hidden = torch.zeros((params['batch_size'], params['state_hidden_size']), requires_grad=True)
    def set_train(self):
        self.train()
        self.hard = False
    def set_eval(self):
        self.eval()
        self.hard = True

    def forward(self, task, state, signal):
        self.signal_hidden[speaker_id] = self.signal_rnn(signal, self.signal_hidden[speaker_id])
        self.state_hidden = self.state_rnn(state, self.state_hidden)
        
        combine = F.tanh(self.combined_linear(torch.cat((self.signal_hidden, self.state_hidden, task), dim=1)))
        signal_score = self.speaker(combine)
        action_score = self.actor(combine)

        signal_prob = F.gumbel_softmax(signal_score, hard=self.hard)
        self.signals.append(signal_prob)
        return action_score, signal_prob

class Sender_Receiver(nn.Module):
    def __init__(self, params):
        super(Sender_Receiver, self).__init__()
        
        self.params = params
        
        self.input_to_hidden = nn.Linear(params['input_size']-params['task_size'], params['sender_hidden_size'])
        self.sender_rnn = nn.GRUCell(params['vocab_size'], params['sender_hidden_size'])
        self.speaker = nn.Linear(params['sender_hidden_size'], params['vocab_size'])

        self.receiver_rnn = nn.GRU(params['vocab_size'], params['receiver_hidden_size'], 1)
        self.calc1 = nn.Linear(params['receiver_hidden_size'] + params['task_size'], params['num_digits'])
        
        self.batch = 1
        self.set_train()
        self.reset()
    
    def reset(self):
        pass
        #self.signal_hidden = torch.zeros((params['num_agents'], params['batch_size'], params['signal_hidden_size']), requires_grad=True)
        #self.state_hidden = torch.zeros((params['batch_size'], params['state_hidden_size']), requires_grad=True)
    def set_train(self):
        self.train()
        self.hard = True
        self.batch = self.params['batch_size']
    def set_eval(self):
        self.eval()
        self.hard = True
        self.batch = 1

    def forward(self, x, cuda):
        dev = torch.device("cpu")
        if cuda:
            dev = torch.device("cuda:0")
        batch_size = x.size()[0]
        sender_hidden = torch.tanh(self.input_to_hidden(x[:,self.params['task_size']:]))
        input_word = torch.zeros(batch_size, self.params['vocab_size'], device=dev)
        output_words = torch.zeros(10, batch_size, self.params['vocab_size'], device=dev)
        for i in range(10):
            sender_hidden = self.sender_rnn(input_word, sender_hidden)
            output_word = F.gumbel_softmax(self.speaker(sender_hidden), hard = self.hard)            
            output_words[i] = output_word
            input_word = output_word.detach()
        self.words = torch.argmax(output_words, dim=2)
        for b in range(batch_size):
            for i in range(1,10):
                if self.words[i-1,b] == 0:
                    output_words[i,b] = torch.zeros(self.params['vocab_size'], device = dev)
                    self.words[i,b] = 0
        #print(self.words)
        #print(output_words)
        #output_words = torch.tensor(output_words)
        #print(output_words.size())
        _, h_n = self.receiver_rnn(output_words, torch.zeros(1,batch_size, self.params['receiver_hidden_size'], device=dev))
        h_n = h_n.view(batch_size, self.params['receiver_hidden_size'])
        score = self.calc1(torch.cat((h_n,x[:,:self.params['task_size']]), dim=1))
        return score

def train():

    dtype = torch.float
    
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda:0")
    cpu = torch.device('cpu')

    #batch, input, hidden, output
    N, task_size, state_size, hidden_size, action_size = 1, 4, 6, 100, 6
    signal_size = 10
    tasks = ['ide', 'inc']
    params = {}
    params['batch_size'] = 1000
    params['num_digits'] = 7
    params['input_size'] = len(tasks) + params['num_digits'] * 2
    params['task_size']=len(tasks)
    params['sender_hidden_size'] = 256
    params['receiver_hidden_size'] = 256
    params['vocab_size'] = 30
    

    dataset = generate_data(params['num_digits'], operations=tasks)
    print('dataset size: train:', len(dataset['train']['x']), 'test', len(dataset['test']['x']))
    train_dataloader = DataLoader(ArithmeticDataset(dataset['train']), batch_size = params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(ArithmeticDataset(dataset['test']), batch_size = len(dataset['test']['x']), shuffle=False)
    
    model = Sender_Receiver(params).to(device)

    loss_function = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    epoch = 0
    if args.resume:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict']),
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    for epoch in range(epoch, epoch + 1000):
        loss_sum = 0
        for i_batch, sample_batched in enumerate(train_dataloader):
            #print(sample_batched)
            output_score = model(sample_batched['x'].to(device=device, dtype=dtype), args.cuda)
            #guess = output_score[:,0] * 16 + output_score[:,1] * 8 + output_score[:,2] * 4 + output_score[:,3] * 2 + output_score[:,4] * 1
            #target = sample_batched['y'][:,0] * 16 + sample_batched['y'][:,1] * 8 + sample_batched['y'][:,2] * 4 + sample_batched['y'][:,3] * 2 + sample_batched['y'][:,4] * 1
            #loss = mseloss(guess, target.to(dtype))
            loss = loss_function(output_score, sample_batched['y'].to(device=device, dtype=dtype))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        print("epoch {}, loss {}".format(epoch, loss_sum))

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0
                #train case
                for i_batch, sample_batched in enumerate(train_dataloader):
                    output_score = model(sample_batched['x'].to(device=device, dtype=dtype), args.cuda)
                    output_binary = torch.round(torch.sigmoid(output_score))
                    equality = torch.eq(output_binary, sample_batched['y'].to(device=device, dtype=dtype)).to(torch.float)
                    reward = torch.eq(torch.sum(equality, dim=1), torch.ones(equality.size()[0], device=device)*equality.size()[1]).to(device=device, dtype=dtype)
                    correct += int(torch.sum(reward).item())
                    total += reward.numel()
                    words = torch.transpose(model.words, 0, 1)
                    task_id = torch.argmax(sample_batched['x'][:,:len(tasks)], dim=1)
                print("train acc: {}/{} = {:.2%}".format(correct, total, correct/total))

                # test case
                for i_batch, sample_batched in enumerate(test_dataloader):
                    output_score = model(sample_batched['x'].to(device=device, dtype=dtype), args.cuda)
                    output_binary = torch.round(torch.sigmoid(output_score))
                    equality = torch.eq(output_binary, sample_batched['y'].to(device=device, dtype=dtype)).to(torch.float)
                    reward = torch.eq(torch.sum(equality, dim=1), torch.ones(equality.size()[0], device=device)*equality.size()[1]).to(device=device, dtype=dtype)
                    correct = int(torch.sum(reward).item())
                    total = reward.numel()
                    words = torch.transpose(model.words, 0, 1)
                    task_id = torch.argmax(sample_batched['x'][:,:len(tasks)], dim=1)
                    
                    add_score = 0
                    add_total = 1
                    sub_score = 0
                    sub_total = 1
                    mul_score = 0
                    mul_total = 1
                    with open('results.csv', 'w') as csvfile:
                        fieldnames = ['op', 'arg1', 'arg2', 'pred', 'ans','cor', 'msg1','msg2','msg3','msg4','msg5','msg6','msg7','msg8','msg9','msg10']
                        if epoch %1 == 0:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
                            writer.writeheader()
                        for i in range(len(task_id)):
                            op_name = tasks[task_id[i]]
                            arg1 = binary_to_int(sample_batched['x'][i,len(tasks):len(tasks)+params['num_digits']])
                            arg2 = binary_to_int(sample_batched['x'][i,len(tasks)+params['num_digits']:])
                            pred = int(binary_to_int(output_binary[i]))
                            ans = binary_to_int(sample_batched['y'][i])
                            
                            if epoch % 1 == 0:
                                #print("{}({:<2},{:<2}) out={:<2}, ans={:<2}  {:<2}  msg='{}'".format(op_name, arg1, arg2, pred, ans, pred==ans, words[i]))
                                writer.writerow({'op': op_name, 
                                                 'arg1': arg1.item(),
                                                 'arg2': arg2.item(),
                                                 'pred': pred,
                                                 'ans': ans.item(),
                                                 'cor': (pred==ans).item(),
                                                 'msg1': words[i][0].item(),'msg2': words[i][1].item(),'msg3': words[i][2].item(),'msg4': words[i][3].item(),'msg5': words[i][4].item(),
                                                 'msg6': words[i][5].item(),'msg7': words[i][6].item(),'msg8': words[i][7].item(),'msg9': words[i][8].item(),'msg10': words[i][9].item()  })
                            if task_id[i] == 0:
                                if pred == ans:
                                    add_score += 1
                                add_total += 1
                            if task_id[i] == 1:
                                if pred == ans:
                                    sub_score += 1
                                sub_total += 1
                            if task_id[i] == 2:
                                if pred == ans:
                                    mul_score += 1
                                mul_total += 1
                    #print("add: {}/{}={:.2%}, sub: {}/{}={:.2%}, mul: {}/{}={:.2%}".format(add_score, add_total, add_score/add_total, sub_score, sub_total, sub_score/sub_total, mul_score, mul_total, mul_score/mul_total))
                    print("test acc: {}/{} = {:.2%}".format(correct, total, correct/total))

                    #print(task)
                    #print(words)



                # _, signal_gumbel = model(task_test, input_test, torch.zeros(task_test.size()[0], signal_size))
                # action_score, _ = model(torch.zeros_like(task_test), input_test, signal_gumbel)
                # action_vector = torch.round(torch.sigmoid(action_score))
                
                # equality = torch.eq(action_vector, output_test).to(device=cpu, dtype=dtype)
                # reward = torch.eq(torch.sum(equality,dim=1), torch.ones(equality.size()[0])*action_size).to(torch.float)

                # for i in range(reward.numel()):
                #     print("task={}, (x={}, y={}), pred={}), signal='{}', correct={}".format(torch.argmax(task_test[i]).item(), to_str(input_test[i]), to_str(output_test[i]), 
                #                                                                             to_str(action_vector[i]), torch.argmax(model.signals[0][i]).item(), int(reward[i])))
                # print("t: {}, acc: {}/{} = {}".format(epoch, int(torch.sum(reward).item()), reward.numel(), torch.sum(reward).item() / reward.numel()))

                # model.reset()

                model.set_train()

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)

if __name__ == "__main__":
    train()