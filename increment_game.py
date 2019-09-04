import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
parser.add_argument('--cuda', action="store_true", help='use gpu')
args = parser.parse_args()
PATH = "increment_game.pt"

def generate_data(size, tasks=['id', 'neg', 'incr', 'decr']):
    def int_to_binary(d):
        b_str = '{0:06b}'.format(d)
        return [int(c) for c in b_str]
    #task:input:output
    task_size = 4
    num_digits = size
    max_number = pow(2,num_digits)
    #task = ['id', 'neg', 'incr', 'decr']
    #task = ['id', 'decr']
    train_data = []
    test_data = []
    for t in tasks:
        if t == 'id':
            taskvec = [1,0,0,0]
            input_ints = [i for i in range(0,max_number)]
            input_vecs = [int_to_binary(d) for d in input_ints]
            for i in range(len(input_vecs)):
                cat_vec = taskvec + input_vecs[i] + input_vecs[i]
                if i in [27,28,29,30,31]:
                    test_data.append(cat_vec)
                else:
                    train_data.append(cat_vec)
        if t == 'incr':
            taskvec = [0,1,0,0]
            input_ints = [i for i in range(0,max_number-1)]
            output_ints = [(i+1) % max_number for i in input_ints]
            input_vecs = [int_to_binary(d) for d in input_ints]
            output_vecs = [int_to_binary(d) for d in output_ints]
            for i in range(len(input_vecs)):
                cat_vec = taskvec + input_vecs[i] + output_vecs[i]
                if i in [26,27,28,29,30]:
                    test_data.append(cat_vec)
                else: 
                    train_data.append(cat_vec)
        if t == 'decr':
            taskvec = [0,0,1,0]
            input_ints = [i for i in range(1,max_number)]
            output_ints = [(i-1) % max_number for i in input_ints]
            input_vecs = [int_to_binary(d) for d in input_ints]
            output_vecs = [int_to_binary(d) for d in output_ints]
            for i in range(len(input_vecs)):
                cat_vec = taskvec + input_vecs[i] + output_vecs[i]
                if i in [27,28,29,30,31]:
                    test_data.append(cat_vec)
                else:
                    train_data.append(cat_vec)
        if t == 'neg':
            taskvec = [0,0,0,1]
            input_ints = [i for i in range(0,max_number)]
            input_vecs = [int_to_binary(d) for d in input_ints]
            output_vecs = [[1-j for j in i] for i in input_vecs]
            for i in range(len(input_vecs)):
                cat_vec = taskvec + input_vecs[i] + output_vecs[i]
                if i in [27,28,29,30,31]:
                    test_data.append(cat_vec)
                else:
                    train_data.append(cat_vec)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    return {'train':{'task':train_data[:,:task_size], 'input':train_data[:,task_size:task_size+num_digits], 'output':train_data[:,task_size+num_digits:]},
            'test':{'task':test_data[:,:task_size], 'input':test_data[:,task_size:task_size+num_digits], 'output':test_data[:,task_size+num_digits:]}}




class SingleLayer(nn.Module):
    def __init__(self, D_in, D_out):
        super(SingleLayer, self).__init__()
        self.linear1 = torch.nn.Linear(D_in,D_out)
    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred

class TwoLayer(nn.Module):
    def __init__(self, D_in, H, D_out, drop=0.5):
        super(TwoLayer, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(p=drop)
    def forward(self, x):
        hidden = F.tanh(self.linear1(x))
        drop = self.dropout(hidden)
        y_pred = self.linear2(drop)
        return y_pred
class ThreeLayer(nn.Module):
    def __init__(self, D_in, H, D_out, drop=0.2):
        super(ThreeLayer, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, H)
        self.linear3 = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(p=drop)
    def forward(self, x):
        hidden = F.tanh(self.linear1(x))
        drop = self.dropout(hidden)
        drop2 = self.dropout(F.tanh(self.linear2(drop)))
        y_pred = self.linear3(drop2)
        return y_pred


def train():

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    #batch, input, hidden, output
    N, task_size, state_size, hidden_size, action_size = 1, 4, 5, 200, 5
    signal_size = task_size

    dataset = generate_data(['id', 'neg', 'incr', 'decr'])
    task_train = torch.tensor(dataset['train']['task'], dtype=dtype, device=device)
    input_train = torch.tensor(dataset['train']['input'], dtype=dtype, device=device)
    output_train = torch.tensor(dataset['train']['output'], dtype=dtype, device=device)

    task_test = torch.tensor(dataset['test']['task'], dtype=dtype, device=device)
    input_test = torch.tensor(dataset['test']['input'], dtype=dtype, device=device)
    output_test = torch.tensor(dataset['test']['output'], dtype=dtype, device=device)

    print(task_train.size(), input_train.size(), output_train.size())

    #sender = SingleLayer(task_size + state_size, signal_size)
    #sender = TwoLayer(task_size + state_size, hidden_size, signal_size)
    sender = ThreeLayer(task_size + state_size, hidden_size, signal_size)
    #receiver = SingleLayer(signal_size + state_size, action_size)
    #receiver = TwoLayer(signal_size + state_size, hidden_size, action_size)
    receiver = ThreeLayer(signal_size + state_size, hidden_size, action_size)

    loss_function = nn.MultiLabelSoftMarginLoss(reduction='none')
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

    for episode in range(epoch, 1000001):
        sender_input = torch.cat((task_train, input_train), dim=1)
        signal_score = sender(sender_input)
        signal_probs = F.softmax(signal_score, dim=1)
        distr_signal = Categorical(signal_probs)
        signal = distr_signal.sample()
        signal_vector = torch.eye(signal_size)[signal]

        receiver_input = torch.cat((signal_vector, input_train), dim=1)
        action_score = receiver(receiver_input)

        loss = loss_function(action_score, output_train)

        action_probs = torch.sigmoid(action_score)
        distr_action = Bernoulli(action_probs)
        action = distr_action.sample()
        equality = torch.eq(action, output_train).to(torch.float)
        reward = torch.eq(torch.sum(equality,dim=1), torch.ones(equality.size()[0])*5).to(torch.float)
        reward = (reward - reward.mean())
        sender_loss = -distr_signal.log_prob(signal) * reward
        #receiver_loss = -torch.sum(distr_action.log_prob(action), dim=1) * reward

        #sender_loss = -distr_signal.log_prob(signal) * torch.sum(loss, dim=1)
        receiver_loss = torch.mean(loss)

        sender.zero_grad()
        receiver.zero_grad()

        # sender_loss.sum().backward()
        # receiver_loss.sum().backward()
        
        sender_loss.sum().backward()
        receiver_loss.backward()

        optimizer.step()

        if episode % 1000 == 0:
            print(sender.linear3.weight.data)
            print(receiver.linear3.weight.data)
            with torch.no_grad():
                sender.eval()
                receiver.eval()
                sender_input = torch.cat((task_train, input_train), dim=1)
                score_signal = sender(sender_input)
                signal = torch.argmax(score_signal, dim=1)
                signal_vector = torch.eye(signal_size)[signal]
                
                receiver_input = torch.cat((signal_vector, input_train), dim=1)
                score_action = receiver(receiver_input)
                action_probs = torch.sigmoid(score_action)
                action_vector = torch.round(action_probs)
                
                equality = torch.eq(action_vector, output_train).to(torch.float)
                reward = torch.eq(torch.sum(equality,dim=1), torch.ones(equality.size()[0])*5).to(torch.float)

                for i in range(reward.numel()):
                    print("{}({},{}), signal='{}', correct={}".format(task_train[i], output_train[i], action_vector[i], signal[i].item(), int(reward[i])))
                print("t: {}, acc: {}/{} = {}".format(episode, int(torch.sum(reward).item()), reward.numel(), torch.sum(reward).item() / reward.numel()))

                sender_input = torch.cat((task_test, input_test), dim=1)
                score_signal = sender(sender_input)
                signal = torch.argmax(score_signal, dim=1)
                signal_vector = torch.eye(signal_size)[signal]
                
                receiver_input = torch.cat((signal_vector, input_test), dim=1)
                score_action = receiver(receiver_input)
                action_probs = torch.sigmoid(score_action)
                action_vector = torch.round(action_probs)
                
                equality = torch.eq(action_vector, output_test).to(torch.float)
                reward = torch.eq(torch.sum(equality,dim=1), torch.ones(equality.size()[0])*5).to(torch.float)

                for i in range(reward.numel()):
                    print("{}({},{}), signal='{}', correct={}".format(task_test[i],output_test[i], action_vector[i], signal[i].item(), int(reward[i])))
                print("t: {}, acc: {}/{} = {}".format(episode, int(torch.sum(reward).item()), reward.numel(), torch.sum(reward).item() / reward.numel()))


                sender.train()
                receiver.train()

            torch.save({'epoch': episode,
                        'sender_state_dict': sender.state_dict(),
                        'receiver_state_dict': receiver.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)

class Gumbel_Linear(nn.Module):
    def __init__(self, task_size, state_size, action_size, hidden_size, signal_size, drop=0.3):
        super(Gumbel_Linear, self).__init__()
        self.linear1 = nn.Linear(task_size + state_size, signal_size)
        self.linear3 = nn.Linear(signal_size + state_size, action_size)
        self.dropout = nn.Dropout(p=drop)
        self.signal = None
    def forward(self, task, state):
        signal = F.gumbel_softmax(self.linear1(torch.cat((task, state), dim=1)), hard=True)
        self.signal = signal
        y_pred = self.linear3(torch.cat((signal, state), dim=1))
        return y_pred

class Gumbel_Net(nn.Module):
    def __init__(self, task_size, state_size, action_size, hidden_size, signal_size, drop=0.1):
        super(Gumbel_Net, self).__init__()
        self.linear1 = nn.Linear(task_size + state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, signal_size)
        self.linear3 = nn.Linear(signal_size + state_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(p=drop)
        self.signal = None
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)
        nn.init.xavier_normal_(self.linear4.weight)
        self.hard = False
        
    def forward(self, task, state, action):
        h1 = self.dropout(F.tanh(self.linear1(torch.cat((task, state), dim=1))))
        signal = F.gumbel_softmax(self.linear2(h1), hard=self.hard)
        self.signal = signal
        h2 = self.dropout(F.tanh(self.linear3(torch.cat((signal, state), dim=1))))
        y_pred = self.linear4(h2)
        
        return y_pred
    def set_train(self):
        self.train()
        self.hard = False
    def set_eval(self):
        self.eval()
        self.hard = True

def to_str(tensor):
    #binary tensor to binary string
    clist = [int(e) for e in tensor]
    s = ""
    for c in clist:
        s += str(c)
    return s


def train_gumbel():
    dtype = torch.float
    
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda:0")
    cpu = torch.device('cpu')

    #batch, input, hidden, output
    N, task_size, state_size, hidden_size, action_size = 1, 4, 5, 200, 5
    signal_size = 10

    dataset = generate_data(size=state_size, tasks=['id','neg', 'incr', 'decr'])
    task_train = torch.tensor(dataset['train']['task'], dtype=dtype, device=device)
    input_train = torch.tensor(dataset['train']['input'], dtype=dtype, device=device)
    output_train = torch.tensor(dataset['train']['output'], dtype=dtype, device=device)

    task_test = torch.tensor(dataset['test']['task'], dtype=dtype, device=device)
    input_test = torch.tensor(dataset['test']['input'], dtype=dtype, device=device)
    output_test = torch.tensor(dataset['test']['output'], dtype=dtype, device=device)

    print(task_train.size(), input_train.size(), output_train.size())

    model = Gumbel_Net(task_size, state_size, action_size, hidden_size, signal_size).to(device)

    loss_function = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    epoch = 0
    if args.resume:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict']),
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    for episode in range(epoch, epoch + 100001):
        pred = model(task_train, input_train, output_train)
        # pred = torch.reshape(pred, (-1, 1))
        # sig = torch.sigmoid(pred)
        # opp = torch.ones_like(sig)
        # negsig = opp - sig
        # probs = torch.cat((sig, negsig), dim=1)
        # logprobs = torch.log(probs)
        # gumbel_output = nn.functional.gumbel_softmax(logprobs, hard=True)
        # print(gumbel_output)
        # return
        
        loss = loss_function(pred, output_train)
        signal_count = torch.sum(model.signal, dim=0)
        print(signal_count)
        symbol_prob = F.log_softmax(signal_count)
        symbol_prob
        return
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 1000 == 0:
            with torch.no_grad():
                model.eval()

                pred = model(task_train, input_train, output_train)
                loss = loss_function(pred, output_train)
                action_vector = torch.round(torch.sigmoid(pred))

                equality = torch.eq(action_vector, output_train).to(device=cpu, dtype=dtype)
                reward = torch.eq(torch.sum(equality,dim=1), torch.ones(equality.size()[0])*action_size).to(device=cpu, dtype=dtype)

                for i in range(reward.numel()):
                    print("task={}, (x={}, y={}), pred={}), signal='{}', correct={}".format(torch.argmax(task_train[i]).item(), to_str(input_train[i]), to_str(output_train[i]), 
                                                                                            to_str(action_vector[i]), torch.argmax(model.signal[i]).item(), int(reward[i])))
                print("t: {}, acc: {}/{} = {}".format(episode, int(torch.sum(reward).item()), reward.numel(), torch.sum(reward).item() / reward.numel()))
                print("loss = {}".format(loss))
                pred = model(task_test, input_test, output_test)
                action_vector = torch.round(torch.sigmoid(pred))
                
                equality = torch.eq(action_vector, output_test).to(device=cpu, dtype=dtype)
                reward = torch.eq(torch.sum(equality,dim=1), torch.ones(equality.size()[0])*action_size).to(torch.float)

                for i in range(reward.numel()):
                    print("task={}, (x={}, y={}), pred={}), signal='{}', correct={}".format(torch.argmax(task_test[i]).item(), to_str(input_test[i]), to_str(output_test[i]), 
                                                                                            to_str(action_vector[i]), torch.argmax(model.signal[i]).item(), int(reward[i])))
                print("t: {}, acc: {}/{} = {}".format(episode, int(torch.sum(reward).item()), reward.numel(), torch.sum(reward).item() / reward.numel()))


                model.set_train()

            torch.save({'epoch': episode,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)


class Gumbel_Shared(nn.Module):
    def __init__(self, task_size, state_size, action_size, hidden_size, signal_size, drop=0.1):
        super(Gumbel_Shared, self).__init__()
        self.state_encoder = nn.Linear(state_size, hidden_size)
        self.signal_encoder = nn.Linear(signal_size, hidden_size)
        self.combined_fc = nn.Linear(hidden_size + hidden_size + task_size, hidden_size)
        self.signal_module = nn.Linear(hidden_size, signal_size)
        self.action_module = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(p=drop)
        self.signals = []
        self.hard = False

    def forward(self, task, state, signal):
        state_vector = F.tanh(self.dropout(self.state_encoder(state)))
        signal_vector = F.tanh(self.dropout(self.signal_encoder(signal)))
        combined_vector = F.tanh(self.combined_fc(torch.cat((task, state_vector, signal_vector), dim=1)))
        signal_score = self.signal_module(combined_vector)
        action_score = self.action_module(combined_vector)
        signal_prob = F.gumbel_softmax(signal_score, hard=self.hard)
        self.signals.append(signal_prob)
        return action_score, signal_prob
    def reset(self):
        self.signals = []
    def set_train(self):
        self.train()
        self.hard = False
    def set_eval(self):
        self.eval()
        self.hard = True

def train_gumbel_shared():
    dtype = torch.float
    
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda:0")
    cpu = torch.device('cpu')

    #batch, input, hidden, output
    N, task_size, state_size, hidden_size, action_size = 1, 4, 6, 100, 6
    signal_size = 10

    dataset = generate_data(size=state_size, tasks=['id','neg', 'incr', 'decr'])
    task_train = torch.tensor(dataset['train']['task'], dtype=dtype, device=device)
    input_train = torch.tensor(dataset['train']['input'], dtype=dtype, device=device)
    output_train = torch.tensor(dataset['train']['output'], dtype=dtype, device=device)

    task_test = torch.tensor(dataset['test']['task'], dtype=dtype, device=device)
    input_test = torch.tensor(dataset['test']['input'], dtype=dtype, device=device)
    output_test = torch.tensor(dataset['test']['output'], dtype=dtype, device=device)

    print(task_train.size(), input_train.size(), output_train.size())

    model = Gumbel_Shared(task_size, state_size, action_size, hidden_size, signal_size).to(device)

    loss_function = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    epoch = 0
    if args.resume:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict']),
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    for episode in range(epoch, epoch + 100001):
        zzz, signal_gumbel = model(task_train, input_train, torch.zeros(task_train.size()[0], signal_size))
        action_score, _ = model(torch.zeros_like(task_train), input_train, signal_gumbel)
        
        loss = loss_function(action_score, output_train)
        #signal_count = torch.sum(model.signal, dim=0)
        #print(signal_count)
        #symbol_prob = F.log_softmax(signal_count)


        model.zero_grad()
        loss.backward()
        optimizer.step()
        model.reset()

        if episode % 1000 == 0:
            with torch.no_grad():
                model.eval()
                # train case
                _, signal_gumbel = model(task_train, input_train, torch.zeros(task_train.size()[0], signal_size))
                action_score, _ = model(torch.zeros_like(task_train), input_train, signal_gumbel)
                loss = loss_function(action_score, output_train)
                action_vector = torch.round(torch.sigmoid(action_score))

                equality = torch.eq(action_vector, output_train).to(device=cpu, dtype=dtype)
                reward = torch.eq(torch.sum(equality,dim=1), torch.ones(equality.size()[0])*action_size).to(device=cpu, dtype=dtype)
                for i in range(reward.numel()):
                    print("task={}, (x={}, y={}), pred={}), signal='{}', correct={}".format(torch.argmax(task_train[i]).item(), to_str(input_train[i]), to_str(output_train[i]), 
                                                                                            to_str(action_vector[i]), torch.argmax(model.signals[0][i]).item(), int(reward[i])))
                print("t: {}, acc: {}/{} = {}".format(episode, int(torch.sum(reward).item()), reward.numel(), torch.sum(reward).item() / reward.numel()))
                print("loss = {}".format(loss))
                model.reset()

                # test case
                _, signal_gumbel = model(task_test, input_test, torch.zeros(task_test.size()[0], signal_size))
                action_score, _ = model(torch.zeros_like(task_test), input_test, signal_gumbel)
                action_vector = torch.round(torch.sigmoid(action_score))
                
                equality = torch.eq(action_vector, output_test).to(device=cpu, dtype=dtype)
                reward = torch.eq(torch.sum(equality,dim=1), torch.ones(equality.size()[0])*action_size).to(torch.float)

                for i in range(reward.numel()):
                    print("task={}, (x={}, y={}), pred={}), signal='{}', correct={}".format(torch.argmax(task_test[i]).item(), to_str(input_test[i]), to_str(output_test[i]), 
                                                                                            to_str(action_vector[i]), torch.argmax(model.signals[0][i]).item(), int(reward[i])))
                print("t: {}, acc: {}/{} = {}".format(episode, int(torch.sum(reward).item()), reward.numel(), torch.sum(reward).item() / reward.numel()))

                model.reset()

                model.set_train()

            torch.save({'epoch': episode,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)

if __name__ == "__main__":
    #train()
    #train_gumbel()
    train_gumbel_shared()
