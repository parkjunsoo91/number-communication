import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.distributions import Categorical

import numpy as np

#simplest model: given a concat of 3 4d vectors
#input state -> hidden embedding -> vocab
#input vocab -> hidden embedding -> guess

class Sender(nn.Module):
    def __init__(self, params):
        super(Sender, self).__init__()
        for p in params:
            setattr(self, p, params[p])

        self.linear1 = nn.Linear(self.stateSize, self.hiddenSize)
        self.linear2 = nn.Linear(self.hiddenSize, self.signalSize)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        #self.linear1 = nn.Linear(self.stateSize, self.signalSize)
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        score = self.linear2(h_relu)
        #score = self.linear1(x)
        distr = F.softmax(score, dim=0)
        return distr

class Receiver(nn.Module):
    def __init__(self, params):
        super(Receiver, self).__init__()
        for p in params:
            setattr(self, p, params[p])

        self.linear1 = nn.Linear(self.signalSize, self.hiddenSize)
        self.linear2 = nn.Linear(self.hiddenSize, self.actionSize)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        score = self.linear2(h_relu)
        distr = F.softmax(score, dim=0)
        return distr

def main():
    #parameters:
    _i, _j, _k = 2,2,2
    params = {}
    params['stateSize'] = _i+_j+_k
    params['signalSize'] = 6 #testing the speaker only first
    params['actionSize'] = _i*_j*_k
    params['hiddenSize'] = 10


    sender = Sender(params)
    receiver = Receiver(params)

    optimizer = optim.Adam([{'params': sender.parameters()}, \
                            {'params': receiver.parameters()}], \
                             lr = 1e-2)

    dataset = MyDataset(_i,_j,_k)

    num_episode = 100000
    batch_size = 10
    log_probs_sender = []
    log_probs_receiver = []
    rewards = []
    running_reward = 0
    score = 0
    eps = np.finfo(np.float32).eps.item()

    sample_index = 0
    for i_episode in range(num_episode):
        x = []
        y = []
        #sample a testcase
        for i in range(batch_size):
            sample_index = np.random.choice(_i * _j * _k)
            #sample_index = 0 if sample_index == 7 else sample_index + 1
            datapair = dataset[sample_index]
            x_inst = datapair['x']
            y_inst = int(datapair['y'])
            x.append(x_inst)
            y.append(y_inst)
        x = np.array(x)
        x = torch.from_numpy(x).float()

        #forward pass sender
        distr = sender(x)
        m = Categorical(distr)
        signalId = m.sample()
        #record log probability 
        log_probs_sender.append(m.log_prob(signalId).unsqueeze(0))

        signal = torch.zeros(batch_size, params['signalSize'])
        for i in range(batch_size):
            signal[i][signalId[i]] = 1

        #forward pass receiver
        distr = receiver(signal)
        m = Categorical(distr)
        actionId = m.sample()
        #print(actionId)
        #record log probability
        log_probs_receiver.append(m.log_prob(actionId).unsqueeze(0))

        #record reward
        reward = []
        for i in range(batch_size):
            r = 1 if actionId[i] == y[i] else 0
            reward.append(r)
        rewards.append(reward)
        score += np.mean(rewards)

        
        #skip values(futureRewards) calculation
        if True:

            #TBD: express values in terms of rewards
            values = rewards
            values = torch.tensor(values).float()
            if i_episode % 1000 == 0:
                #print("values")
                #print(values)
                pass
            #TBD: normalize reward
            #values = (values - values.mean()) / (values.std() + eps)
            if i_episode % 1000 == 0:
                #print(values)
                pass
            #get gradient of loss
            policy_loss_sender = []
            policy_loss_receiver = []
            for log_prob, value in zip(log_probs_sender, values):
                policy_loss_sender.append(-log_prob * value)
            for log_prob, value in zip(log_probs_receiver, values):
                policy_loss_receiver.append(-log_prob * value)
            
            #print("policy loss len: ", len(policy_loss))
            optimizer.zero_grad()
            policy_loss_sender = torch.cat(policy_loss_sender).sum()
            policy_loss_sender.backward()
            policy_loss_receiver = torch.cat(policy_loss_receiver).sum()
            policy_loss_receiver.backward()
            #update policy
            optimizer.step()

            log_probs_sender = []
            log_probs_receiver = []
            rewards = []

        

        if i_episode % 1000 == 0:
            running_reward = running_reward * 0.9 + score/10 * 0.1
            score = 0
            #print(y, int(labelId))
            print('Episode {}\tRunning reward: {:.2f}'.format(
                i_episode, running_reward))
        if running_reward > 95:
            print("Solved! Running reward is now {}".format(running_reward))
            break
    score = 0

    for i in range (_i*_j*_k):
        sample_index = i
        datapair = dataset[sample_index]
        x = torch.from_numpy(datapair['x']).float()
        y = int(datapair['y'])
        distr = sender(x)
        signalId = int(torch.argmax(distr))
        signal = torch.zeros(params['signalSize'])
        signal[signalId] = 1
        distr = receiver(signal)
        actionPred = torch.argmax(distr)
        print (x, y, int(actionPred), signalId)
        if int(actionPred) == y:
            score += 1
    print('score ', score/(_i*_j*_k))

        
        


        



class MyDataset(Dataset):
    def __init__(self, _i, _j, _k):
        # 3 attributes, 2 instances -> 8 combinations
        self.frame = []
        self._i = _i
        self._j = _j
        self._k = _k
        for i in range(_i):
            for j in range(_j):
                for k in range(_k):
                    i_vec = np.zeros((_i))
                    i_vec[i] = 1
                    j_vec = np.zeros((_j))
                    j_vec[j] = 1
                    k_vec = np.zeros((_k))
                    k_vec[k] = 1
                    
                    label = np.array([i*_j*_k + j*_k + k])
                    item = np.concatenate((i_vec, j_vec, k_vec, label))
                    self.frame.append(item)


    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        sample = {'x': self.frame[idx][0:self._i + self._j + self._k], 'y': self.frame[idx][-1]}
        
        return sample







if __name__ == "__main__":
    main()







