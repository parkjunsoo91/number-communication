
import argparse
import gym
import numpy as np
from itertools import count
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Bernoulli

class Policy(nn.Module):
    def __init__(self, params):
        super(Policy, self).__init__()
        #for p in params:
        #    setattr(self, p, params[p])
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.batch_size = params['batch_size']
        self.output_size = params['output_size']
        
        self.rnn_cell = nn.RNNCell(self.input_size, self.hidden_size)
        
        self.hx = torch.zeros(self.batch_size, self.hidden_size)
        self.hidden_states = []

        self.linear = nn.Linear(self.hidden_size, self.output_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, input):
        #input of rnn(seq_len, batch, input_size)
        #input of rnncell(batch, input_size)
        #print('input\t', input)
        #print('hx\t', self.hx)
        #print('rnncell\t', self.rnn_cell.weight_ih, self.rnn_cell.weight_hh)
        self.hx = self.rnn_cell(input, self.hx)
        self.hidden_states.append(self.hx)

        scores = self.linear(self.hx)
        #print('hx\t', self.hx)
        #print('score\t', scores)
        # cursor_probs = F.softmax(scores[:2])
        # write_probs = F.softmax(scores[2:])

        # return cursor_probs, write_probs
        probs =  F.softmax(scores, dim=1)
        #print('probs\t', probs.shape, probs)
        return probs

    def reset(self):
        self.hx = torch.zeros(self.batch_size, self.hidden_size)
        del self.rewards[:]
        del self.saved_log_probs[:]


def main():
    #environment settings
    env = gym.make('Copy-v0')
    #env = gym.make('DuplicatedInput-v0')
    render_period = 100


    #model settings
    params = {'input_size': 6, 
              'output_size': 12,
              'hidden_size': 10,
              'batch_size': 1
              }    
    policy = Policy(params)

    #training settings
    num_episode = 1000000
    learning_rate = 0.01
    gamma = 0.99
    running_reward = 0

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    eps = np.finfo(np.float32).eps.item()

    for i_episode in range(num_episode):

        observation = env.reset()
        if i_episode % render_period == 0:
            os.system('cls')
            env.render()
        
        for t_step in count():
            #process observation
            observed_char = torch.zeros(6)
            observed_char[observation] = 1.0

            action_probs = policy(observed_char.unsqueeze(0))
            m = Categorical(action_probs)
            action_vector = m.sample()

            #policy.saved_log_probs.append(m.log_prob(action).unsqueeze(0))
            policy.saved_log_probs.append(m.log_prob(action_vector))
            action_id = int(action_vector.item())

            direction = action_id // 6
            writebool = 0 if action_id % 6 == 5 else 1
            character = 0 if action_id % 6 == 5 else action_id % 6
            action = (direction, writebool, character)
            observation, reward, done, info = env.step(action)
            
            if i_episode % render_period == 0:
                #time.sleep(2)
                os.system('cls')
                env.render()
                print(observation, reward, done, info)
            policy.rewards.append(reward - 0.1) #crucial to penalize steps
            
            if done:
                break
        
        running_reward = running_reward * 0.99 + t_step * 0.01    

        #should update policy
        R = 0
        futureRewards = []
        policy_loss = []

        for r in policy.rewards[::-1]:
            R = r + gamma * R
            futureRewards.insert(0, R)
        futureRewards = torch.tensor(futureRewards)

        if futureRewards.size()[0] > 1:
            futureRewards = (futureRewards - futureRewards.mean()) / (futureRewards.std() + eps)
        #print('futureRewards\t', futureRewards)
        for log_prob, reward in zip(policy.saved_log_probs, futureRewards):
            policy_loss.append(-log_prob * reward)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        policy.reset()

        if i_episode % 100 == 0:
            print('Episode {}\tLast reward: {}\tRunning reward: {:.2f}'.format(
                i_episode, t_step, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {}".format(running_reward))
            break

if __name__ == '__main__':
    main()
