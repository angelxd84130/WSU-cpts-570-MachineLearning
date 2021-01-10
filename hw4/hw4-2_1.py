#!/usr/bin/python
import gym
import time
import matplotlib.pyplot as plt
import numpy as np
env = gym.make('CartPole-v1')
#result = np.array([], dtype=np.float)
result = []
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sys import meta_path


def main(result):
    num_episodes = 10000
    for _ in range(num_episodes):
        state = env.reset() # reset() resets the environment
        episode_reward = 0

        for t in range(1, 10000): # no of steps
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            env.render() # show
            policy.rewards.append(reward)
            episode_reward += reward
            if done:
                print("episode ended!")
                break

        finish_episode_and_update()
        #print("_ ", _)
        #result = np.insert(result, _, episode_reward)
        result.append(episode_reward)
        print(result)



class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 24)
        self.affine2 = nn.Linear(24, 36)
        self.affine3 = nn.Linear(36, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.affine2(x)
        x = self.affine3(x)
        action_scores = F.relu(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

from torch.distributions import Categorical
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state) # FORWARD PASS
    m = Categorical(probs) # we are sampling from a distribution to add some exploration to the policy's behavior.
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

gamma = 0.99 # discount factor
def finish_episode_and_update():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    print("R", R)

    #print(returns)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward() # backward pass
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

main(result)

plt.title("RL")
plt.xlabel("episode number")
plt.ylabel("total reward")
plt.plot([i for i in range(10000)], result)
plt.show()

env.close()