import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.distributions import Categorical

def preprocess(state):
    state = np.array(state)
    state = np.where(state > 0, np.log2(state), 0)
    state = torch.FloatTensor(state).unsqueeze(0)
    state = state.unsqueeze(0)
    return state

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.affine = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(288, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128,action_dim)
        self.critic = nn.Linear(128,1)

    def forward(self, x):
        x = self.affine(x)
        action_probs = F.softmax(self.actor(x), dim = 1)
        expected_return = self.critic(x)

        return action_probs, expected_return
    
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.K_epochs = 4
        self.clamp = 0.2
        self.gamma = 0.995
        self.lr = 0.001
        self.policy = PPO(state_dim, action_dim)
        self.old_policy = PPO(state_dim, action_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.lr)
        self.MseLoss = nn.MSELoss()

    def act(self, state, memory):
        state = preprocess(state)

        with torch.no_grad():
            probs, _ = self.old_policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        logprobs = dist.log_prob(action).detach().cpu()
        memory.states.append(state.detach().cpu())
        memory.actions.append(action.detach().cpu())
        memory.logprobs.append(logprobs.detach().cpu())
        return action.item(), 
    def update(self, memory):
        old_states = torch.FloatTensor(np.array(memory.states)).squeeze(1)
        old_actions = torch.LongTensor(np.array(memory.actions))
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).unsqueeze(0)
        
        rewards = []
        reward_ebatkakoi = 0
        for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
            if done:
                reward_ebatkakoi = 0
            reward_ebatkakoi = reward + (self.gamma * reward_ebatkakoi) 
            rewards.insert(0, reward_ebatkakoi)
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)
        for _ in range(self.K_epochs):
            probs, predict = self.policy(old_states)
            predict = torch.squeeze(predict)
            dist = Categorical(probs)
            dist_entropy = dist.entropy()
            logprobs = dist.log_prob(old_actions)

            advantages = rewards - predict.detach()
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = advantages*ratios
            surr2 = torch.clamp(ratios, 1-self.clamp, 1+self.clamp)*advantages
            loss = -torch.min(surr1, surr2) + self.MseLoss(predict, rewards) - 0.01*dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.old_policy.load_state_dict(self.policy.state_dict())
    
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
    def push(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:], self.rewards[:], self.dones[:]

def checkpoint(policy, optimizer, episode):
        file = f"checkpoint_{episode}.tar"
        folder = "checkpoint"
        path = os.path.join(folder, file)
        model = {
            "policy": policy.state_dict(),
            "optim": optimizer.state_dict()
        }
        if not os.path.exists("checkpoint"):
            os.makedirs("checkpoint")
        torch.save(model, path)
        print("SAVE IS DONE")
def load_checkpoint(agent, episode):
        file = f"checkpoint_{episode}.tar"
        folder = "checkpoint"
        path = os.path.join(folder, file)
        model = torch.load(path)
        agent.policy.load_state_dict(model["policy"])
        agent.optimizer.load_state_dict(model["optim"])
        agent.old_policy.load_state_dict(agent.policy.state_dict())
        print("load is done")
        return agent