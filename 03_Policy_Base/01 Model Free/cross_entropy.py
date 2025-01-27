import typing as tt 
from dataclasses import dataclass

import torch 
import torch.nn as nn 
import torch.optim as optim 

from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
import gymnasium as gym

HIDDEN_SIZE = 128 
BATCH_SIZE = 16 
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, observation_size, hidden_size, num_action):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_action),
        )
    def forward(self, x):
        return self.net(x)
    
@dataclass
class EpisodeSteps:
    observe: np.ndarray
    action: int

@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeSteps]

    
def iterate_batch(env, net, size):
    batch = []
    episode_reward = 0
    episode_steps = []
    observation, _ = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        observe = torch.tensor(observation)
        action_probability = sm(net(observe.unsqueeze(0)))
        action_value = action_probability.data.numpy()[0]
        
        action = np.random.choice(len(action_value), p=action_value)
        next_observation, reward, is_done, is_trunc, _ = env.step(action) 

        episode_reward += reward
        step = EpisodeSteps(observe=observation, action=action)
        episode_steps.append(step)

        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []

            next_observation, _ = env.reset()
            if len(batch) == size:
                yield batch
                batch = []

            observation = next_observation

def filter_batch(batch, percent):
    rewards = list(map(lambda x: x.reward, batch))
    reward_bound = np.percentile(rewards, percent) 
    reward_mean = np.mean(rewards)

    train_observations = []
    train_actions = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue

        train_observations.extend(map(lambda step: step.observe, episode.steps))
        train_actions.extend(map(lambda step: step.action, episode.steps))

    train_observation_value = torch.FloatTensor(np.vstack(train_observations))
    train_action_value = torch.LongTensor(train_actions)

    return train_observation_value, train_action_value, reward_bound, reward_mean


if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = int(env.action_space.n)

    net = Net(obs_size, HIDDEN_SIZE, n_actions)

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batch(env, net, BATCH_SIZE)):
        
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        
        action_scores_v = net(obs_v)
        
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))

        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 30:
            print("Solved!")
            break

    writer.close()
    
         
