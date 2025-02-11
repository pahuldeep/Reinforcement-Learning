import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import gymnasium as gym
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
from typing import List

# Hyperparameters
HIDDEN_SIZE = 768
BATCH_SIZE = 16
LR = 0.0001
NUM_ACTIONS = 10  # Example: Number of discrete actions in MiniWoB Social-Media

# Load a Hugging Face pre-trained model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_ACTIONS)

@dataclass
class EpisodeSteps:
    observe: str
    action: int

@dataclass
class Episode:
    reward: float
    steps: List[EpisodeSteps]

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()

def adaptive_percentile(iter_no, start=90, end=50, decay=0.01):
    return max(end, start - decay * iter_no * (start - end))

def iterate_batch(env, net, size):
    batch = []
    episode_reward = 0
    episode_steps = []
    observation, _ = env.reset()
    
    sm = nn.Softmax(dim=1)
    while True:
        inputs = tokenizer(observation, return_tensors="pt", truncation=True, padding=True)
        action_logits = net(**inputs).logits
        action_probabilities = sm(action_logits)
        action_value = action_probabilities.detach().numpy()[0]
        
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
    rewards = [episode.reward for episode in batch]
    reward_bound = np.percentile(rewards, percent)
    reward_mean = np.mean(rewards)
    
    train_observations = []
    train_actions = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_observations.extend([step.observe for step in episode.steps])
        train_actions.extend([step.action for step in episode.steps])
    
    inputs = tokenizer(train_observations, return_tensors="pt", truncation=True, padding=True)
    train_action_value = torch.LongTensor(train_actions)
    
    return inputs, train_action_value, reward_bound, reward_mean

if __name__ == "__main__":
    # Set up the MiniWoB Social-Media environment
    env = gym.make("miniwob/social-media-v1")
    
    # Define the optimizer and loss function
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    writer = SummaryWriter(comment="-miniwob-social-media")
    
    for iter_no, batch in enumerate(iterate_batch(env, model, BATCH_SIZE)):
        current_percentile = adaptive_percentile(iter_no)
        obs_inputs, acts_v, reward_b, reward_m = filter_batch(batch, current_percentile)
        
        optimizer.zero_grad()
        action_scores_v = model(**obs_inputs).logits
        probs_v = nn.Softmax(dim=1)(action_scores_v)
        entropy = compute_entropy(probs_v)
        
        weights = torch.FloatTensor([episode.reward for episode in batch])
        weights = weights / weights.sum()
        loss_v = (objective(action_scores_v, acts_v) * weights).mean() - 0.001 * entropy
        
        loss_v.backward()
        optimizer.step()
        scheduler.step()
        
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        
        if reward_m > 50:
            print("Solved!")
            break
    
    # Save the model checkpoint
    checkpoint_path = "miniwob_social_media_checkpoint.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'reward_mean': reward_m,
        'iteration': iter_no
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    writer.close()