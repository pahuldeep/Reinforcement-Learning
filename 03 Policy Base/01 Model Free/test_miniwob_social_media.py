import torch
import gymnasium as gym
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load the Hugging Face model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)  # NUM_ACTIONS = 10

# Load the checkpoint
checkpoint_path = "miniwob_social_media_checkpoint.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def test_agent(env, net):
    observation, _ = env.reset()
    total_reward = 0
    
    sm = torch.nn.Softmax(dim=1)
    while True:
        # Tokenize the observation
        inputs = tokenizer(observation, return_tensors="pt", truncation=True, padding=True)
        action_logits = net(**inputs).logits
        action_probabilities = sm(action_logits)
        action = torch.argmax(action_probabilities).item()
        
        # Perform the action
        observation, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        
        if done or trunc:
            break
    
    print(f"Total reward in test episode: {total_reward}")
    env.close()

if __name__ == "__main__":
    # Set up the MiniWoB Social-Media environment
    env = gym.make("miniwob/social-media-v1")
    
    # Test the agent
    test_agent(env, model)