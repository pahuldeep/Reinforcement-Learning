import torch
import gymnasium as gym
import numpy as np
import cv2
from atari_model import Generator
from atari_wrapper import InputWrapper

LATENT_VECTOR_SIZE = 100
ENV_NAME = "Breakout-v4"
MAX_STEPS = 1000

def load_generator(model_path, output_shape, device="cpu"):
    generator = Generator(output_shape=output_shape)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator

def action_from_generator(generated_frame, action_space):
    action_value = np.mean(generated_frame)    

    action = int((action_value + 1) / 2 * (action_space.sample() - 1))  # Normalize and scale
    action = np.clip(action, 0, action_space.n - 1)  

    return action

def render_frames(env_frame, generated_frame):

    # Resize both frames for better visualization
    env_frame_resized = cv2.resize(env_frame, (400, 400))
    generated_frame_resized = cv2.resize(generated_frame, (400, 400))

    combined_frame = np.hstack((env_frame_resized, generated_frame_resized))

    cv2.imshow("Environment Frame (Left) | Generated Frame (Right)", combined_frame)

def run_atari_session(generator, env_name, max_steps, device="cpu"):
    env = gym.make(env_name, render_mode="rgb_array")  # Use 'rgb_array' for frame rendering
    obs, info = env.reset()
    total_reward = 0
    latent_vector_size = generator.pipe[0].in_channels

    for step in range(max_steps):
        # Generate latent vector and generated frame
        latent_vector = torch.FloatTensor(1, latent_vector_size, 1, 1).normal_(0, 1).to(device)
        generated_frame = generator(latent_vector).detach().cpu().numpy()
    
        generated_frame = np.clip(generated_frame, -1, 1)
        generated_frame = (generated_frame + 1) / 2.0  
        generated_frame = (generated_frame * 255).astype(np.uint8) 
        generated_frame = np.moveaxis(generated_frame[0], 0, -1)  # Reshape to (H, W, C)


        env_frame = env.render()
        render_frames(env_frame, generated_frame)
        
        action = action_from_generator(generated_frame, env.action_space)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}/{max_steps}: Reward = {reward}")
        if terminated or truncated:
            break

        # Add a small delay for real-time effect
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    env.close()
    cv2.destroyAllWindows()
    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the generator model
    env_temp = InputWrapper(gym.make(ENV_NAME))
    output_shape = env_temp.observation_space.shape
    env_temp.close()

    generator = load_generator("Environment\SynthAtari\weight\generator_7k.pth", output_shape=output_shape, device=device)
    
    run_atari_session(generator, ENV_NAME, MAX_STEPS, device=device)
