import time
import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes

# Register MiniWoB environments with Gymnasium
mini_wob_envs = gym.register_envs(miniwob)

# Create the MiniWoB environment
env = gym.make('miniwob/click-button-v1', render_mode='human')

# Reset the environment
observation, info = env.reset()

while True:
    # Display the instruction to the user
    print("Instruction:", observation["utterance"])
    time.sleep(2)  # Short delay for readability

    # Extract the target button text from the instruction
    target_text = observation["utterance"].split()[-1]
    print("Target Button:", target_text)

    # Search for the button element with matching text in the DOM elements
    target_element = next(
        (element for element in observation["dom_elements"] if element["text"] == target_text), 
        None
    )

    if target_element is not None:
        # Create a click action targeting the found button
        action = ActionTypes.click_element(ref=target_element["ref"])
        observation, reward, terminated, truncated, info = env.step(action)

        # Print feedback
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")

        if terminated or truncated:
            break
    else:
        print("Target button not found in the DOM elements.")

    time.sleep(1)  # Short delay before the next iteration

# Close the environment
env.close()
