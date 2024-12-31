import time
import gymnasium
import miniwob
from miniwob.action import ActionTypes
# Register MiniWoB environments
gymnasium.register_envs(miniwob)

env = gymnasium.make('miniwob/click-button-v1', render_mode='human')

# Reset the environment to start a new episode
observation, info = env.reset()
print("Instruction:", observation["utterance"])  # Displays the task instruction
time.sleep(4)

# Find the button specified in the instruction
target_text = observation["utterance"].split()[-1]  # Extract the target button text
print("Target Button:", target_text)

# Look for the button element with the matching text in the DOM
target_element = None
for element in observation["dom_elements"]:
    if element["text"] == target_text:
        target_element = element
        break

if target_element is not None:    
    # Click the target button
    action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=target_element["ref"])
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
else:
    print("Target button not found!")

time.sleep(2)
env.close()
