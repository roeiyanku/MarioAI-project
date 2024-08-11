import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
from utils import get_current_date_time_string

# Paths and configurations
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

# Check for CUDA availability
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available, using CPU")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = False  # whether or not to train
DISPLAY = True  # whether or not to display game screen
CKPT_SAVE_INTERVAL = 500  # episode interval for when to save trained model
NUM_OF_EPISODES = 50_000  # number of episodes to run

# Setup environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Initialize agent
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Load pre-trained model if not training
if not SHOULD_TRAIN:
    folder_name = "2024-04-29-17_55_18"  # trained model directory
    ckpt_name = "model_21000_iter.pt"  # trained model to run
    # Load model on CPU
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

# Run episodes
env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

for i in range(NUM_OF_EPISODES):
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        # Choose action and perform it
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(a)
        total_reward += reward

        if SHOULD_TRAIN:
            # Store transition and learn if training
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state

    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        # Save the model checkpoint
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

    print("Total reward:", total_reward)

# Close environment
env.close()
