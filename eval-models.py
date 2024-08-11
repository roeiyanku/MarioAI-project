import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import os

from utils import *

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = False
DISPLAY = True
CKPT_SAVE_INTERVAL = 500
NUM_OF_EPISODES = 50


model_ep_cnt = 21_000
max_rewards = []
average_rewards = []

for m in range(21_000, model_ep_cnt + 1, CKPT_SAVE_INTERVAL):

    model_rewards = []
    max = 0

    env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)

    env = apply_wrappers(env)

    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

    if not SHOULD_TRAIN:
        folder_name = "2024-04-29-17_55_18"
        ckpt_name = f"model_{m}_iter.pt"
        agent.load_model(os.path.join("models", folder_name, ckpt_name))
        agent.epsilon = 0.2
        agent.eps_min = 0.0
        agent.eps_decay = 0.0

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)

    for i in range(NUM_OF_EPISODES):    
        #print("Episode:", i)
        done = False
        state, _ = env.reset()
        total_reward = 0
        while not done:
            a = agent.choose_action(state)
            new_state, reward, done, truncated, info  = env.step(a)
            total_reward += reward

            if SHOULD_TRAIN:
                agent.store_in_memory(state, a, reward, new_state, done)
                agent.learn()

            state = new_state

        #print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

        if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
            agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

        if total_reward > max:
            max = total_reward

        model_rewards.append(total_reward)
        
        #print("Total reward:", total_reward)

    env.close()

    avg = sum(model_rewards) / len(model_rewards)

    max_rewards.append(max)
    average_rewards.append(avg)

    print(f"model_{m} max reward: {max} avg reward: {avg}")

for num in max_rewards:
    print("\nMAX")
    print(num)

for num in average_rewards:
    print("\nAVERAGE")
    print(num)