import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import os

from utils import *
import csv

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = False
CKPT_SAVE_INTERVAL = 1000
NUM_OF_EPISODES = 500
csv_fields = ['Episode Number', 'Epsilon Value', 'Learn Step Counter', 'Total Reward']
gamma_diffs = [1.5, 2.0]

for gamma in gamma_diffs:

    CSV_FILE_NAME = f"gamma-diff-{gamma}_data.csv"
    # initializing the titles and rows list


    with open(CSV_FILE_NAME, 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=csv_fields, lineterminator='\n')
        csvwriter.writeheader()

        env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
        env = JoypadSpace(env, RIGHT_ONLY)

        env = apply_wrappers(env)

        agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n, gamma=gamma)

        if not SHOULD_TRAIN:
            folder_name = "2024-04-29-17_55_18"
            ckpt_name = "model_15000_iter.pt"
            agent.load_model(os.path.join("models", folder_name, ckpt_name))
            agent.epsilon = 0.2
            agent.eps_min = 0.0
            agent.eps_decay = 0.0

        env.reset()
        next_state, reward, done, trunc, info = env.step(action=0)

        print(f"Training 500 epoch w/ gamma={gamma}....")

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

            epsilon = agent.epsilon
            replay_buff_size = len(agent.replay_buffer)
            step_cntr = agent.learn_step_counter
            #print("Total reward:", total_reward, "Epsilon:", epsilon, "Size of replay buffer:", replay_buff_size, "Learn step counter:", step_cntr)
            csvwriter.writerow(
                {
                    'Episode Number': i, 
                    'Epsilon Value': epsilon, 
                    'Learn Step Counter': step_cntr, 
                    'Total Reward': total_reward
                }
            )

            if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
                agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

            #print("Total reward:", total_reward)
        env.close()
    
    print("Done, data written to ", CSV_FILE_NAME)



