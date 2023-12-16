# apply the learnt policy and render it

import aprel
import numpy as np
import gym
from typing import List, Union
import pickle
import numpy as np
from moviepy.editor import ImageSequenceClip
import warnings
import os
from stable_baselines3 import PPO

env_name = 'MountainCarContinuous-v0'
gym_env = gym.make(env_name)

np.random.seed(2023)
gym_env.seed(2023)

env = gym_env

# load policy 
model = PPO.load("ppo_mountain_car", env=env)

## Run the policy

max_episode_length = 1000
traj = []
obs = env.reset()
env_has_rgb_render = True
if env_has_rgb_render:
    try:
        frames = [np.uint8(env.render(mode='rgb_array'))]
    except:
        env_has_rgb_render = False
done = False
t = 0
while not done and t < max_episode_length:
    act = env.action_space.sample()
    traj.append((obs,act))
    obs, _, done, _ = env.step(act)
    t += 1
    if env_has_rgb_render:
        frames.append(np.uint8(env.render(mode='rgb_array')))
traj.append((obs, None))
if env_has_rgb_render:
    clip = ImageSequenceClip(frames, fps=30)
    clip_path = 'test_policy.mp4'
    clip.write_videofile(clip_path, audio=False)
    print('video clip written to test_policy.mp4')
else:
    clip_path = None

env.close()
