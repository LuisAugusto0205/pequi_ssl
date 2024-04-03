import gymnasium as gym
import rsoccer_gym
from gymnasium.wrappers import TransformObservation
from rsoccer_gym.ssl.ssl_go_to_ball.ssl_gym_go_to_ball import SSLGoToBallEnv

import numpy as np
import time


env = SSLGoToBallEnv(field_type=2)
obs, _ = env.reset()

moves = {
    'w': [0, 1, 0],
    's': [0, -1, 0],
    'a': [-1, 0, 0],
    'd': [1, 0, 0],
    'e': [0, 0, 1],
    'q': [0, 0, -1]
}

obs_names = ['X', 'Y', 'sin(O)', 'cos(O)', 'Vx', 'Vy', 'Vo']

done = False
n_frames = 30 # frames to skip between each action
steps = 0
while True:
    if done:
        obs, _ = env.reset()
    env.render()

    #if (steps % n_frames) == 0:
    with open('command.txt', 'r') as file:
        key = file.readline().strip()#input("\nDo a move: ")
    action = moves[key]

    observation, reward, done, *_ = env.step(action)
    steps+=1
    #print(f'\r{"  ".join([f"{n} {o:>7.2f}" for o, n in zip(observation[4:], obs_names)])}', end='')
    print(f'\r {reward:>7}', end='')
    time.sleep(0.03)