import random
import json
import matplotlib.pyplot as plt
import time

import gym
# import tensorflow as tf
from numpy import argmax

from util import Counter

# constants
EPSILON = 0
ALPHA = 0.2
GAMMA = 0.9

env = gym.make('Copy-v0')
# currently 500000 iterations
q_vals = Counter() # Counter(json.load(open('counter.txt')))# Counter()

action_space = env.action_space
actions = [(0, 0, 0),
           (0, 0, 1),
           (0, 0, 2),
           (0, 0, 3),
           (0, 0, 4),
           (0, 1, 0),
           (0, 1, 1),
           (0, 1, 2),
           (0, 1, 3),
           (0, 1, 4),
           (1, 0, 0),
           (1, 0, 1),
           (1, 0, 2),
           (1, 0, 3),
           (1, 0, 4),
           (1, 1, 0),
           (1, 1, 1),
           (1, 1, 2),
           (1, 1, 3),
           (1, 1, 4)]
    #list(range(action_space.n))
print(action_space.sample())
print(env.observation_space)
# print(env.observation_space.low)
env.reset()
# 300
total_reward = 0
size_q = []
for i_episode in range(10000):
    observation = env.reset()
    episode_reward = 0
    for t in range(5000):

        # e-greedy action selection
        rand = random.random()
        if rand < EPSILON:
            action = action_space.sample()
        else:
            action = actions[argmax([q_vals[str((observation, act))] for act in actions])]

        # print('action:', action)
        # env.render(mode="human")

        # time.sleep(1)

        # if action[1]:
        #     print('action:', action)
        #     env.render(mode="human")
        #     time.sleep(5)

        next_observation, reward, done, info = env.step(action)
        if done:
            # terminal state q value is 0
            for act in actions:
                q_vals[str((next_observation, act))] = 0
            env.render()
            print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
            print(env.observation_space)
            break

        # update q values
        q_vals[str((observation, action))] += \
            ALPHA * (reward + (GAMMA * max([q_vals[str((next_observation, act))] for act in actions])) -
                     q_vals[str((observation, action))])

        episode_reward += reward
        observation = next_observation

    total_reward += episode_reward

    print('Episode reward:', episode_reward)
    size_q.append(len(q_vals.keys()))
    print('Total entries', len(q_vals.keys()))

print('Total reward:', total_reward)
plt.plot(size_q)
plt.show()
# with open('counter.txt', 'w') as convert_file:
#     convert_file.write(json.dumps(q_vals))

env.close()
