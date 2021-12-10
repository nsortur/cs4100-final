import random

import gym
import matplotlib.pyplot as plt
# import tensorflow as tf
from numpy import argmax

from util import Counter

# constants
EPSILON = 0.2
ALPHA = 0.2
GAMMA = 0.9

env = gym.make('Tennis-ram-v0')
q_vals = Counter()
action_space = env.action_space

actions = list(range(action_space.n))
print(action_space.sample())
print(env.observation_space)
# print(env.observation_space.low)
env.reset()

episode_rewards = []
q_size = []
# 300
for i_episode in range(1000):
    observation = env.reset()
    episode_reward = 0
    for t in range(5000):
        # env.render()
        obs_string = observation.tobytes()
        # e-greedy action selection
        rand = random.random()
        if rand < EPSILON:
            action = action_space.sample()
        else:
            action = actions[argmax([q_vals[(obs_string, act)] for act in actions])]

        next_observation, reward, done, info = env.step(action)
        if done:
            # terminal state q value is 0
            for act in actions:
                q_vals[(next_observation.tobytes(), act)] = 0

            print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
            break

        # update q values
        q_vals[(obs_string, action)] += \
            ALPHA * (reward + (GAMMA * max([q_vals[(next_observation.tobytes(), act)] for act in actions])) -
                     q_vals[(obs_string, action)])

        episode_reward += reward
        observation = next_observation

    episode_rewards.append(episode_reward)
    q_size.append( len(q_vals.keys()))
    print('Episode reward:', episode_reward)
    print('Total entries', len(q_vals.keys()))
    # print('best q value', q_vals.max())
    # print('min q val', q_vals.min())
    # print('average q val', q_vals.average())

f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(episode_rewards)
ax2 = f2.add_subplot(111)
ax2.plot(q_size)
plt.show()
#
# plt.plot(episode_rewards)
# plt.plot(q_size)
# plt.show()
env.close()
