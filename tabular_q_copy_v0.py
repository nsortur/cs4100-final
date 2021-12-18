import random
import json
from statistics import mean

import matplotlib.pyplot as plt
import time

import gym
# import tensorflow as tf
import numpy as np
from numpy import argmax

from RLUtil import pickMaxAction, maxValForState, extractPolicy
from util import Counter

env = gym.make('Copy-v0')
# q_vals = Counter()  # Counter(json.load(open('counter.txt')))# Counter()

action_space = env.action_space
print(action_space)
print(env.observation_space)
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

# print(len(actions))
# list(range(action_space.n))
# print(action_space.sample())
# print(env.observation_space)
# print(env.observation_space.low)


# constants
EPSILON = .01
ALPHA = .3
GAMMA = .9
EPISODE_COUNT = 3600
import random
import json
from statistics import mean

import matplotlib.pyplot as plt
import time

import gym
# import tensorflow as tf
import numpy as np
from numpy import argmax

from RLUtil import pickMaxAction, maxValForState, extractPolicy
from util import Counter

env = gym.make('Copy-v0')
# q_vals = Counter()  # Counter(json.load(open('counter.txt')))# Counter()

action_space = env.action_space
print(action_space)
print(env.observation_space)
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

# print(len(actions))
# list(range(action_space.n))
# print(action_space.sample())
# print(env.observation_space)
# print(env.observation_space.low)


# constants
EPSILON = .01
ALPHA = .3
GAMMA = .9
EPISODE_COUNT = 50000

q_vals = Counter()
rewards = []


def stateIn(q_vals, s):
    """determines whether a state is in q val table"""
    for key in q_vals.keys():
        if key[0] == s:
            return True
    return False


# q-learning
for episodeNum in range(EPISODE_COUNT):
    # initializing state for beginning of episode
    state = env.reset()
    episode_reward = []

    done = False
    # keep stepping until we cannot step anymore
    while not done:
        # choosing action
        if random.random() < EPSILON or (not stateIn(q_vals, state)):
            action = env.action_space.sample()
        else:
            action = pickMaxAction(state, q_vals)

        # when state is not discovered yet
        if action is None:
            action = env.action_space.sample()

        # taking action
        state_prime, reward, done, info = env.step(action)

        # update step
        td_error = reward + GAMMA * maxValForState(state_prime, q_vals) - \
                   q_vals[(state, action)]

        q_vals[(state, action)] += ALPHA * td_error

        state = state_prime
        episode_reward.append(reward)

    rewards.append(sum(episode_reward))

    # stopping learning after we average rewards above 26
    if np.mean(rewards[-100:]) > 26.0:
        break
    print("Episode: %d, R: %d" % (episodeNum, sum(episode_reward)))

plt.xlabel("Episodes")
plt.ylabel("Reward ")
plt.scatter(range(len(rewards)), rewards)

# plt.plot(rewards)
plt.show()

env.close()

print(q_vals)
print(len(q_vals.keys()))

print(extractPolicy(q_vals))
0

q_vals = Counter()
rewards = []


def stateIn(q_vals, s):
    """determines whether a state is in q val table"""
    for key in q_vals.keys():
        if key[0] == s:
            return True
    return False


# q-learning
for episodeNum in range(EPISODE_COUNT):
    # initializing state for beginning of episode
    state = env.reset()
    episode_reward = []

    done = False
    # keep stepping until we cannot step anymore
    while not done:
        # choosing action
        if random.random() < EPSILON or (not stateIn(q_vals, state)):
            action = env.action_space.sample()
        else:
            action = pickMaxAction(state, q_vals)

        # when state is not discovered yet
        if action is None:
            action = env.action_space.sample()

        # taking action
        state_prime, reward, done, info = env.step(action)

        # update step
        td_error = reward + GAMMA * maxValForState(state_prime, q_vals) - \
                   q_vals[(state, action)]

        q_vals[(state, action)] += ALPHA * td_error

        state = state_prime
        episode_reward.append(reward)

    rewards.append(sum(episode_reward))

    # stopping
    if np.mean(rewards[-100:]) > 25.0:
        break
    print("Episode: %d, R: %d" % (episodeNum, sum(episode_reward)))

plt.xlabel("Episodes")
plt.ylabel("Reward ")
plt.scatter(range(len(rewards)), rewards)

# plt.plot(rewards)
plt.show()

env.close()

print(q_vals)
print(len(q_vals.keys()))

print(extractPolicy(q_vals))
