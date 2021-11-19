import gym
from util import Counter

env = gym.make('Tennis-ram-v0')
env.reset()
for i_episode in range(1):
    observation = env.reset()
    episode_reward = 0
    for t in range(10):
        env.render()
        # print('s:', observation)
        action = env.action_space.sample()
        # print('a:', action)
        observation, reward, done, info = env.step(action)
        # print('reward', reward)
        episode_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print('Episode reward:', episode_reward)
env.close()