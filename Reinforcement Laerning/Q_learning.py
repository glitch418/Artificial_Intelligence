import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  30000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    Q_table = defaultdict(default_Q_value)
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        while (not done):
            
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs,a)] for a in range(env.action_space.n)])
                action = np.argmax(prediction)

            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if not (terminated or truncated):
                Q_table[(obs, action)] = (
                    (1 - LEARNING_RATE) * Q_table[(obs, action)] + 
                    LEARNING_RATE * (reward + DISCOUNT_FACTOR * max([Q_table[(next_obs,a)] for a in range(env.action_space.n)]))
                )
            else:
                Q_table[(obs, action)] = (
                    (1 - LEARNING_RATE) * Q_table[(obs, action)] + 
                    LEARNING_RATE * reward
                )
            
            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
            
        EPSILON = max(EPSILON * EPSILON_DECAY, 0.001)

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    model_file = open(f'Q_TABLE_QLearning.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()