import gymnasium as gym
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt


env = gym.make("Taxi-v3")
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# load array
# print the array


observation, info = env.reset()

epochs = 0
penalties, reward = 0, 0

terminated = False



# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1

# For plotting metrics
scores = []
episodes = []
eps = []
points = []
for i in range(1, 10001):
    rand = random.seed(a=None, version=2)
    observation, info = env.reset(seed=rand)    
    #print(observation)
    epochs, penalties, reward, = 0, 0, 0
    terminated = False
    score = 0
    while not terminated:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[observation]) # Exploit learned values
        epsilon = epsilon * 0.99997
        if epsilon < 0.01:
            epsilon = 0.01
        next_observation, reward, terminated, truncated, info = env.step(action)        
        old_value = q_table[observation, action]
        next_max = np.max(q_table[next_observation])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[observation, action] = new_value

        score += reward

        observation = next_observation
        epochs += 1
    scores.append(score)
    episodes.append(i)
    if i % 50 == 0:
        points.append(np.mean(scores[-50:]))
        eps.append(np.mean(episodes[-50:]))
plt.figure(1)
plt.plot(eps,points)
plt.xlabel('X-episodes')
plt.ylabel('Y-reward')
plt.title("taxi-training-Q-learning")
plt.savefig("taxi-training-Q-learning.png")
print("Training finished.\n")
# save numpy array as csv file
from numpy import savetxt
# define data
# save to csv file
savetxt('foo.csv', q_table, delimiter=',')


data = loadtxt("./foo.csv", delimiter=',')

total_epochs, total_penalties = 0, 0
episodes = 1000
scores_test = []
episodes_test = []
eps_test = []
points_test = []
total = 0
for _ in range(episodes):
    rand = random.seed(a=None, version=2)
    observation, info = env.reset(seed=rand)    
    epochs, penalties, reward = 0, 0, 0
    score = 0
    terminated = False
    
    while not terminated:
        action = np.argmax(data[observation])
        observation, reward, terminated, truncated, info = env.step(action)        
        score += reward

    scores_test.append(score)
    episodes_test.append(_)
    if _ % 20 == 0:
        points_test.append(np.mean(scores_test[-20:]))
        eps_test.append(np.mean(episodes_test[-20:]))
    print("Iteration ",_, " score of ",score," points")
    total += score
plt.figure(2)
plt.plot(eps_test,points_test)
plt.xlabel('X-episodes_')
plt.ylabel('Y-reward_')
plt.title("taxi-testing-Q-learning")
plt.savefig("taxi-testing-Q-learning.png")
print("the average reward is ",total / 1000)
