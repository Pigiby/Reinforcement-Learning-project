import gymnasium as gym
import numpy as np
env = gym.make("MountainCar-v0")
observation, info = env.reset()
total_score=0
import matplotlib.pyplot as plt
episodes, scores, eps,points = [],[],[],[]
for _ in range(1000):
    done = False
    score = 0
    while not done:
        
        action = env.action_space.sample() #random_action
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
        if done:
            observation, info = env.reset()
            scores.append(score)
            episodes.append(_)
    if _ % 20 == 0:
        eps.append(np.mean(episodes[-20:]))
        points.append(np.mean(scores[-20:]))
    total_score += score
    print("iteration ",_,"score ",score)
plt.plot(eps,points)
plt.xlabel('X-episodes')
plt.ylabel('Y-reward')
plt.title("mountain-Car-testing-random")
plt.savefig("mountain-Car-testing-random.png")
    
print("average ",total_score/1000)
env.close()