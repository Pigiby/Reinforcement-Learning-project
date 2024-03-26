import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random

device = (
    "cpu"
)#device that pytorch is going to use


#neural network with 5 layers: input-hidden-hidden-hidden-output


class DeepQNetwork(nn.Module):
    def __init__(self,lr,input_dims,fc1_dims,fc2_dims,fc3_dims,n_actions):
        super(DeepQNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss() 
        self.device = device



    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)

        return actions
#definition of the class agent with all its hyper-parameters

class Agent():
    def __init__(self,gamma,epsilon,lr,input_dims,batch_size,n_actions,max_mem_size = 10000,eps_end=0.03, eps_dec = 0.9997):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0
        self.Q_eval = DeepQNetwork(self.lr,n_actions=n_actions,input_dims=input_dims,fc1_dims=128,fc2_dims=128,fc3_dims=128).to(device)
        #replay_experience_buffer(state,action,reward,new_state,terminated)
        self.state_memory = np.zeros((self.mem_size,*input_dims),dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size,dtype =np.float32)        
        self.new_state_memory = np.zeros((self.mem_size,*input_dims),dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype =np.bool_)


    def store_transition(self,state,action,reward,next_state,done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def choose_action(self,observation):
        #epsilon-greedy strategy
        if np.random.random() > self.epsilon:
            state = T.tensor(observation).to(self.Q_eval.device) #convert the state into a tensor
            actions = self.Q_eval.forward(state) #pass in input this state to the NN
            action = T.argmax(actions).item()  #choose the action that maximizes the output
            
        else:
            action = np.random.choice(self.action_space) #choose a random action
        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return
        max_mem = min(self.mem_counter,self.mem_size)
        batch = np.random.choice(max_mem,self.batch_size,replace=False)
        #setting the gradient to zero, because by default the gradients are accumulated by default in buffers whenver .backward is called
        self.Q_eval.optimizer.zero_grad()
        batch_index = np.arange(self.batch_size,dtype= np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch].to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        #using the bellman-equation for updating the weigths of the neural network
        # q_target(s,a) <- reward + gamma * max (q_next(s_next,a_next))
        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next,dim=1)[0]
        #computes the loss and the gradient
        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        

while(1):
    value = input("type test/train\n")
    if value == "test":
        scores,eps_history,avg_points_test,avg_episodes_test = [],[],[],[]
        env = gym.make("MountainCar-v0")
        #hyper-parameteres setting
        agent = Agent(gamma = 0.99,epsilon=1,batch_size=128,n_actions=3,eps_end=0.05,input_dims=[2],lr=1e-3)
        scores,eps_history,avg_points,avg_episodes = [],[],[],[]        
        #loading the trained model        
        agent.Q_eval.load_state_dict(T.load("./mountaincar_model/model.pth"))
        score = 0
        for i in range(1000):
            done = False
            points = 0
            epochs = 0
            observation,info = env.reset()
            while not done:
                #convert the state in a tensor
                state = T.tensor(observation).to(agent.Q_eval.device)
                #pass the state to the NN
                actions = agent.Q_eval.forward(state)
                #take the index of the max output
                #.item returns the value of this tensor as a number. 
                action = T.argmax(actions).item()
                observation,reward,terminated,truncated,info = env.step(action)
                done = terminated or truncated
                points += reward
            score += points
            scores.append(points)
            eps_history.append(i)
            if i % 50 == 0:
                avg_points_test.append(np.mean(scores[-50:]))
                avg_episodes_test.append(np.mean(eps_history[-50:]))
            print("iteration ",i, "score", points)
        plt.figure(1)
        plt.plot(avg_episodes_test,avg_points_test)
        plt.xlabel('X-episodes')
        plt.ylabel('Y-reward')
        plt.title("MountainCar-testing")
        plt.savefig("MountainCar-testing.png")
        avg = score / 100
        print("the average reward is ",avg)
        env.close()
    elif value == "train":
        env = gym.make("MountainCar-v0")
        agent = Agent(gamma = 0.99,epsilon=1,batch_size=128,n_actions=3,eps_end=0.05,input_dims=[2],lr=1e-3)
        scores,eps_history,avg_points,avg_episodes = [],[],[],[]   
        n_games = 10000
        TAU = 0.005
        for i in range(n_games):
            
            score = 0
            done = False
            observation,info = env.reset(seed=random.randint(0,100))
            while not done:
                action = agent.choose_action(observation)
                
    
                observation_next,reward,terminated,truncated,info = env.step(action)
                score += reward
                done = terminated or truncated
                agent.store_transition(observation,action,reward,observation_next,done)

                agent.learn()
                observation = observation_next  
            scores.append(score)
            eps_history.append(i)
            #epsilon decreases at the end of the episode
            agent.epsilon = agent.epsilon * agent.eps_dec 
            if agent.epsilon < agent.eps_min:
                agent.epsilon = agent.eps_min

            print('-----------------------episode',i,"\n score%.2f" %score,"\nepsilon %.2f" %agent.epsilon)
            if i % 500 == 0:
                avg_points.append(np.mean(scores[-500:]))
                avg_episodes.append(np.mean(eps_history[-500:]))
        plt.figure(2)
        plt.plot(avg_episodes,avg_points)
        plt.xlabel('X-episodes')
        plt.ylabel('Y-reward')
        plt.title("MountainCar-training")
        plt.savefig("MountainCar-training.png")
        T.save(agent.Q_eval.state_dict(),"./mountaincar_model/model.pth")



