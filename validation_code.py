import sys
import gym
import gym_factory
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from time import strftime
import os
# np.set_printoptions(threshold=sys.maxsize)

# Editable global variables
BATCH_SIZE = 32     # batch size for each training 
MAX_MEM = 100       # size of memory in experience replay
PROB = 0.5          # probability of skittle generation
EPISODES = 5000     # total episodes 

# Non editable global variables 
N_DISCRETE_ACTIONS = 27
N_COLOURS = 7
SpaceBetweenSkittles = 2
BELT_SPEED = 5
WS_LENGTH = 30 *BELT_SPEED
GEN_LENGTH = 30 *BELT_SPEED
BELT_WIDTH = (((N_COLOURS*2)-1)*BELT_SPEED)+1
BELT_LENGTH = WS_LENGTH + GEN_LENGTH
HEIGHT = 6 *BELT_SPEED
SkittleTypes = BELT_WIDTH/(BELT_SPEED*SpaceBetweenSkittles)
WORLD_ARRAY_SIZE = BELT_LENGTH*BELT_WIDTH*HEIGHT

FILE = Path("""/Users/ritwik/Documents/Github/gym-factory/Trained_DDQN/Q_eval.pt""")


class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()

        # convolution layer 1
        self.conv1 = nn.Conv3d(1, 1, (271,47,16), stride = 1)

        # convolution layer 2
        self.conv2 = nn.Conv3d(1,1,(11,6,7),stride=1)
        
        self.fc1 = nn.Linear(300, 2048)             # fully connected layer 1
        
        self.fc2 = nn.Linear(2048, 256)             # fully connected layer 2
        
        self.fc3 = nn.Linear(256, 27)               # fully connected layer 3

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()  
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        if T.cuda.is_available():
            print("Using CUDA")
        else:
            print("CUDA not available")
        self.to(self.device)

    def forward(self, observation):
        observation = T.Tensor(observation)
        # unsqueeze if not 4D
        if len(list(observation.shape)) == 3:
            observation = T.unsqueeze(observation, 0)
            observation = T.unsqueeze(observation, 0)
        elif len(list(observation.shape)) == 4:
            observation = T.unsqueeze(observation, 1)
        
        observation = observation.to(self.device)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = observation.reshape(-1, 10, 300)

        observation = F.relu(self.fc1(observation))

        actions = F.relu(self.fc2(observation))
        actions = self.fc3(actions)

        return actions


def main():
    print("Initializing Factory Gym Environment...")
    env = gym.make('factory-v0')
    print("Factory Gym Environment Initialized")

    # print("Initializing Agent...")
    # brain = Agent(gamma = 0.9,alpha=0.0001, maxMemorySize=MAX_MEM,replace=None)
    # print("Agent Initialized")

    # device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    loaded_model = DeepQNetwork(0.0001)
    loaded_model.load_state_dict(T.load(FILE, map_location=T.device('cpu')))
    loaded_model.eval()

    observation,_,_,_ = env._observe()

    for i in range(EPISODES):
        actions = loaded_model.forward(observation)
        a = actions.cpu().detach().numpy()
        act = np.unravel_index(a.argmax(), a.shape)
        armSpeed, bestAction =  int(act[0]), int(act[1])

        env.step(bestAction,armSpeed)

        observation,belt,position,_ = env._observe()

        belt = np.array(belt)

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(2, 1, 1)
        
        beltn = np.swapaxes(belt,0,1)
        ax.matshow(beltn, cmap=plt.cm.get_cmap('nipy_spectral', 7))
        ax.axis('off')
        # plt.pause(0.05)
        ax = fig.add_subplot(2,1,2, projection='3d')  
        # ax = plt.axes(projection='3d')

        # Data for a three-dimensional line
        xline = position[0]
        yline = position[1]
        zline = position[2]
        
        violet = np.where(belt == 1)
        indigo = np.where(belt == 2)
        blue = np.where(belt == 3)
        green = np.where(belt == 4)
        yellow = np.where(belt == 5)
        orange = np.where(belt == 6)
        red = np.where(belt == 7)

        # print(zline, xline, yline)
        # print(len(violet))
        # print(belt)
        
        for i in range(len(violet[0])):
            ax.scatter(violet[0][i], violet[1][i], color = 'k', s = 4) 
        for i in range(len(indigo[0])):
            ax.scatter(indigo[0][i], indigo[1][i], color = 'y', s = 4)
        for i in range(len(blue[0])):
            ax.scatter(blue[0][i], blue[1][i], color = 'm', s = 4)
        for i in range(len(green[0])):
            ax.scatter(green[0][i], green[1][i], color = 'c', s = 4)
        for i in range(len(yellow[0])):
            ax.scatter(yellow[0][i], yellow[1][i], color = 'b', s = 4)
        for i in range(len(orange[0])):
            ax.scatter(orange[0][i], orange[1][i], color = 'g', s = 4)
        for i in range(len(red[0])):
            ax.scatter(red[0][i], red[1][i], color = 'r', s = 4)
        
        ax.scatter(xline, yline, zline, marker = "^", color = 'k', s = 50)
        
        # abclines3d(2, 3, 1, a = diag(3), col = "gray")
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 66)
        ax.set_zlim(0, 31)

        # env.render(rewardHistory, lossHistory, i+1, notLast=False)

        # _, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
        # ax1.plot(episodes, reward, 'r-', linewidth=0.5)
        # ax1.set_ylabel('Rewards per episode')
        # ax1.set_xlabel('Episodes')
        # ax2.plot(episodes, loss, 'b-', linewidth=0.5)
        # ax2.set_ylabel('Episodic Loss')
        # ax2.set_xlabel('Episodes')
        


        plt.pause(0.05)
        plt.close()




        


if __name__ == "__main__":
    main()
