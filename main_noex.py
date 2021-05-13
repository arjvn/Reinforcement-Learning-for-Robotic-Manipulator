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
import os
# np.set_printoptions(threshold=sys.maxsize)

BATCH_SIZE = 1

N_DISCRETE_ACTIONS = 27
MAX_MEM = 2  # 100

N_COLOURS = 7
SpaceBetweenSkittles = 2
PROB = 0.5

BELT_SPEED = 5
WS_LENGTH = 30 *BELT_SPEED
GEN_LENGTH = 30 *BELT_SPEED
BELT_WIDTH = (((N_COLOURS*2)-1)*BELT_SPEED)+1
BELT_LENGTH = WS_LENGTH + GEN_LENGTH
HEIGHT = 6 *BELT_SPEED
SkittleTypes = BELT_WIDTH/(BELT_SPEED*SpaceBetweenSkittles)
WORLD_ARRAY_SIZE = BELT_LENGTH*BELT_WIDTH*HEIGHT

EPISODES = 500

PATH = Path("""/Users/ritwik/Documents/Github/gym-factory/DDQN_no_exR/""")
PATH1 = Path("""/Users/ritwik/Documents/Github/gym-factory/DDQN_no_exR/DDQN_no_exR.png""")
# PATH = Path(os.path.join(os.getcwd(), '/DQN_no_exR/'))
# PATH1 = Path(os.path.join(os.getcwd(), '/DQN_no_exR/DQN_no_exR.png'))
# PATH = Path("""~/Documents/gym-factory/DDQN_no_exR/""")
# PATH1 = Path("""~/Documents/gym-factory/DDQN_no_exR/DDQN_no_exR.png""")

class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()

        # self.conv1 = nn.Conv3d(32, 1, (300,66,31), stride = 1)        # convolution layer 1
        self.conv1 = nn.Conv3d(1, 1, (271,47,16), stride = 1)
        # self.conv1.cuda()

        # self.conv2 = nn.Conv3d(1,300,1,stride=1)       # convolution layer 2
        self.conv2 = nn.Conv3d(1,1,(11,6,7),stride=1)
        # self.conv2.cuda()

        # self.fc1 = nn.Linear(30, 2048)                        # fully connected layer 1
        self.fc1 = nn.Linear(300, 2048)
        # self.fc1.cuda()
        # fc1 output is torch.Size([1, 2048])
        self.fc2 = nn.Linear(2048, 256)                       # fully connected layer 2
        # self.fc2.cuda()
        # fc2 output is torch.Size([1, 256])
        self.fc3 = nn.Linear(256, 27)                       # fully connected layer 2
        # self.fc3.cuda()
        # fc3 output is torch.Size([1, 27])

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()  #nn.SmoothL1Loss() nn.MSELoss() nn.L1Loss
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        if T.cuda.is_available():
            print("Using CUDA")
        else:
            print("CUDA not available")
        self.to(self.device)
        # self.norm = nn.InstanceNorm3d(32768, affine=True)

    def forward(self, observation):
        observation = T.Tensor(observation)
        # unsqueeze if not 4D
        # print('obs shape 1', observation.size())
        if len(list(observation.shape)) == 3:
            observation = T.unsqueeze(observation, 0)
            observation = T.unsqueeze(observation, 0)
        elif len(list(observation.shape)) == 4:
            observation = T.unsqueeze(observation, 1)
        # observation = self.norm(observation)
        # print('obs shape', observation.size())
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = observation.reshape(-1, 10, 300)

        observation = F.relu(self.fc1(observation))

        actions = F.relu(self.fc2(observation))
        actions = self.fc3(actions)

        # print('actions:',actions)

        return actions


class Agent(object):
    def __init__(self, gamma, epsilon, alpha,
                maxMemorySize, epsEnd=0.05,
                replace=10, actionSpace=[i for i in range(27)], speedSpace=[i for i in range(3)]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.speedSpace = speedSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

    def storeTransition(self, state, action, speed, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, speed, reward, state_])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, speed, reward, state_]
        self.memCntr += 1

    def chooseAction(self, observation):
        # print("epsilon: ", self.EPSILON)
        rand = np.random.random()
        
        actions = self.Q_eval.forward(observation)
        # epsilon soft approach
        if rand < 1 - self.EPSILON:
            # action = T.argmax(actions[1]).item() # choose greedy action
            a = actions.cpu().detach().numpy()
            # print ('action: ', a)
            # print ('max action: ', np.max(a))
            # print ('action dimensions', a.shape)
            # print("actions: ", a)
            act = np.unravel_index(a.argmax(), a.shape)
            # print("actions: ", a)
            # print ("act: ", act)
            armSpeed, bestAction =  int(act[0]), int(act[1])
            # print ("armSpeed: ", armSpeed)
            # print ("bestAction: ", bestAction)
        else:
            bestAction = np.random.choice(self.actionSpace) # else choose random action
            armSpeed = np.random.choice(self.speedSpace)
            # print("best Action: ", bestAction)
            # print("armSpeed: ", armSpeed)
        self.steps += 1

        return bestAction, armSpeed

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()       # zeros gradients for batch optimization

        # check whether target network should be replaced
        if self.replace_target_cnt is not None and \
            self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        length = 0

        while(length != BATCH_SIZE):
            # get some subset of the array of memory samples
            if self.memCntr + batch_size < self.memSize:    # at start memCntr = 0, memSize = 5k
                # print("cond if: ", range(self.memCntr))
                memStart = int(np.random.choice(range(self.memCntr)))
            else:
                # print("cond else: ", range(self.memCntr - batch_size-1))
                memStart = int(np.random.choice(range(self.memCntr - batch_size-1)))
            miniBatch = self.memory[memStart:memStart+batch_size]
            # print('mini batch', miniBatch)
            miniBatch = list(miniBatch)
            length = len(miniBatch)
        memory = np.array(list(miniBatch))


        # feedforward the current state and the predicted state
        # print("Qpred")
        # print('memory: ', memory[:, 0][:])
        # print("memory(state): ", memory[:, 0].shape)
        Qpred = self.Q_eval.forward(list(memory[:, 0])).to(self.Q_eval.device)
        # print("Qnext")
        Qnext = self.Q_next.forward(list(memory[:, 4])).to(self.Q_eval.device)
        # Qnext = self.Q_eval.forward(list(memory[:, 4])).to(self.Q_eval.device)

        # print("Qnext shape: ", Qnext.shape)
        # print("Qnext: ", Qnext)
        # maxA = T.argmax(Qnext, dim = 1).to(self.Q_eval.device)
        # print('maxA', maxA)
        rewards = T.Tensor(list(memory[:, 3])).to(self.Q_eval.device)
        # print("rewards size: ", rewards.shape)
        Qtarget = Qpred.clone()
        # print("Q target size:", Qtarget.shape)
        # print("Q target: ", Qtarget[:, maxA])
        # print("self gamma: ", T.max(Qnext[1]))
        # print("Qnext: ", Qnext)
        # self.GAMMA*T.max(Qnext[1])
        # Qtarget[maxA] = rewards + self.GAMMA*T.max(Qnext[1])

        for i in range(BATCH_SIZE):
            # get speed and action index
            armspeed = memory[i][2]
            action = memory[i][1]
            # print ("armSpeed: ", armspeed)
            # print("action: ", action)

            # print("Qtarget: ", Qtarget[i])
            # print("reward: ", rewards[i])
            # print("gamma: ", self.GAMMA)
            # print("Qnext: ", Qnext[i])

            Qtarget[i,armspeed, action] = rewards[i] + self.GAMMA*T.max(Qnext[i])
            # print("Qtarget 2: ", Qtarget[i, armspeed, action])
            # print("Qpred", Qpred[i, armspeed, action])


        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        # print("loss: ", loss)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

        return loss.item()

    def saveModel(self):
        
        save_prefix = PATH
        save_path1 = '{}/Q_eval.pt'.format(save_prefix)
        save_path2 = '{}/Q_next.pt'.format(save_prefix)
        output = open(save_path1, mode="wb")
        T.save(self.Q_eval.state_dict(), output)
        output.close() 

        output = open(save_path2, mode="wb")
        T.save(self.Q_next.state_dict(), output)
        output.close()

# set window for running averages in graphs
window_size = 10

def main():
    print("Initializing Factory Gym Environment...")
    env = gym.make('factory-v0')
    print("Factory Gym Environment Initialized")

    print("Initializing Agent...")
    brain = Agent(gamma = 0.9, epsilon=0.5,
                alpha=0.0001, maxMemorySize=MAX_MEM,
                replace=20)
    print("Agent Initialized")

    print ('Initializing memory')
    # observation = np.zeros((32,300,66,31), int)
    cnt = 1
    while brain.memCntr < brain.memSize:
        sys.stdout.write("\r%i of 2 memories stored" % cnt)
        sys.stdout.flush()
        # for i in range(31):
        #     observation[i+1,:,:,:] = observation[i,:,:,:]
        state,_,_,_ = env._observe()
        # observation[i,:,:,:] = state
        # observation = state
        action = random.randint(0,26)
        armSpeed = random.randint(0,9)
        reward = env.step(action,armSpeed)
        newState,_,_,_ = env._observe()

        brain.storeTransition(state, action, armSpeed, reward, newState)
        cnt +=1
    print()
    print ('Memory initialization complete')

    print ('Initializing Training Variables...')
    batch_size = BATCH_SIZE
    epsHistory = []
    rewardHistory = []
    lossHistory = []
    beltHistory = []
    positionHistory = []
    itemHistory = []

    state, belt, position, item = env._observe()

    beltHistory.append(belt)
    positionHistory.append(position)
    itemHistory.append(item)

    print('Beginning Training...')
    for i in range(EPISODES):
        print('starting episode', i+1, 'epsilon:', brain.EPSILON)
        epsHistory.append(brain.EPSILON)

        bestAction, armSpeed = brain.chooseAction(state)

        reward = env.step(bestAction,armSpeed)
        print("Reward: ", reward)
        newState, belt, position, item = env._observe()
        beltHistory.append(belt)
        positionHistory.append(position)
        
        # for x in range(31):
        #     observation[x+1,:,:,:] = observation[x,:,:,:]
        # observation[x,:,:,:] = newState
        brain.storeTransition(state, bestAction, armSpeed, reward, newState)



        state = newState
        loss = brain.learn(batch_size)
        print("Loss: ", loss)

        rewardHistory.append(reward)
        lossHistory.append(loss)

        # print("i ", i)
        # print ("i%10 ", i%10)
        if ((i-1)%100 == 0):
            env.render(rewardHistory, lossHistory, i+1, PATH1)
        
    print('Training Complete!')
    # env.render(rewardHistory, lossHistory, i+1, notLast=False)
    
    brain.saveModel()
    episodes = np.linspace(0, EPISODES, EPISODES)
    # to store data
    episodes = np.array(episodes)
    np.save(PATH/'episodes.npy', episodes)
    rewardHistory = np.array(rewardHistory)
    np.save(PATH/'rewardHistory.npy', rewardHistory)
    lossHistory = np.array(lossHistory)
    np.save(PATH/'lossHistory.npy', lossHistory)
    beltHistory = np.array(beltHistory)
    np.save(PATH/'beltHistory.npy', beltHistory)
    positionHistory = np.array(positionHistory)
    np.save(PATH/'positionHistory.npy', positionHistory)
    itemHistory = np.array(itemHistory)
    np.save(PATH/'itemHistory.npy', itemHistory)

    env.render(rewardHistory, lossHistory, i+1, PATH1, notLast=False)

if __name__ == "__main__":
    main()
