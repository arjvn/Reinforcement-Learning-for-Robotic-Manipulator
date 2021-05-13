import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint
import numpy as np
from random import randint
from copy import copy
from math import floor, ceil
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
from pathlib import Path

PROB = 0.8
N_DISCRETE_ACTIONS = 27
N_COLOURS = 7
BELT_SPEED = 5
WS_LENGTH = 30 *BELT_SPEED
GEN_LENGTH = 30 *BELT_SPEED
BELT_WIDTH = (((N_COLOURS*2)-1)*BELT_SPEED)+1
BELT_LENGTH = WS_LENGTH + GEN_LENGTH
HEIGHT = 6 *BELT_SPEED
IDLE_COST, GOAL_REWARD, COLLISION_COST = -0.2, 1.0, -1.0
MIN_REWARD = -1
MAX_REWARD = 1
SpaceBetweenSkittles = 2 
SkittleTypes = BELT_WIDTH/(BELT_SPEED*SpaceBetweenSkittles)

# dictionary of possible actions (in 3D space)
dirDict = {0:(0,0,0),1:(0,0,1),2:(0,1,0),
          3:(0,0,-1),4:(0,-1,0),5:(0,1,1),
          6:(0,1,-1),7:(0,-1,-1),8:(0,-1,1),
          9:(1,0,0), 10:(1,0,1), 11:(1,1,0),
          12:(1,0,-1), 13:(1,-1,0), 14:(1,1,1),
          15:(1,1,-1), 16:(1,-1,-1), 17:(1,-1,1),
          18:(-1,0,0), 19:(-1,0,1), 20:(-1,1,0),
          21:(-1,0,-1), 22:(-1,-1,0), 23:(-1,1,1),
          24:(-1,1,-1), 25:(-1,-1,-1), 26:(-1,-1,1)}
heighDict = {0:4, 5:3, 10:2, 15:1, 20:0}

# dictionary of rewards
rewardDict = {0:(-10), #bump bounds,
                1:(-15), #pick item - item which was at the right location
                2:(10), #pick item - item which was at the wrong location
                4:(19), #Place Correct Location
                5:(-5), #Bump Belt
                7:(-1), #Free move
                8:(-3)} #Stay

actionDict={v:k for k,v in dirDict.items()}

class FactoryEnv(gym.Env):
  metadata = {'render.modes': ['human']} 

  def __init__(self, gen = None, workspace = None, world = None):
    self.observation_size = BELT_LENGTH
    self.prob = PROB
    self.fresh = True
    self.finished = False

    self.gen = [[0 for i in range(BELT_WIDTH)] for j in range(GEN_LENGTH)]  # for skittle generation

    self.workspace = [[0 for i in range(BELT_WIDTH)] for j in range(WS_LENGTH)] # for arm effector
    self.world = np.zeros((WS_LENGTH, BELT_WIDTH, HEIGHT), int)
    self.world[0][0][HEIGHT-1] = 9       # initial position of arm effector
    self.position = (0,0,HEIGHT-1)      # store initial position
    self.hold = False
    self.item = 0

    self.PreviousMoveDia = False
    self.CurrentMoveDia = False

  def step(self, action, armSpeed):
    reward_indicator = self.armMove(action, armSpeed) # beltMove is called inside armMove
    _,ay,az = self.position[0], self.position[1], self.position[2] 

    # dictionary for height
    if (az < 5): 
      az = 0
    elif (az < 10):
      az = 5
    elif (az < 15):
      az = 10
    elif (az < 20):
      az = 15
    elif (az < 30):
      az = 20

    # check reward indicator
    if reward_indicator > 8 or reward_indicator < 0:
      raise ("Error in return from ArmMove(): return out of bounds.")
    if ay > BELT_WIDTH or ay < 0:
      raise ("Error in return from ArmMove(): return out of bounds.")
    if reward_indicator == 3:  
      error = abs((ceil((ay+1)/10) - self.item))
      if error < 1 or error > 7:
          print("ceil is ", (ay+1)/10, "self.item is ", self.item, "error is ", error)
          raise ("Error in skittle placement error calculation.")
      reward = floor((error)**(1/3)*-10) + heighDict[az]

    elif reward_indicator == 6:  # BUMP ITEM
      error = abs((ceil((ay+1)/10 - self.item)))
      if error < 0 or error > 7:
          raise ("Error in skittle placement error calculation.")
      reward = floor((error)**(1/3)*-5) + heighDict[az]
    
    else:   
      reward = rewardDict[reward_indicator] + heighDict[az]

    # print("action: ", action, ", arm speed: ", armSpeed,", reward indicator: ", reward_indicator, ", result: ", reward, ", hold: ", self.hold,", item: ", self.item,", new position: ", self.position)
    return reward


  def reset(self):
    # Reset the state of the environment to the initial state
    self.finished = False

    self.fresh = True

  def _observe(self):
    # returns the array representing belt and arm position in workspace
    genNP = np.array(self.gen)
    wsNP = np.array(self.workspace)
    
    beltNP = np.concatenate((wsNP, genNP), axis = 0)
    observation = np.zeros((300, 66, 31), dtype=np.int64)

    # observation[:self.world.shape[0], :self.world.shape[1], :self.world.shape[2]] = self.world
    observation[self.position[0], self.position[1], self.position[2]+1] = 9
    observation[:,:,0] = beltNP
    return observation.tolist(), beltNP.tolist(), self.position, self.item

  def beltMove(self):
    # function to move the belt 
    newSkittleColumn = []
    
    # for loop to generate the skittles numbered from 1 to 7 randomly
    for i in range(BELT_WIDTH):
      if (i%BELT_SPEED==0):
          p = np.random.random()
          if p > self.prob:
            newSkittleColumn.append(randint(0, 7)) # allocate value between (1,7) as there are 7 skittle types
          else:
            newSkittleColumn.append(0) # allocate empty space which = 0
      else:
          newSkittleColumn.append(0) # allocate empty space which = 0

    emptBeltRow = [0 for i in range(BELT_WIDTH)] # dummy row for insertion

    beltGEN_MAT = copy(self.gen)
    beltWS_MAT = copy(self.workspace)

    # check the previous movement of the arm, and move the belt 
    if self.PreviousMoveDia == True:    # if diagonal, belt moves by 7 steps
        for i in range(2):        #push Rand
            beltGEN_MAT = beltGEN_MAT.append(emptBeltRow)
            transfer = beltGEN_MAT.pop(0) #oldest piece of data
            beltWS_MAT = beltWS_MAT.append(transfer)
            beltWS_MAT = beltWS_MAT.pop(0) #oldest piece of data
            if (len(beltWS_MAT)!=WS_LENGTH):
                raise Exception("BELT_GEN exceeds bounds ")
        beltGEN_MAT.append(newSkittleColumn)        #push Rand
        transfer = beltGEN_MAT.pop(0) #oldest piece of data
        beltWS_MAT.append(transfer)
        beltWS_MAT.pop(0) #oldest piece of data
        for i in range(2):        #push Rand
            # beltGEN_MAT = beltGEN_MAT.append(emptBeltRow)
            beltGEN_MAT = beltGEN_MAT.append(emptBeltRow)
            transfer = beltGEN_MAT.pop(0) #oldest piece of data
            beltWS_MAT = beltWS_MAT.append(transfer)
            beltWS_MAT = beltWS_MAT.pop(0) #oldest piece of data
            if (len(beltWS_MAT)!=WS_LENGTH):
                raise Exception("BELT_GEN exceeds bounds ")


    else:                         # if diagonal, belt moves by 5 steps
        for i in range(4):        #push (4)
            beltGEN_MAT.append(emptBeltRow)
            transfer = beltGEN_MAT.pop(0) #oldest piece of data
            beltWS_MAT.append(transfer) # transfer last row of gen to first row of workspace
            beltWS_MAT.pop(0) #oldest piece of data
            if (len(beltWS_MAT)!=WS_LENGTH):
                raise Exception("BELT_GEN exceeds bounds ")
        beltGEN_MAT.append(newSkittleColumn)        #push Rand
        transfer = beltGEN_MAT.pop(0) # oldest piece of data
        beltWS_MAT.append(transfer)   
        beltWS_MAT.pop(0) #oldest piece of data

    if self.CurrentMoveDia == True:
        for i in range(2):        #push Rand
            # beltGEN_MAT.append(emptBeltRow)
            beltGEN_MAT.append(emptBeltRow)
            transfer = beltGEN_MAT.pop(0) # oldest piece of data
            beltWS_MAT.append(transfer)   # transfer last row of gen to first row of workspace
            beltWS_MAT.pop(0) #oldest piece of data
            if (len(beltWS_MAT)!=WS_LENGTH):
                raise Exception("BELT_GEN exceeds bounds ")
    
    self.workspace = beltWS_MAT
    self.gen = beltGEN_MAT
    self.world[:, :, 0] = beltGEN_MAT

  def armMove(self, action, armSpeed):
    print("Next Action: ", action)
    print("Next Speed: ", armSpeed)
    Direction = dirDict[action]
    
    # check if arm movement is diagonal
    self.PreviousMoveDia = self.CurrentMoveDia
    if (action != 0 or action != 1 or action != 2 or action != 3 
      or action != 4 or action != 9 or action != 18):
      self.CurrentMoveDia = False
    else:
      self.CurrentMoveDia = True

    # move belt
    self.beltMove()

    # then move arm
    if (Direction==(0,0,0)):
        # no change in position
        return 8
    
    # map armspeed of 1 to 9 to [5,15]
    armSpeed = floor(((2/9)*(armSpeed+1)+(7/9))*5)
    
    # Check Bounds
    dx,dy,dz = ceil(Direction[0]*armSpeed), ceil(Direction[1]*armSpeed) , ceil(Direction[2]*armSpeed)

    ax,ay,az = self.position[0], self.position[1],self.position[2]
    
    if (ax+dx > self.world.shape[0]-1 or ax+dx < 0
        or ay+dy > self.world.shape[1]-1 or ay+dy < 0
        or az+dz > self.world.shape[2]-1 or az+dz < 0): # out of bounds
        # no change in position
        return 0
    # Check if arm interacts with the belt
    if az+dz == 0:
        if self.workspace[ax+dx][ay+dy] == 0:
            if self.hold == True:
                # place item
                self.workspace[ax+dx][ay+dy] = self.item
                if (ay+dy)%5 == 0:
                    if floor((ay+dy)/10)+1 == self.item and self.item != 0:
                        self.world[ax][ay][az] = 0
                        self.world[ax+dx][ay+dy][az+dz] = 9  
                        self.position = (ax+dx, ay+dy, az+dz)
                        return 4
                    else:
                        self.world[ax][ay][az] = 0
                        self.world[ax+dx][ay+dy][az+dz] = 9  
                        self.position = (ax+dx, ay+dy, az+dz)
                        
                        return 3
            else:   # hit the belt
                self.world[ax][ay][az] = 0
                self.world[ax+dx][ay+dy][az+dz] = 9
                self.position = (ax+dx, ay+dy, az+dz)
                
                return 5

        else:   #belt spot is occupied
            if self.hold == False:
                self.hold = True
                self.item = self.workspace[ax+dx][ay+dy]
                self.world[ax][ay][az] = 0
                self.world[ax+dx][ay+dy][az+dz] = 9  
                self.position = (ax+dx, ay+dy, az+dz)
                if self.item == ((ay/10)+1):   # picked item which was at right place
                  
                  return 1
                elif self.item != ((ay/10)+1): # picked item which was at wrong place
                  
                  return 2
            else:
                self.world[ax+dx][ay+dy][az+dz] = 9
                self.position = (ax+dx, ay+dy, az+dz)
                
                return 6
    # If not bound nor hit belt, then move in free space
    self.world[ax][ay][az] = 0
    # test
    # print("x", ax+dx)
    # print("y", ay+dy)
    # print("z", az+dz)
    self.world[ax+dx][ay+dy][az+dz] = 9
    self.position = (ax+dx, ay+dy, az+dz)
    if (self.hold == False and self.item != 0):
      raise Exception("Manipulator hold error - no hold but has item")
    if (self.hold == True and self.item == 0):
      raise Exception("Manipulator hold error - hold but has no item")
    return 7

  def render(self,reward, loss, episodes, dir, mode='human', close=False, notLast=True):
    conveyor = np.concatenate((self.workspace,self.gen))
    conveyor = np.swapaxes(conveyor,0,1)
    
    # Plot function
    episodes = np.linspace(0, episodes, episodes)
    # Graph for number of steps in each episode
    _, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    ax1.plot(episodes, reward, 'r-', linewidth=0.5)
    ax1.set_ylabel('Rewards per episode')
    ax1.set_xlabel('Episodes')
    ax2.plot(episodes, loss, 'b-', linewidth=0.5)
    ax2.set_ylabel('Episodic Loss')
    ax2.set_xlabel('Episodes')  
    ax3.matshow(conveyor, cmap=plt.cm.get_cmap('nipy_spectral', 7))
    ax3.axis('off')
    plt.pause(0.05)
    
    # plt.show()
    
    if (notLast):
      plt.close()
    else:
      savedir = os.path.join(dir, 'final.png')
      plt.savefig(savedir)
      plt.close()

    plt.show()