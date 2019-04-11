import math
import operator
from functools import reduce

import numpy as np

import gym
from gym import error, spaces, utils


class FlatImgWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=9,
            shape=(imgSize,),
            dtype='uint8'
        )

    def observation(self, obs):
        image = obs['image']
        obs = image.flatten()
        return obs


class AutoReset(gym.core.Wrapper):
    """
    Automatically reset the environment after a done step
    """
    def __init__(self, env):
        super().__init__(env)
        #self.__dict__.update(vars(env))#add for example num env to the variables


    def step(self, action):
        obs, rewards, dones, info = self.env.step(action)
        if dones :
            obs = self.env.reset()
        return obs,rewards,dones,info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ExtendNenv(gym.core.Wrapper):
    """
    Makes data compatible with openai baseline, particularly the num_envs parameter
    """

    def __init__(self, env):
        super().__init__(env)
        #self.__dict__.update(vars(env))
        self.num_envs=1
        #self.__dict__.update({"num_envs":1})
        #print(vars(self))


    def step(self, action):
        obs, rewards, dones, info = self.env.step(action)
        return np.expand_dims(obs,axis=0),np.expand_dims(rewards,axis=0),np.expand_dims(dones,axis=0),info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return np.expand_dims(obs, axis=0)


class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (env.agentPos, env.agentDir, action)

        # Get the count for this (s,a) pair
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this (s,a) pair
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        reward += bonus

        return obs, reward, done, info

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (env.agentPos)

        # Get the count for this key
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this key
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        reward += bonus

        return obs, reward, done, info

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use rgb image as the only observation output
    """

    def __init__(self, env):
        super().__init__(env)
        # Hack to pass values to super wrapper
        self.__dict__.update(vars(env))
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class FullyFlatObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(self.env.width*self.env.height*3,),  # number of cells
            dtype='uint8'
        )
    def observation(self, obs):
        #full_grid = self.env.grid.encode().flatten()
        #index = 3*(self.env.agent_pos[0]+self.env.width*self.env.agent_pos[1])
        #full_grid[index:index+3] = np.array([255, self.env.agent_dir, 0])
        full_grid = self.env.grid.encode()
        full_grid[self.env.agent_pos[0]][self.env.agent_pos[1]] = np.array([10, self.env.agent_dir, 0])
        full_grid = full_grid.flatten()
        return full_grid



class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )
    def observation(self, obs):
        full_grid = self.env.grid.encode()
        full_grid[self.env.agent_pos[0]][self.env.agent_pos[1]] = np.array([255, self.env.agent_dir, 0])
        return full_grid

class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=64):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, imgSize + self.numCharCodes * self.maxStrLen),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, "mission string too long"
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

class PosWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.env.width,self.env.height),
            #shape=(6,),  # number of cells
            shape=(2,),
            dtype='uint8'
        )
    def observation(self, obs):
        dir = np.zeros(4,dtype=float)
        dir[self.env.agent_dir]=1.
        position = np.asarray([self.env.agent_pos[0],self.env.agent_pos[1]])
        return position#np.concatenate((position,dir))


class SimpleActionWrapper(gym.core.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        #dir = self.env.agent_dir
        #while dir != action:
        #    action = action-1 if action > 0 else 3
        #    self.env.step(0)
        #self.env.agent_dir = action
        return self.env.step(action)
        #return self.env.step(2)

