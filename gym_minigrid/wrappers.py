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
        env.obs = False
    def observation(self, obs):
        #full_grid = self.env.grid.encode().flatten()
        #index = 3*(self.env.agent_pos[0]+self.env.width*self.env.agent_pos[1])
        #full_grid[index:index+3] = np.array([255, self.env.agent_dir, 0])
        full_grid = self.env.grid.encode()
        full_grid[:,self.env.agent_pos[0],self.env.agent_pos[1]] = np.array([10, self.env.agent_dir, 0])
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
            high=10,
            shape=(3,self.env.width, self.env.height),  # number of cells
            dtype='uint8'
        )
        self.env.obs=True
    def observation(self, obs):
        full_grid = self.env.grid.encode()
        full_grid[:,self.env.agent_pos[0],self.env.agent_pos[1]] = np.array([10, self.env.agent_dir, 0])
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

class OnlyAddInfo(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert hasattr(env.unwrapped, "add_obs"),"need add_obs attributes in gym env"
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            # shape=(6,),  # number of cells
            shape=(env.unwrapped.add_obs,),
            dtype='float32'
        )

    def observation(self, obs):
        obs = self.unwrapped.addings
        return obs#np.concatenate((position,dir))

class AddInfo(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if hasattr(env.unwrapped, "add_obs"):
            adds=env.unwrapped.add_obs
            self.modified = True
        else :
            self.modified=False
            adds=0
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            # shape=(6,),  # number of cells
            shape=(env.observation_space.shape[0] + adds,),
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        if self.modified:
            obs = np.concatenate((obs,self.unwrapped.addings))
        return obs#np.concatenate((position,dir))

class PosWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            #shape=(6,),  # number of cells
            shape=(2,),
            dtype='float32'
        )
        self.half_gridsize = self.env.width/2
        self.unwrapped.obs=False
        #print(self.env.width)
        #print(self.env.height)

    def observation(self, obs):
        dir = np.zeros(4,dtype=float)
        dir[self.unwrapped.agent_dir]=1.

        position = (np.asarray([self.unwrapped.agent_pos[0],self.unwrapped.agent_pos[1]] )-self.half_gridsize)*3/self.half_gridsize
        etat=position
        #etat=np.concatenate((position,dir))
        return etat#np.concatenate((position,dir))


class SimpleActionWrapper(gym.core.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.action_space = spaces.Discrete(4)
        self.unwrapped.simple_direction=True

    def step(self, action):
        return self.env.step(action)


class MoveReward(gym.core.Wrapper):
    """
    When the agent reach th reward, he waits one more step in the exact same position.
    Then he can compute the Q-value of state where he never goes without diverging
    """

    def __init__(self, env):
        super().__init__(env)
        self.delayed_reward=False

    def reset(self):
        return self.env.reset()

    def step(self, action):

        if self.delayed_reward :
            self.delayed_reward=False
            return self.old_obs,self.old_reward,self.old_done,self.old_info

        obs, reward, done, info = self.env.step(action)
        if info["reached"]:
            self.delayed_reward=True
            self.old_obs=obs
            self.old_reward=reward
            self.old_info = info
            self.old_done=done
            return obs, 0., False, info
        else:
            return obs, reward, done, info

class AllImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use rgb image as the only observation output
    """

    def __init__(self, env):
        super().__init__(env)
        # Hack to pass values to super wrapper
        self.__dict__.update(vars(env))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            #shape=(6,),  # number of cells
            shape=(640,640,3),
            dtype='float32'
        )

    def observation(self, obs):
        obs = self.env.render(mode='rgb_array')
        return obs

class OneFullyFlatObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(self.env.width*self.env.height,),  # number of cells
            dtype='uint8'
        )
        env.obs = False
    def observation(self, obs):
        #full_grid = self.env.grid.encode().flatten()
        #index = 3*(self.env.agent_pos[0]+self.env.width*self.env.agent_pos[1])
        #full_grid[index:index+3] = np.array([255, self.env.agent_dir, 0])
        full_grid = self.env.grid.encode()[0]
        full_grid[self.env.agent_pos[0],self.env.agent_pos[1]] = np.array([10])
        full_grid = full_grid.flatten()
        return full_grid




class OneFullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(1,self.env.width, self.env.height),  # number of cells
            dtype='uint8'
        )
        self.env.obs=True
    def observation(self, obs):
        full_grid = self.env.grid.encode()[0]
        full_grid[self.env.agent_pos[0],self.env.agent_pos[1]] = np.array([10])
        return full_grid