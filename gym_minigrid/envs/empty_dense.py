import numpy as np

from gym_minigrid.envs import EmptyEnv
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyDenseEnv(EmptyEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8,**kwargs ):
        super().__init__(
            size=size,
            max_steps=20,
            **kwargs
        )
        self.max_distance = np.linalg.norm(np.array([self.width,self.height]))#sqrt(pow(self.width,2.) +pow(self.height,2.))


    def _gen_grid(self, width, height,**kwargs):
        super()._gen_grid(width,height,**kwargs)
        self.mission = "Goal generation"

    def step(self, action,goal_end=True):
        obs, reward, done, info = super().step(action,goal_end=False)

        reward = np.linalg.norm(np.asarray(self.agent_pos)-np.asarray(self.pos_goal))#sqrt(pow(self.agent_pos[0]-self.pos_goal[0],2) + pow(self.agent_pos[1] - self.pos_goal[1],2))        reward = - reward / self.max_distance
        reward /= self.max_distance*self.max_steps
        return obs, -reward, done, info

class EmptyDenseEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6)

class EmptyDenseEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-Dense-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyDenseEnv6x6'
)

register(
    id='MiniGrid-Dense-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyDenseEnv'
)

register(
    id='MiniGrid-Dense-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyDenseEnv16x16'
)
