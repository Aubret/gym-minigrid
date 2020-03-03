import random

import numpy as np
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, **kwargs):
        super().__init__(
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height,total_sum=None,**kwargs):

        # Create an empty grid
        self.grid = Grid(width, height,total_sum=None,**kwargs)#**kwargs)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (width //2, height//2)
        #self.start_pos=(1,1)
        self.start_dir = 0


        # Place a goal square in the bottom-right corner

        #if self.goal is None :
        ###############The goal definition
        #self.pos_goal = (width-2, height-2)
        #self.grid.set(self.pos_goal[0], self.pos_goal[1], Goal())



        #else :
        #    index = np.argmax(self.goal,axis=0)
        #    self.pos_goal = (width-2,1) if index ==0 else (1,height-2)
        #    self.grid.set(self.pos_goal[0], self.pos_goal[1], Goal())
        """
        while True :
            self.start_pos = (random.randint(1,width-2),random.randint(1,height-2))
            if self.start_pos != self.pos_goal:
                break
        self.start_dir = random.randint(0,3)
        """

        self.mission = "get to the green goal square"

    def _reward(self):

        """
        Compute the reward to be given upon success
        """
        #x,y = self.agent_pos
        #goal_x, goal_y = self.pos_goal

        #reward = -np.sqrt(math.pow(x-goal_x,2)+math.pow(y-goal_y,2)) / self.width
        #return reward

        return 1
        #return 1 - 0.9 * (self.step_count / self.max_steps)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16)

class EmptyWallEnv(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_grid(self, width, height,*args,**kwargs):
        super()._gen_grid( width, height,*args,**kwargs)
        self.grid.vert_wall(width//2, (height//2)-5, 4)

class EmptyWallHorzEnv(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_grid(self, width, height,*args,**kwargs):
        super()._gen_grid( width, height,*args,**kwargs)
        self.grid.horz_wall(width//2-6, (height//2), 5)

register(
    id='MiniGrid-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6'
)

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)

register(
    id='MiniGrid-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16'
)
