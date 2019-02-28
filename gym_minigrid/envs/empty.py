import random

import numpy as np
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8,max_steps=None,**kwargs):
        if max_steps is None :
            max_steps = 4*size*size
        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height,stats=None,**kwargs):

        # Create an empty grid
        self.grid = Grid(width, height,stats)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        #self.start_pos = (1, 1)
        #self.start_dir = 0


        # Place a goal square in the bottom-right corner
        if self.goal is None :
            self.pos_goal = (width-2, height-2)
        else :
            index = np.argmax(self.goal,axis=0)
            self.pos_goal = (width-2,1) if index ==0 else (1,height-2)
        self.grid.set(self.pos_goal[0], self.pos_goal[1], Goal())

        while True :
            self.start_pos = (random.randint(1,width-2),random.randint(1,height-2))
            if self.start_pos != self.pos_goal:
                break
        self.start_dir = random.randint(0,3)

        self.mission = "get to the green goal square"

class EmptyEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16)

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
