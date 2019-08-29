from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, Lava
import itertools as itt
import numpy as np

class SimpleLava(MiniGridEnv):

    CONST_RAND = [5,3,2,7,8,5,2,4,1,8,3,9,5,6]
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9 , obstacle_type=Lava, **kwargs):
        self.obstacle_type = obstacle_type
        super().__init__(
            size=size,
            **kwargs
        )

    def _gen_grid(self, width, height,**kwargs):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        cpt=0
        for i in range(2,width-2,2):
            for j in range(1,height-1):
                self.grid.set(i, j, self.obstacle_type())
            self.grid.set(i,self.CONST_RAND[cpt],None)
            cpt+=1


