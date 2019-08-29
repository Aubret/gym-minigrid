#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class FourRoomsEnv(MiniGridEnv):
    """
    Classical 4 rooms Gridworld environmnet.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None,**kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        #self.add_obs = 8
        #self.add_obs = 4
        self.add_obs = 1

        super().__init__(**kwargs)


    def _gen_grid(self, width, height,total_sum=None,**kwargs):
        # Create the grid
        self.grid = Grid(width, height,total_sum=total_sum,**kwargs)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2


        self.pos_doors = []

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    #pos = (xR, self._rand_int(yT + 1, yB))
                    pos=(xR,(yT+yB)//2)
                    self.pos_doors.append(pos)
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    #pos = (self._rand_int(xL + 1, xR), yB)
                    pos=((xL + xR)//2, yB)
                    self.pos_doors.append(pos)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.start_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.start_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.start_pos = (1*width//4,1*height//4)
            self.start_dir = 0

            #self.place_agent()


        if self._goal_default_pos is not None:
            goal = Goal()
            self.grid.set(*self._goal_default_pos, goal)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.grid.set(3*width//4,3*height//4,Goal())
            #self.place_obj(Goal())

        #self.mission = 'Reach the goal'

    def reset(self):
        MiniGridEnv.reset(self)
        self.distance_rooms()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.distance_rooms()
        return obs, reward, done, info


    def distance_rooms(self):
        agent_pos_x,agent_pos_y = self.agent_pos
        addings = []
        for i in range(len(self.pos_doors)):
            if i == 0:
                x_door,y_door=self.pos_doors[i]
                dx, dy = agent_pos_x-x_door,agent_pos_y-y_door
                #dx, dy = abs(agent_pos_x-x_door),abs(agent_pos_y-y_door)
                addings.append(dx)
                #addings.append(dy)
                #addings.append(abs(agent_pos_x-x_door)+abs(agent_pos_y-y_door))
        halfsize=self.width//2
        self.addings = (np.array(addings)-halfsize)*3/halfsize

register(
    id='MiniGrid-FourRooms-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv'
)
