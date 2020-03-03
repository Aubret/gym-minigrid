import gym

from deepq import deepq
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import *
from gym_minigrid.envs.empty import *
from baselines.a2c import a2c
import time


def main():
    env= SimpleActionWrapper(PosWrapper(EmptyEnv()))
    model = deepq(env=env, network='mlp', total_timesteps=3000, lr=5e-4, buffer_size=1000, gamma=0.9, print_freq=20,target_network_update_freq=200,hiddens=[64,64],dueling=False)

    for i in range(400):
        obs, done = env.reset(), False
        while not done:
            action,obs,rew,done = model.step(obs)
            if i > 50:
                model.learn(1)


    #model = a2c.learn(env=env, network='mlp', nsteps=2,total_timesteps=3000,lr=5e-4,gamma=0.9,log_interval=10)
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            action,obs,rew,done = model.step(obs)
            #actions = env.action_space.sample()



if __name__ == '__main__':
    main()