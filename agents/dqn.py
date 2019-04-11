import gym

from deepq import deepq
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import *
from gym_minigrid.envs.empty import *
from baselines.a2c import a2c
import time


def main():
    #env = gym.make("CartPole-v0")
    #env=ExtendNenv(ImgObsWrapper(EmptyEnv())) #gym.make("MiniGrid-Empty-8x8-v0"))
    env=ImgObsWrapper(EmptyEnv()) #gym.make("MiniGrid-Empty-8x8-v0"))
    #wrap= ImgObsWrapper(env)
    #env.observation_space = env.observation_space.spaces.items()
    #for key,obs_space in env.observation_space.spaces.items():
     #   env.observation_space = obs_space
    #break
    #print(env.observation_space)
    model = deepq(env=env, network='mlp', total_timesteps=3000, lr=5e-4, buffer_size=50000, gamma=0.9, print_freq=100)

    for i in range(300):
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