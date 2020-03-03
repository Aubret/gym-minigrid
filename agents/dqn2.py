import gym
from deepq import deepq
from a2c import a2c
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import *
from gym_minigrid.envs.empty import *
import tensorflow as tf
import time


def main():
    env=ExtendNenv(AutoReset(ImgObsWrapper(EmptyEnv()))) #gym.make("MiniGrid-Empty-8x8-v0"))
    #Do not work
    model = a2c( network='mlp',env=env, nsteps=2,total_timesteps=3000,lr=7e-3,gamma=0.9,print_freq=100,ent_coef=0.,)#,observ_placeholder=tf.TensorShape([63]))
    num_timesteps=0
    for i in range(3000):
        model.learn(i)
        num_timesteps+=1

    obs, done=env.reset(), False
    for _ in range(100):
        episode_rew = 0.
        while not done:
            obs, action,rew, done = model.step(obs)
            env.render()
            episode_rew += rew

    env.close()


if __name__ == '__main__':
    main()