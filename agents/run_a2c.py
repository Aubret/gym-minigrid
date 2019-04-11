import gym

from agents.a2c_agent import a2c
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import *
from gym_minigrid.envs.empty import *
import tensorflow as tf
import time
import agents.tools as tools

def main():
    #env=ExtendNenv(AutoReset(ImgObsWrapper(EmptyEnv()))) #gym.make("MiniGrid-Empty-8x8-v0"))
    seed=1
    cuda_deterministic = False
    log_dir = '/tmp/gym'
    save_dir = './trained_models'
    num_processes=4
    gamma=0.9
    no_cuda=False
    num_updates=500

    #For parallelism
    tools.create_directories(log_dir, save_dir)
    device = tools.prepare_cuda(no_cuda, cuda_deterministic, seed, num_processes)
    envs = tools.make_envs_gridworld(num_processes,gamma,device)

    model = a2c(envs=envs, num_steps=4,lr=7e-4,gamma=gamma,log_interval=50,entropy_coef=0.01,seed=seed,log_dir=log_dir,save_dir=save_dir,num_processes=num_processes,device=device)#,observ_placeholder=tf.TensorShape([63]))
    for _ in range(num_updates):
        model.nstep(render=False)
        model.learn()
            #actions = env.action_space.sample()
    envs.close()
    #obs, done=env.reset(), False
    #model.init_env(render=True)

    envs = tools.make_envs_gridworld(num_processes,gamma,device)
    model.envs=envs
    for _ in range(100):
        model.init_env(render=True)
        done = [False]
        while not done[0]:
            obs, action,rew, done = model.step(render=True)
            time.sleep(0.1)
            model.init_steps()
        print("Episode reward", model.episode_rewards[-2])

    envs.close()


if __name__ == '__main__':
    main()