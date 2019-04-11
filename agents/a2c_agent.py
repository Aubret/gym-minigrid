import torch
import time
import numpy as np
from a2c_ppo_acktr import algo as algorithm
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import *
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule

from baselines import logger


class a2c:

    def __init__(self,envs,algo='a2c',lr=7e-4,eps=1e-5,alpha=0.99,gamma=0.9,use_gae=False,
        tau=0.95,entropy_coef=0.01,value_loss_coef=0.5,max_grad_norm=0.5,seed=1,
        num_processes=1,num_steps=4,log_interval=100,save_interval=100,log_dir='/tmp/gym',
        save_dir='/trained_models',add_timestep=False,recurrent_policy=False,use_linear_lr_decay=False,
        use_linear_clip_decay = False,device=torch.device("cpu")):

        self.log_interval=log_interval
        self.gamma=gamma
        self.use_gae=use_gae
        self.tau=tau
        self.num_steps=num_steps
        self.num_processes=num_processes
        self.envs=envs
        self.num_updates=0



        #Initialize directory, cuda et compute the total number of update
        #self.num_updates = int(total_timesteps) // num_steps // num_processes  # compute number of updates from number of steps, each process make numsteps moves for one update


        # create policy model neural network and distribution
        self.actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                              base_kwargs={'recurrent': recurrent_policy})
        self.actor_critic.to(device)

        #We want to use A2C
        self.agent = algorithm.A2C_ACKTR(self.actor_critic, value_loss_coef,
                               entropy_coef, lr=lr,
                               eps=eps, alpha=alpha,
                               max_grad_norm=max_grad_norm)

        # Object which generate batch of learning
        self.rollouts = RolloutStorage(num_steps, num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  self.actor_critic.recurrent_hidden_state_size)
        self.rollouts.to(device)
        self.init_env()

        self.episode_rewards = [0.]
        self.start = time.time()
        self.freq_timer=self.start

    def init_env(self,render=False):
        # Reste environment and
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs) #init the rollout
        if render :
            self.envs.render()

    def step(self,render=False):
        step = self.rollouts.step
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                self.rollouts.obs[step],
                self.rollouts.recurrent_hidden_states[step],
                self.rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = self.envs.step(action)
        if render:
            self.envs.render()

        #For logging mean rewards for the the last interactions
        self.episode_rewards[-1] += reward[0][0].item()
        if done[0]:
            self.episode_rewards.append(0.)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                   for done_ in done])

        # add interaction in rollout to learn after
        self.rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)
        return obs,action,reward,done

    def nstep(self,render=False):
        #num steps is the number of steps for one update
        for _ in range(self.num_steps):
            obs,action,reward,done= self.step(render=render)
        return obs,action,reward,done

    def init_steps(self):
        # init the first observation
        self.rollouts.after_update()


    def learn(self):
        assert len(self.rollouts.obs)>1,"Must interact with environment before learning"
        self.num_updates+=1


        # compute value function for the last interaction only
        with torch.no_grad():
            next_value = self.actor_critic.get_value(self.rollouts.obs[-1],
                                                self.rollouts.recurrent_hidden_states[-1],
                                                self.rollouts.masks[-1]).detach()

        # compute the target for each interaction, the last is value
        self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)

        # Do learning step
        value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
        self.init_steps()

        if self.num_updates % self.log_interval == 0:
            # Calculate the fps (frame per second)
            fps = int((self.num_steps * self.num_processes * self.log_interval) / (time.time() - self.freq_timer))

            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            # ev = explained_variance(values, rewards)
            logger.record_tabular("mean_rewards", np.mean(self.episode_rewards[-self.log_interval:-2]))
            logger.record_tabular("nupdates", self.num_updates)
            logger.record_tabular("total_timesteps", self.num_steps * self.num_processes * self.num_updates)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(dist_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("total_time", float(time.time() - self.start))
            logger.dump_tabular()
            self.freq_timer = time.time()

        return self.actor_critic
