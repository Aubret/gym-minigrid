import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq as baseline_deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds
from baselines.deepq.deepq import ActWrapper
from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func


from gym_minigrid.wrappers import *
from gym_minigrid.envs.empty import *

"""
def model(inpt, num_actions, scope, reuse=False):
    This model takes as input an observation and returns values of all actions.
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out
"""

class deepq:
    def __init__(self,
        network,
        env,
        seed=None,
        lr=5e-4,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        batch_size=32,
        print_freq=50,
        checkpoint_freq=10000,
        checkpoint_path=None,
        gamma=0.98,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        **network_kwargs
            ):


        set_global_seeds(seed)
        #U.initialize()
        self.env = env
        sess = get_session()

        # Create the environment
        self.batch_size =batch_size
        self.print_freq = print_freq
        self.target_network_update_freq = target_network_update_freq
        #self.env = ImgObsWrapper(EmptyEnv())        # Create all the functions necessary to train the model
        model=build_q_func(network, **network_kwargs)
        def make_obs_ph(name):
            return ObservationInput(env.observation_space, name=name)

        self.act, self.train, self.update_target, self.debug = baseline_deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=param_noise,
            double_q=True
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': model,
            'num_actions': env.action_space.n,
        }

        self.act = ActWrapper(self.act, act_params)
        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        self.exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps), initial_p=1., final_p=exploration_final_eps)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        self.update_target()

        self.learning_steps =0
        self.episode_rewards = [0.0]

    def step(self,obs):
        #obs = self.env.reset()
        action = self.act(obs, update_eps=self.exploration.value(self.learning_steps))[0] #obs[None]
        new_obs, rew, done, _ = self.env.step(action)
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs


        self.episode_rewards[-1] += rew  # Last element
        if done :
            self.episode_rewards.append(0)
        if done and len(self.episode_rewards) % self.print_freq == 0:
            logger.record_tabular("steps", self.learning_steps)
            logger.record_tabular("episodes", len(self.episode_rewards))
            logger.record_tabular("mean episode reward", round(np.mean(self.episode_rewards[-101:-1]), 2))
            logger.dump_tabular()
        return action,obs,rew,done


    def learn(self,timesteps=1):

        for _ in range(timesteps):
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.

            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
            self.train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            # Update target network periodically.
            if self.learning_steps % self.target_network_update_freq == 0:
                self.update_target()
            self.learning_steps+=1

