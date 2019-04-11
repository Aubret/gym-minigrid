import time
import functools
import tensorflow as tf
import numpy as np

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy


from baselines.a2c.utils import Scheduler, find_trainable_variables

from tensorflow import losses
#from baselines.a2c.runner import Runner

from runner import Runner

class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess) #just do one action step

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)



class a2c:

    def __init__(self,
        network,
        env,
        seed=None,
        nsteps=4,
        total_timesteps=int(80e6),
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=7e-4,
        lrschedule='linear',
        epsilon=1e-5,
        alpha=0.99,
        gamma=0.95,
        print_freq=100,
        load_path=None,
        **network_kwargs):

        set_global_seeds(seed)
        self.env= env
        self.nsteps=nsteps
        self.gamma = gamma
        self.print_freq = print_freq

        # Get the nb of env
        #self.env.num_envs=1 #probably parallelism

        self.policy = build_policy(env, network, **network_kwargs)  #generate policy function, with step and evaluate method

        # Instantiate the model object (that creates step_model and train_model)
        self.model = Model(policy=self.policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule) #contain all necessery function
        if load_path is not None:
            self.model.load(load_path)

        # Instantiate the runner object
        self.runner = Runner(env, self.model, nsteps=self.nsteps, gamma=self.gamma) #Use to run n step in environment and give observations

        # Calculate the batch_size
        self.nbatch = self.env.num_envs*nsteps
        self.total_timesteps = total_timesteps

        # Start total timer
        self.tstart = time.time()
        self.freq_timer = self.tstart

    def step(self,obs):
        #self.episode_rewards = [0.0]
        #action = self.act(obs, update_eps=self.exploration.value(self.learning_steps))[0] #obs[None]
        actions, values, states, neglogp = self.model.step(obs)#,S=None,M=[False])#states is the initial state, run n steps in environment
        obs, rewards, dones, _ = self.env.step(actions)#look like S is the next state for evaluation

        #self.episode_rewards[-1] += rew  # Last element
        #self.episode_rewards.append(0)

        #if done and len(self.episode_rewards) % self.print_freq == 0:
        #    logger.record_tabular("steps", self.learning_steps)
        #    logger.record_tabular("episodes", len(self.episode_rewards))
        #    logger.record_tabular("mean episode reward", round(np.mean(self.episode_rewards[-101:-1]), 1))
        #    logger.dump_tabular()
        return obs,actions,rewards,dones

    def learn(self,num_timesteps,timesteps=1):

        for update in range(timesteps):



            # Get mini batch of experiences
            obs, states, rewards, masks, actions, values =self.runner.run() # run n steps in environment

            policy_loss, value_loss, policy_entropy = self.model.train(obs, states, rewards, masks, actions, values)

            if num_timesteps % self.print_freq == 0:
                # Calculate the fps (frame per second)
                fps = int((num_timesteps * self.nbatch) / (time.time()-self.freq_timer))


                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                print(values,rewards)
                ev = explained_variance(values, rewards)
                logger.record_tabular("nupdates", num_timesteps)
                logger.record_tabular("total_timesteps", num_timesteps*self.nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(ev))
                logger.record_tabular("total_time", float(time.time()-self.tstart))
                logger.dump_tabular()
                self.freq_timer = time.time()
        return self.model
