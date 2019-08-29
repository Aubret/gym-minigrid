from a2c import a2c
from gym_minigrid.wrappers import *
from gym_minigrid.envs.empty import *
import torch
import os
import time
import glob
from collections import deque
from a2c_ppo_acktr import algo as algorithm
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import *
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot
from visdom import Visdom
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import explained_variance

#from baselines import logger



def prepare_cuda(no_cuda=False, cuda_deterministic=False,seed=1,num_processes=8):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not no_cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if not no_cuda else "cpu")
    return device

def create_directories(log_dir='/tmp/gym',save_dir='/trained_models'):
    try:
        os.makedirs(log_dir)  # create directory for log files
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    eval_log_dir = log_dir + "_eval"
    try:
        os.makedirs(eval_log_dir)  # create directory for evaluation
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

def make_envs_gridworld(num_processes,gamma,device,num_frame_stack=None):


    #envs = [lambda : ExtendNenv(AutoReset(FlatImgWrapper(EmptyEnv()))) for _ in range(num_processes)]
    envs = [lambda : FlatImgWrapper(EmptyEnv()) for _ in range(num_processes)]
    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs

def main(
        algo='a2c',
        lr=7e-4,
        eps=1e-5,
        alpha=0.99,
        gamma=0.9,
        use_gae=False,
        tau=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        seed=1,
        cuda_deterministic=False,
        num_processes=1,
        num_steps=4,
        ppo_epoch=4,
        num_mini_batch=32,
        clip_param=0.2,
        log_interval=100,
        save_interval=100,
        eval_interval=None,
        vis_interval=100,
        num_env_steps=10000,
        log_dir='/tmp/gym',
        save_dir='/trained_models',
        no_cuda=False,
        add_timestep=False,
        recurrent_policy=False,
        use_linear_lr_decay=False,
        use_linear_clip_decay = False,
        vis=False,
        port=8097

         ):

    num_updates = int(num_env_steps) // num_steps // num_processes #compute number of updates from number of steps, each process make numsteps moves for one update
    create_directories(log_dir,save_dir)
    device = prepare_cuda(no_cuda,cuda_deterministic,seed,num_processes)
    if vis:
        viz = Visdom(port=port)
        win = None

    #env=ExtendNenv(AutoReset(ImgObsWrapper(EmptyEnv()))) #gym.make("MiniGrid-Empty-8x8-v0"))


    #For parallelism
    envs = make_envs_gridworld(num_processes,gamma,device)

    #create policy model neural network and distribution
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': recurrent_policy})
    actor_critic.to(device)

    #Choose algorithm
    if algo == 'a2c':
        agent = algorithm.A2C_ACKTR(actor_critic, value_loss_coef,
                               entropy_coef, lr=lr,
                               eps=eps, alpha=alpha,
                               max_grad_norm=max_grad_norm)
    elif algo == 'ppo':
        agent = algorithm.PPO(actor_critic, clip_param, ppo_epoch, num_mini_batch,
                         value_loss_coef, entropy_coef, lr=lr,
                               eps=eps,
                               max_grad_norm=max_grad_norm)
    elif algo == 'acktr':
        agent = algorithm.A2C_ACKTR(actor_critic, value_loss_coef,
                               entropy_coef, acktr=True)

    #Object which generate batch of learning
    rollouts = RolloutStorage(num_steps, num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    #Reste environment and start rollout with reset obs
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    #episode_rewards = deque(maxlen=10)
    #episode_rewards=torch.FloatTensor([0.])
    episode_rewards = [0.]
    start = time.time()
    freq_timer=start
    #main loop
    for update in range(num_updates):

        #decaying learning rate
        if use_linear_lr_decay:
            # decrease learning rate linearly
            if algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, update, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, update, num_updates, lr)

        #increase born while number of updates increase
        if algo == 'ppo' and use_linear_clip_decay:
            agent.clip_param = clip_param  * (1 - update / float(num_updates))

        #num steps is the number of steps for one update
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            envs.render()

            #print(episode_rewards)
            #print(reward)
            episode_rewards[-1] += reward[0][0].item()
            if done[0]:
                episode_rewards.append(0.)
            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            #add interaction in rollout to learn after
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        #compute value function for the last interaction only
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        #compute the target for each interaction, the last is value
        rollouts.compute_returns(next_value, use_gae, gamma, tau)

        #Do learning step
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        #init the first observation
        rollouts.after_update()

        if update % log_interval == 0 and update is not 0:
            # Calculate the fps (frame per second)
            fps = int((num_steps * num_processes *log_interval) / (time.time() - freq_timer))

            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            #ev = explained_variance(values, rewards)
            logger.record_tabular("mean_rewards",np.mean(episode_rewards[-log_interval:-1]))
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", num_steps * num_processes * update)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(dist_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("total_time", float(time.time() - start))
            logger.dump_tabular()
            freq_timer = time.time()



        if vis and update % vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, log_dir, "Gridworld",
                                  algo, num_env_steps)
            except IOError:
                pass


if __name__ == '__main__':
    main()