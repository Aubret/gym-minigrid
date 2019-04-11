import torch
import glob
import os
from gym_minigrid.wrappers import *
from gym_minigrid.envs.empty import *
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.envs import *


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
