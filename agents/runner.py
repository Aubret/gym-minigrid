import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner

#not use this
class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        #self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype


    def compute_rewards(self,epi_rewards,epi_dones,last_obs):
        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(last_obs, S=None, M=epi_dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(epi_rewards, epi_dones, last_values)):


                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)
                epi_rewards[n] = rewards
        return epi_rewards


    def run(self):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        epi_obs, epi_rewards, epi_actions, epi_values, epi_dones = [],[],[],[],[]

        mb_states = self.states
        for numsteps in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)

            # Append the experiences
            epi_obs.append(np.copy(self.obs))
            epi_actions.append(actions)
            epi_values.append(values)
            #epi_dones.append(self.dones)
            # Take actions in env and look the results
            obs, rewards, dones, _ = self.env.step(actions)
            self.env.render()
            epi_rewards.append(rewards)
            epi_dones.append(dones)
            if dones: #compute the reward before switching episode
                self.obs = self.env.reset()
                self.dones = False
                self.states=None
                epi_rewards = np.asarray(epi_rewards, dtype=np.float32).swapaxes(1, 0)
                epi_obs = np.asarray(epi_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape((numsteps + 1,) + self.env.observation_space.shape)  # .reshape(self.batch_ob_shape)
                epi_actions = np.asarray(epi_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
                epi_values = np.asarray(epi_values, dtype=np.float32).swapaxes(1, 0)
                epi_dones = np.asarray(epi_dones, dtype=np.bool).swapaxes(1, 0)
                epi_masks = epi_dones[:, :-1]
                epi_dones = epi_dones[:, 1:]


                mb_rewards.extend(self.compute_rewards(epi_rewards,epi_dones,obs))
                mb_obs.extend(epi_obs)
                mb_actions.extend(epi_actions)
                mb_values.extend(epi_values)
                mb_dones.extend(epi_dones)
                epi_obs, epi_rewards, epi_actions, epi_values, epi_dones = [], [], [], [], []
                continue
            self.states = states
            self.dones = dones
            self.obs = obs
        #epi_dones.append(self.dones)
        print(epi_dones)
        if not dones:
            epi_rewards = np.asarray(epi_rewards, dtype=np.float32).swapaxes(1, 0)
            epi_obs = np.asarray(epi_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape((numsteps + 1,) + self.env.observation_space.shape)  # .reshape(self.batch_ob_shape)
            epi_actions = np.asarray(epi_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
            epi_values = np.asarray(epi_values, dtype=np.float32).swapaxes(1, 0)
            epi_dones = np.asarray(epi_dones, dtype=np.bool).swapaxes(1, 0)
            epi_masks = epi_dones[:, :-1]
            #epi_dones = epi_dones[:, 1:]
            #Concat last iteartions
            mb_rewards.extend(self.compute_rewards(epi_rewards, epi_dones,obs))
            mb_obs.extend(epi_obs)
            mb_actions.extend(epi_actions)
            mb_values.extend(epi_values)
            mb_dones.extend(epi_dones)

        # Batch of steps to batch of rollouts



        #print(self.batch_action_shape)
        #print(mb_actions.shape)
        #mb_actions = mb_actions.reshape(self.batch_action_shape)
        #mb_actions = mb_actions.reshape([numsteps+1])

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values
