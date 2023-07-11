import numpy as np
import torch

from tas.typing import Observation, Action, Dynamics, Reward, Done, Transition, Trajectory


class NumpyReplay:
    def __init__(self, obs_dim: int, act_dim: int, dyn_dim: int, size: int):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.dyn_buf = np.zeros((size, dyn_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    

    def __len__(self):
        return self.size


    def store(self, obs: Observation, act: Action, dyn: Dynamics, next_obs: Observation, rew: Reward, done: Done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.dyn_buf[self.ptr] = dyn
        self.next_obs_buf[self.ptr] = next_obs
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    
    def store_batch(self, obs: Observation, act: Action, dyn: Dynamics, next_obs: Observation, rew: Reward, done: Done):
        batch_size = obs.shape[0]
        idxs = np.arange(self.ptr, self.ptr + batch_size) % self.max_size

        self.obs_buf[idxs] = obs
        self.act_buf[idxs] = act
        self.dyn_buf[idxs] = dyn
        self.next_obs_buf[idxs] = next_obs
        self.rew_buf[idxs] = rew
        self.done_buf[idxs] = done
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)


    def sample(self, batch_size: int=32) -> Transition:
        idxs = np.random.randint(0, self.size, size=batch_size)

        return Transition(
            observations=self.obs_buf[idxs],
            actions=self.act_buf[idxs],
            dynamics=self.dyn_buf[idxs],
            next_observations=self.next_obs_buf[idxs],
            rewards=self.rew_buf[idxs],
            dones=self.done_buf[idxs],
        )
    


class TorchReplay:
    def __init__(self, obs_dim: int, act_dim: int, dyn_dim: int, size: int, device: torch.device):
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.dyn_buf = torch.zeros((size, dyn_dim), dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size
    

    def __len__(self):
        return self.size
    

    def store(self, obs: Observation, act: Action, dyn: Dynamics, next_obs: Observation, rew: Reward, done: Done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.dyn_buf[self.ptr] = dyn
        self.next_obs_buf[self.ptr] = next_obs
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    

    def store_batch(self, obs: Observation, act: Action, dyn: Dynamics, next_obs: Observation, rew: Reward, done: Done):
        batch_size = obs.shape[0]
        idxs = torch.arange(self.ptr, self.ptr + batch_size) % self.max_size

        self.obs_buf[idxs] = obs
        self.act_buf[idxs] = act
        self.dyn_buf[idxs] = dyn
        self.next_obs_buf[idxs] = next_obs
        self.rew_buf[idxs] = rew
        self.done_buf[idxs] = done
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
    

    def store_transition(self, tran: Transition):  # assume (batch, ...) shape
        self.store_batch(tran.observations, tran.actions, tran.dynamics, tran.next_observations, tran.rewards, tran.dones)
    

    def store_trajectory(self, traj: Trajectory):  # assume (time, batch, ...) shape
        t, b = traj.observations.shape[:2]
        self.store_batch(
            traj.observations.view(t * b, -1), traj.actions.view(t * b, -1), traj.dynamics.view(t * b, -1),
            traj.next_observations.view(t * b, -1), traj.rewards.view(t * b), traj.dones.view(t * b).float()
        )


    def sample(self, batch_size: int=256) -> Transition:
        idxs = torch.randint(0, self.size, size=(batch_size,))

        return Transition(
            observations=self.obs_buf[idxs],
            actions=self.act_buf[idxs],
            dynamics=self.dyn_buf[idxs],
            next_observations=self.next_obs_buf[idxs],
            rewards=self.rew_buf[idxs],
            dones=self.done_buf[idxs],
        )