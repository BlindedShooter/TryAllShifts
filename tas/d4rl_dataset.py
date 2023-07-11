from typing import Union
import numpy as np
from tas.typing import *


__all__ = ["D4RLTransitionLoader", "D4RLTrajectoryLoader"]

class D4RLTransitionLoader:
    """
    Randomly shuffles and returns a minibatch of transitions of a D4RL dataset.
    """
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        timeouts: np.ndarray,
        terminals: np.ndarray,
        batch_size: int = 2,
        #return_done: bool = True  # whether to aggregate timeouts and terminals
        **_  # for "infos/...".
    ) -> None:
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.timeouts = timeouts
        self.terminals = terminals
        self.batch_size = batch_size

        self.indices = np.arange(len(self.observations), dtype=np.int64)
        self.batched_indices = np.lib.stride_tricks.as_strided(
            self.indices,  # original array
            (len(self.indices) // self.batch_size, self.batch_size),  # (Number of Full batches, Batch Size)
            (8 * self.batch_size, 8),  # Strides a full batch size memory for batch dimension, and then a single int64.
            writeable=False
        )

    def __iter__(self) -> Transition:
        np.random.shuffle(self.indices)

        for batch_idx in self.batched_indices:
            yield self.__getitem__(batch_idx)


    def __len__(self) -> int:
        return self.batched_indices.shape[0]


    @property
    def num_samples(self):
        "Number of samples."
        return self.indices.shape[0]


    def __getitem__(self, idx: Union[int, np.ndarray[np.int64]]) -> Transition:
        return Transition(
            observations=self.observations[idx],  # (B, ...)
            actions=self.actions[idx],  # (B, ...)
            rewards=self.rewards[idx],  # (B,)
            next_observations=self.next_observations[idx],  # (B, ...)
            dones=self.terminals[idx]  # (B,)
        )



class D4RLTrajectoryLoader:
    """
    Randomly shuffles and returns a minibatch of trajectories of a D4RL dataset.
    """
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
        batch_size: int = 2,
        trajectory_len: int = 4,
        #return_done: bool = True  # whether to aggregate timeouts and terminals
        **_  # for "infos/..."
    ) -> None:
        def get_windowed_view(arr):
            nonlocal trajectory_len
            return np.moveaxis(np.lib.stride_tricks.sliding_window_view(
                arr, window_shape=trajectory_len, axis=0, writeable=False
            ), [0, -1], [1, 0])  # (N, ..., Window) -> (Window, N, ...)

        self.observations = get_windowed_view(observations)  # (W, N - W + 1, ...)
        self.actions = get_windowed_view(actions)  # (W, N - W + 1, ...)
        self.rewards = get_windowed_view(rewards)[..., None]  # (W, N - W + 1, 1)
        self.terminals = get_windowed_view(terminals)[..., None]  # (W, N - W + 1, 1)
        self.batch_size = batch_size
        self.trajctory_len = trajectory_len

        self.observations = np.concatenate(
            (self.observations, next_observations[None, trajectory_len - 1:]),
            axis=0  # (W, N - W + 1, ...) + (1, N - W + 1,...) -> (W + 1, N - W + 1, ...)
        )
        
        self.indices = np.arange(self.observations.shape[1])[
            ~self.terminals[:-1, :, 0].any(axis=0)
        ]  # Trajectory should be from single episode  (what about step sampling balance within episode?)

        self.batched_indices = np.lib.stride_tricks.as_strided(
            self.indices,  # original array
            (len(self.indices) // self.batch_size, self.batch_size),  # (Number of Full batches, Batch Size)
            (8 * self.batch_size, 8),  # Strides a full batch size memory for batch dimension, and then a single int64.
            writeable=False
        )


    def __iter__(self) -> Trajectory:
        np.random.shuffle(self.indices)

        for batch_idx in self.batched_indices:
            yield self.__getitem__(batch_idx)


    def __len__(self) -> int:
        return self.batched_indices.shape[0]


    @property
    def num_samples(self):
        "Number of samples."
        return self.indices.shape[0]


    def __getitem__(self, idx: Union[int, np.ndarray[np.int64]]) -> Trajectory:
        return Trajectory(
            all_observations=self.observations[:, idx],  # (B, W + 1, ...)
            actions=self.actions[:, idx],  # (B, W, ...)
            rewards=self.rewards[:, idx],  # (B, W, 1)
            dones=self.terminals[:, idx]  # (B, W, 1)
        )
