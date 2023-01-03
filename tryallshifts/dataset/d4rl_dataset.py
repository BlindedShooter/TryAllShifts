from typing import Union
import numpy as np
from tryallshifts.common.types import Transition, Trajectory


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
        batch_size: int = 32,
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
        return (
            self.observations[idx],  # (B, ...)
            self.actions[idx],  # (B, ...)
            self.rewards[idx],  # (B,)
            self.next_observations[idx],  # (B, ...)
            self.timeouts[idx],  # (B,)
            self.terminals[idx]  # (B,)
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
        timeouts: np.ndarray,
        terminals: np.ndarray,
        batch_size: int = 32,
        trajectoy_len: int = 10,
        #return_done: bool = True  # whether to aggregate timeouts and terminals
        **_  # for "infos/..."
    ) -> None:
        def get_windowed_view(arr):
            nonlocal trajectoy_len
            return np.moveaxis(np.lib.stride_tricks.sliding_window_view(
                arr, window_shape=trajectoy_len, axis=0, writeable=False
            ), -1, 1)  # (B, ..., Window) -> (B, Window, ...)

        self.observations = get_windowed_view(observations)  # (N - W + 1, W, ...)
        self.actions = get_windowed_view(actions)  # (N - W + 1, W, ...)
        self.rewards = get_windowed_view(rewards)  # (N - W + 1, W)
        self.timeouts = get_windowed_view(timeouts)  # (N - W + 1, W)
        self.terminals = get_windowed_view(terminals)  # (N - W + 1, W)
        self.batch_size = batch_size
        self.trajctory_len = trajectoy_len

        self.observations = np.concatenate(
            (self.observations, next_observations[trajectoy_len - 1:, None]),
            axis=1  # (N, Window, ...) + (N, "1", ...) -> (N, Window + 1, ...)
        )

        self.indices = np.arange(len(self.observations))[
            ~(self.timeouts | self.terminals)[:, :-1].any(axis=-1)
        ]  # Trajectory should be from single episode

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
        return (
            self.observations[idx],  # (B, W + 1, ...)
            self.actions[idx],  # (B, W, ...)
            self.rewards[idx],  # (B, W)
            self.timeouts[idx],  # (B, W)
            self.terminals[idx]  # (B, W)
        )
