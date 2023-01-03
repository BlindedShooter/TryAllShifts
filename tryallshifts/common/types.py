"""
Defines types used in TAS

variables:
    Transition: A minibatch of transitions
    Trajectory: A minibatch of trajectories
"""
from typing import TypeVar, Tuple
import numpy as np


Transition = TypeVar("Transition", bound=Tuple[
        np.ndarray[np.float32],  # observation  (B, obs.dim.)
        np.ndarray[np.float32],  # action  (B, act.dim)
        np.ndarray[np.float32],  # reward  (B,)
        np.ndarray[np.float32],  # next observation  (B, obs.dim.)
        np.ndarray[np.bool_],  # timeout   (B,)
        np.ndarray[np.bool_]   # terminal  (B,)
])



Trajectory = TypeVar("Trajectory", bound=Tuple[
        np.ndarray[np.float32],  # observations (B, N + 1, obs.dim.)
        np.ndarray[np.float32],  # actions  (B, N, act.dim.)
        np.ndarray[np.float32],  # rewards  (B, N,)
        np.ndarray[np.bool_],  # timeout    (B, ) <-- always return a continuous trajectory.
        np.ndarray[np.bool_]   # terminal   (B, ) <-- always return a continuous trajectory.
])
