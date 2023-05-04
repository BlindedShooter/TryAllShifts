"""
Defines types used in TAS
"""
from typing import Union, Optional, NamedTuple, Callable, Sequence, Dict, Tuple
import numpy as np
import torch


ValueLike = Union[float, np.ndarray, torch.TensorType]
InfoDict = Dict[str, ValueLike]

FloatArray = Union[np.ndarray[np.float32], torch.FloatTensor]
BoolArray = Union[np.ndarray[np.bool_], torch.LongTensor]
State = FloatArray
Observation = FloatArray
Action = FloatArray
ActionDist = FloatArray
Reward = FloatArray
Done = Union[FloatArray, BoolArray]
Dynamics = FloatArray

TrajState = Sequence[State]
TrajObservation = Sequence[Observation]
TrajAction = Sequence[Action]
TrajActionDist = Sequence[ActionDist]
TrajReward = Sequence[Reward]
TrajDone = Sequence[Done]
TrajDynamics = Sequence[Dynamics]


ObservationPolicy = Callable[[Observation], Action]  # Observation -> Action
StatePolicy = Callable[[State], Action]  # State -> Action
DynamicsObservationPolicy = Callable[[Observation, Dynamics], Action]  # [O, D] -> Action
DynamicsStatePolicy = Callable[[State, Dynamics], Action]  # [S, D] -> Action

ObservationStochPolicy = Callable[[Observation], ActionDist]  # Observation -> Action
StateStochPolicy = Callable[[State], ActionDist]  # State -> Action
DynamicsObservationStochPolicy = Callable[[Observation, Dynamics], ActionDist]  # [O, D] -> Action
DynamicsStateStochPolicy = Callable[[State, Dynamics], ActionDist]  # [S, D] -> Action

Policy = Union[
    ObservationPolicy, StatePolicy, DynamicsObservationPolicy, DynamicsStatePolicy,
    ObservationStochPolicy, StateStochPolicy, DynamicsObservationStochPolicy, DynamicsStateStochPolicy
]


class Transition(NamedTuple):
    observation: State  # (B, ...)
    action: Action  # (B, ...)
    reward: Reward  # (B, )
    next_observation: State  # (B, ...)
    timeout: Optional[Done]  # (B, )
    terminal: Optional[Done]  # (B, )


class Trajectory(NamedTuple):
    observations: TrajState  # (B, W + 1, ...)
    actions: TrajAction  # (B, W, ...)
    rewards: TrajReward  # (B, W, )
    timeout: Optional[Done]  # (B, )
    terminal: Optional[Done]  # (B, )



TransitionDynamics = Callable[[State, Action], Transition]
TrajectoryDynamics = Callable[[State, Action], Trajectory]
