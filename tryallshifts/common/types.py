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


    def to_tensor(self, dtype=torch.float32, device=None):
        return Transition(
            torch.from_numpy(self.observation).to(dtype=dtype, device=device),
            torch.from_numpy(self.action).to(dtype=dtype, device=device),
            torch.from_numpy(self.reward).to(dtype=dtype, device=device),
            torch.from_numpy(self.next_observation).to(dtype=dtype, device=device),
            torch.from_numpy(self.timeout).to(dtype=dtype, device=device)  if self.timeout is not None else None,
            torch.from_numpy(self.terminal).to(dtype=dtype, device=device) if self.terminal is not None else None
        )

    def to_numpy(self):
        if isinstance(self.observation, torch.TensorType):
            return Transition(
                self.observation.detach().cpu().numpy(),
                self.action.detach().cpu().numpy(),
                self.reward.detach().cpu().numpy(),
                self.next_observation.detach().cpu().numpy(),
                self.timeout.detach().cpu().numpy()  if self.timeout is not None else None,
                self.terminal.detach().cpu().numpy() if self.terminal is not None else None
            )
        else:
            return self


class Trajectory(NamedTuple):
    observations: TrajState  # (B, W + 1, ...)
    actions: TrajAction  # (B, W, ...)
    rewards: TrajReward  # (B, W, )
    timeout: Optional[Done]  # (B, )
    terminal: Optional[Done]  # (B, )



TransitionDynamics = Callable[[State, Action], Transition]
TrajectoryDynamics = Callable[[State, Action], Trajectory]
