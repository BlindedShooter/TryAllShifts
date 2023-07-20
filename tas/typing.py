from typing import Union, Optional, Callable, Sequence, Dict, Tuple, Any, List, Final, TypeVar
from dataclasses import dataclass, fields
from functools import partial

import numpy as np
import torch
from torch.distributions import Distribution

from tas.distributions.continuous import FasterTransformedDistribution


# hmm
import gym
SettableEnv = gym.Env

ArrayLike = Union[np.ndarray, torch.Tensor]
InfoDict = Dict[str, Union[int, float, ArrayLike]]

Observation = ArrayLike
Action = ArrayLike
Reward = ArrayLike
Done = ArrayLike

State = ArrayLike  # encoded observation
Value = ArrayLike
ActionDist = Union[Distribution, FasterTransformedDistribution]
Dynamics = ArrayLike

ValueFuncType = Callable[[Observation, Optional[Dynamics]], Value]
CriticType = Callable[[Observation, Action, Optional[Dynamics]], Value]
ActorType = Callable[[Observation, Optional[Dynamics]], Action]


@dataclass
class TensorDataclass:
    def to(self, device: torch.device, dtype=torch.float32):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device, dtype=dtype)
        return self
    

    def to_tensor(self, device: torch.device, dtype=torch.float32):
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                self.__dict__[k] = torch.from_numpy(v).to(device, dtype=dtype)
        return self
    

    def to_numpy(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.detach().cpu().numpy()
        return self


    def torch_cat(self, other, dim: int = 1, raise_none = False):  # 0 for time, 1 for batch
        return self.__class__(**{
            k.name: noneconcat(getattr(self, k.name), getattr(other, k.name), dim=dim, raise_none=raise_none) for k in fields(self)
        })


@dataclass
class Transition(TensorDataclass):
    """
    A transition is a tuple of (s, a, r, s', d).
    
    Args:
        observations: Observations of the transition. (<B>, O)
        next_observations: Next observations of the transition. (<B>, O)
        actions: Actions of the transition. (<B>, A)
        rewards: Rewards of the transition. (<B>, 1)
        dones: Done flags of the transition. (<B>, 1)
        states: States of the transition. (<B>, S)
        action_dists: Action distributions of the transition. (Distrubution[<B>, A, Dist. Params>])
        values: Values of the transition. (<B>, 1)
        dynamics: Dynamics of the transition. (<B>, D)
    """
    observations: Optional[Observation] = None  # (<B>, O)
    next_observations: Optional[Observation] = None  # (<B>, O)
    actions: Optional[Action] = None  # (<B>, A)
    rewards: Optional[Reward] = None  # (<B>, 1)
    dones: Optional[Done] = None  # (<B>, 1)

    states: Optional[State] = None  # (<B>, S)
    next_states: Optional[State] = None  # (<B>, S)
    action_dists: Optional[ActionDist] = None  # (Distrubution[<B>, A, Dist. Params>])
    values: Optional[Value] = None  # (<B>, 1)
    dynamics: Optional[Dynamics] = None  # (<B>, D)


    @property
    def __len__(self):
        return self.actions.shape[0]


SelfTrajectory = TypeVar('SelfTrajectory', bound='Trajectory')
@dataclass
class Trajectory(TensorDataclass):
    """
    A trajectory is a sequence of consecutive transitions.
    
    Args:
        all_observations: Observations of the trajectory. (T + 1, <B>, O)
        actions: Actions taken in the trajectory. (T, <B>, A)
        rewards: Rewards received in the trajectory. (T, <B>, 1)
        dones: Done flags of the trajectory. (T, <B>, 1)
        states: States of the trajectory. (T, <B>, S)
        action_dists: Action distributions of the trajectory. (T, <B>, A, Dist. Params)
        values: Values of the trajectory. (T, <B>, 1)
        dynamics: Dynamics of the trajectory. (T, <B>, D)
    
    Properties:
        observations: Observations of the trajectory. (T, <B>, O)
        next_observations: Next observations of the trajectory. (T, <B>, O)
    """
    all_observations: Optional[Observation] = None  # (T + 1, <B>, O)
    actions: Optional[Action] = None  # (T, <B>, A)
    rewards: Optional[Reward] = None  # (T, <B>, 1)
    dones: Optional[Done] = None  # (T, <B>, 1)

    states: Optional[State] = None  # (T, <B>, S)
    action_dists: Optional[ActionDist] = None  # (T, <B>, A, Dist. Params)
    values: Optional[Value] = None  # (T, <B>, 1)
    dynamics: Optional[Dynamics] = None  # (T, <B>, D)


    @property
    def observations(self):
        return self.all_observations[:-1]
    
    @property
    def next_observations(self):
        return self.all_observations[1:]
    
    
    def __len__(self):
        return self.actions.shape[0]


    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, int):
            return Transition(
                observations=self.observations[idx] if self.observations is not None else None,
                next_observations=self.next_observations[idx] if self.next_observations is not None else None,
                actions=self.actions[idx] if self.actions is not None else None,
                rewards=self.rewards[idx] if self.rewards is not None else None,
                dones=self.dones[idx] if self.dones is not None else None,
                states=self.states[idx] if self.states is not None else None,
                action_dists=self.action_dists[idx] if self.action_dists is not None else None,
                values=self.values[idx] if self.values is not None else None,
                dynamics=self.dynamics[idx] if self.dynamics is not None else None
            )
        else:
            return Trajectory(
                all_observations=self.all_observations[idx.start:idx.stop+1:idx.step] if self.all_observations is not None else None,
                actions=self.actions[idx] if self.actions is not None else None,
                rewards=self.rewards[idx] if self.rewards is not None else None,
                dones=self.dones[idx] if self.dones is not None else None,
                states=self.states[idx] if self.states is not None else None,
                action_dists=self.action_dists[idx] if self.action_dists is not None else None,
                values=self.values[idx] if self.values is not None else None,
                dynamics=self.dynamics[idx] if self.dynamics is not None else None
            )


def noneconcat(x: Optional[ArrayLike], y: Optional[ArrayLike], dim: int, raise_none = False) -> Optional[torch.Tensor]:
    if x is not None and y is not None:
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return torch.cat([x, y], dim=dim)
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return np.concatenate([x, y], axis=dim)
        else:
            raise ValueError(f'x and y must be of the same type: {type(x)}, {type(y)}')
    elif x is None and y is None:
        return None
    elif raise_none:
        raise ValueError(f'only one of the attributes is None: {x}, {y} and raise_none is True')
    else:
        return None
