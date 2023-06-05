"""
Defines types used in TAS
"""
from typing import Union, Optional, NamedTuple, Callable, Sequence, Dict, Tuple, Any, List
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

TrajState = State
TrajObservation = Observation
TrajAction = Action
TrajActionDist = ActionDist
TrajReward = Reward
TrajDone = Done
TrajDynamics = Dynamics


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
    observation: Optional[Observation] = None  # (B, ...)
    action: Optional[Action] = None  # (B, ...)
    action_dist: Optional[ActionDist] = None  # (B, ...)
    reward: Optional[Reward] = None  # (B, 1)
    next_observation: Optional[Observation] = None  # (B, ...)
    timeout: Optional[Done] = None  # (B, 1)
    terminal: Optional[Done] = None  # (B, 1)
    state: Optional[State] = None  # (B, E)
    next_state: Optional[State] = None  # (B, E)
    dynamics: Optional[Dynamics] = None


    @property
    def batch_size(self): 
        return self.reward.shape[0]  # hmm...


    def to_tensor(self, dtype=torch.float32, device=None):
        if isinstance(self.observation, np.ndarray):
            return Transition(
                torch.from_numpy(self.observation).to(dtype=dtype, device=device) if self.observation is not None else None,
                torch.from_numpy(self.action).to(dtype=dtype, device=device) if self.action is not None else None,
                torch.from_numpy(self.action_dist).to(dtype=dtype, device=device) if self.action_dist is not None else None,
                torch.from_numpy(self.reward).to(dtype=dtype, device=device) if self.reward is not None else None,
                torch.from_numpy(self.next_observation).to(dtype=dtype, device=device) if self.next_observation is not None else None,
                torch.from_numpy(self.timeout).to(dtype=dtype, device=device)  if self.timeout is not None else None,
                torch.from_numpy(self.terminal).to(dtype=dtype, device=device) if self.terminal is not None else None,
                torch.from_numpy(self.state).to(dtype=dtype, device=device) if self.state is not None else None,
                torch.from_numpy(self.next_state).to(dtype=dtype, device=device) if self.next_state is not None else None,
                torch.from_numpy(self.dynamics).to(dtype=dtype, device=device) if self.dynamics is not None else None
            )
        else:
            raise TypeError
        

    def to(self, dtype=torch.float32, device=None):
        if isinstance(self.observation, torch.TensorType):
            return Transition(
                self.observation.to(dtype=dtype, device=device) if self.observation is not None else None,
                self.action.to(dtype=dtype, device=device) if self.action is not None else None,
                self.action_dist.to(dtype=dtype, device=device) if self.action_dist is not None else None,
                self.reward.to(dtype=dtype, device=device) if self.reward is not None else None,
                self.next_observation.to(dtype=dtype, device=device) if self.next_observation is not None else None,
                self.timeout.to(dtype=dtype, device=device)  if self.timeout is not None else None,
                self.terminal.to(dtype=dtype, device=device) if self.terminal is not None else None,
                self.state.to(dtype=dtype, device=device) if self.state is not None else None,
                self.next_state.to(dtype=dtype, device=device) if self.next_state is not None else None,
                self.dynamics.to(dtype=dtype, device=device) if self.dynamics is not None else None
            )
        else:
            raise TypeError


    @property
    def flattened_view(self):
        return Transition(
            self.observation.view((self.batch_size, -1)) if self.observation is not None else None,
            self.action.view((self.batch_size, -1)) if self.action is not None else None,
            self.action_dist.view((self.batch_size, -1)) if self.action_dist is not None else None,
            self.reward.view((self.batch_size, -1)) if self.reward is not None else None,
            self.next_observation.view((self.batch_size, -1)) if self.next_observation is not None else None,
            self.timeout.view((self.batch_size, -1))  if self.timeout is not None else None,
            self.terminal.view((self.batch_size, -1)) if self.terminal is not None else None,
            self.state.view((self.batch_size, -1)) if self.state is not None else None,
            self.next_state.view((self.batch_size, -1)) if self.next_state is not None else None,
            self.dynamics.view((self.batch_size, -1)) if self.dynamics is not None else None
        )


    def to_numpy(self):
        if isinstance(self.observation, torch.TensorType):
            return Transition(
                self.observation.detach().cpu().numpy() if self.observation is not None else None,
                self.action.detach().cpu().numpy() if self.action is not None else None,
                self.action_dist.detach().cpu().numpy() if self.action_dist is not None else None,
                self.reward.detach().cpu().numpy() if self.reward is not None else None,
                self.next_observation.detach().cpu().numpy() if self.next_observation is not None else None,
                self.timeout.detach().cpu().numpy()  if self.timeout is not None else None,
                self.terminal.detach().cpu().numpy() if self.terminal is not None else None,
                self.state.detach().cpu().numpy() if self.state is not None else None,
                self.next_state.detach().cpu().numpy() if self.next_state is not None else None,
                self.dynamics.detach().cpu().numpy() if self.dynamics is not None else None
            )
        else:
            raise TypeError
    

    def to_numpy(self):
        if isinstance(self.observation, torch.TensorType):
            return Transition(
                self.observation.detach() if self.observation is not None else None,
                self.action.detach() if self.action is not None else None,
                self.action_dist.detach() if self.action_dist is not None else None,
                self.reward.detach() if self.reward is not None else None,
                self.next_observation.detach() if self.next_observation is not None else None,
                self.timeout.detach()  if self.timeout is not None else None,
                self.terminal.detach() if self.terminal is not None else None,
                self.state.detach() if self.state is not None else None,
                self.next_state.detach() if self.next_state is not None else None,
                self.dynamics.detach() if self.dynamics is not None else None
            )
        else:
            raise TypeError
    

    def check_shapes(self):
        pass


    def __getitem__(self, key: str) -> ValueLike:
        if key == "O":
            return self.observation
        elif key == "A":
            return self.action
        elif key == "R":
            return self.reward
        elif key == "S":
            return self.state
        elif key == "N":
            return self.next_observation
        elif key == "B":
            return self.action_dist
        elif key == "M":
            return self.next_state
        elif key == "T":
            return self.terminal
        elif key == "D":
            return self.dynamics
        else:
            raise KeyError("Transition keys are one of (O, A, R, S, N, B, M, T, D)")



class Trajectory(NamedTuple):
    observations: Optional[TrajObservation] = None  # (B, W + 1, ...)
    actions: Optional[TrajAction] = None  # (B, W, ...)
    action_dists: Optional[TrajActionDist] = None  # (B, W, ...)
    rewards: Optional[TrajReward] = None  # (B, W, )
    timeouts: Optional[Done] = None  # (B, )
    terminals: Optional[Done] = None  # (B, )
    states: Optional[TrajState] = None  # (B, W + 1, E)
    dynamics: Optional[TrajDynamics] = None # (B, W, D) or (B, D)


    @property
    def batch_size(self):
        return self.rewards.shape[0]

    @property
    def traj_len(self):
        return self.rewards.shape[1]


    def to_tensor(self, dtype=torch.float32, device=None):
        if isinstance(self.observations, np.ndarray):
            return Trajectory(
                torch.from_numpy(self.observations).to(dtype=dtype, device=device) if self.observations is not None else None,
                torch.from_numpy(self.actions).to(dtype=dtype, device=device) if self.actions is not None else None,
                torch.from_numpy(self.action_dists).to(dtype=dtype, device=device) if self.action_dists is not None else None,
                torch.from_numpy(self.rewards).to(dtype=dtype, device=device) if self.rewards is not None else None,
                torch.from_numpy(self.timeouts).to(dtype=dtype, device=device)  if self.timeouts is not None else None,
                torch.from_numpy(self.terminals).to(dtype=dtype, device=device) if self.terminals is not None else None,
                torch.from_numpy(self.states).to(dtype=dtype, device=device) if self.states is not None else None,
                torch.from_numpy(self.dynamics).to(dtype=dtype, device=device) if self.dynamics is not None else None, 
            )
        else:
            raise TypeError


    def to(self, dtype=torch.float32, device=None):
        if isinstance(self.observations, torch.TensorType):
            return Trajectory(
                self.observations.to(dtype=dtype, device=device) if self.observations is not None else None,
                self.actions.to(dtype=dtype, device=device) if self.actions is not None else None,
                self.action_dists.to(dtype=dtype, device=device) if self.action_dists is not None else None,
                self.rewards.to(dtype=dtype, device=device) if self.rewards is not None else None,
                self.timeouts.to(dtype=dtype, device=device)  if self.timeouts is not None else None,
                self.terminals.to(dtype=dtype, device=device) if self.terminals is not None else None,
                self.states.to(dtype=dtype, device=device) if self.states is not None else None,
                self.dynamics.to(dtype=dtype, device=device) if self.dynamics is not None else None, 
            )
        else:
            raise TypeError
        
    
    def flattened_view(self):
        return Trajectory(
            self.observations.view((self.batch_size, self.traj_len + 1, -1)) if self.observations is not None else None,
            self.actions.view((self.batch_size, self.traj_len, -1)) if self.actions is not None else None,
            self.action_dists.view((self.batch_size, self.traj_len, -1)) if self.action_dists is not None else None,
            self.rewards.view((self.batch_size, self.traj_len, -1)) if self.rewards is not None else None,
            self.timeouts.view((self.batch_size, self.traj_len, -1))  if self.timeouts is not None else None,
            self.terminals.view((self.batch_size, self.traj_len, -1)) if self.terminals is not None else None,
            self.states.view((self.batch_size, self.traj_len + 1, -1)) if self.states is not None else None,
            self.dynamics.view((self.batch_size, self.traj_len, -1)) if self.dynamics is not None else None, 
        )


    def check_shapes(self):
        pass


    def to_numpy(self):
        if isinstance(self.observations, torch.TensorType):
            return Trajectory(
                self.observations.detach().cpu().numpy() if self.observations is not None else None,
                self.actions.detach().cpu().numpy() if self.actions is not None else None,
                self.action_dists.detach().cpu().numpy()  if self.action_dists is not None else None,
                self.rewards.detach().cpu().numpy() if self.rewards is not None else None,
                self.timeouts.detach().cpu().numpy()  if self.timeouts is not None else None,
                self.terminals.detach().cpu().numpy() if self.terminals is not None else None,
                self.states.detach().cpu().numpy() if self.states is not None else None,
                self.dynamics.detach().cpu().numpy() if self.states is not None else None
            )
        else:
            raise TypeError
        

    def detach(self):
        if isinstance(self.observations, torch.TensorType):
            return Trajectory(
                self.observations.detach() if self.observations is not None else None,
                self.actions.detach() if self.actions is not None else None,
                self.action_dists.detach()  if self.action_dists is not None else None,
                self.rewards.detach() if self.rewards is not None else None,
                self.timeouts.detach()  if self.timeouts is not None else None,
                self.terminals.detach() if self.terminals is not None else None,
                self.states.detach() if self.states is not None else None,
                self.dynamics.detach() if self.states is not None else None
            )
        else:
            raise TypeError


    def __getitem__(self, key: str) -> ValueLike:
        if key == "O":
            return self.observations[:, :-1]
        elif key == "A":
            return self.actions
        elif key == "R":
            return self.rewards
        elif key == "N":
            return self.observations[:, 1:]
        elif key == "S":
            return self.states
        elif key == "B":
            return self.action_dists
        elif key == "T":
            return self.terminals
        elif key == "D":
            return self.dynamics
        else:
            raise KeyError("Trajectory keys are one of (O, A, R, S, B, T, D)")


TransitionDynamics = Callable[[Transition,], Dynamics]
TrajectoryDynamics = Callable[[Trajectory,], Dynamics]
