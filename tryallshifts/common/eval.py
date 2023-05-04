from typing import Optional, Callable, List, Tuple, Any, NamedTuple, Iterable

import numpy as np
from gym.envs.mujoco import MujocoEnv
import torch

from tryallshifts.model.world_model import WorldModel
from tryallshifts.common.types import *


class TransitionPredResult(NamedTuple):
    state: State
    action: Action
    pred_state: State
    next_state: State
    pred_rew: Reward
    true_rew: Reward
    pred_done: Done
    true_done: Done


class TrajectoryPredResult(NamedTuple):
    states: TrajState
    actions: TrajAction
    pred_states: TrajState
    next_states: TrajState
    pred_rews: TrajReward
    true_rews: TrajReward
    pred_dones: TrajDone
    true_dones: TrajDone


class EvalVecMujocoEnv:
    envs: List[MujocoEnv]

    def __init__(self, make_env: Callable[[], MujocoEnv], n_envs: int=1):
        self.make_env = make_env
        self.envs = [make_env() for _ in range(n_envs)]
    
    @property
    def n_envs(self) -> int:
        return len(self.envs)
    
    @property
    def nq(self) -> int:
        return self.envs[0].model.nq
    
    @property
    def nv(self) -> int:
        return self.envs[0].model.nv

    
    def reset(self):
        for env in self.envs:
            env.reset()


    def set_state(self, states_):  # (B, ...)
        if isinstance(states_, torch.Tensor):
            states = states_.detach().cpu().numpy()
        else:
            states = states_

        if states.shape[0] > self.n_envs:
            self.envs += [self.make_env() for _ in range(states.shape[0] - self.n_envs)]

        for env_idx in range(states.shape[0]):
            state = states[env_idx]
            self.envs[env_idx].set_state(state[:self.nq], state[self.nq:])


    def step(self, actions_, return_info: bool = False):
        if isinstance(actions_, torch.Tensor):
            actions = actions_.detach().cpu().numpy()
        else:
            actions = actions_

        next_obs = np.empty((actions.shape[0], *self.envs[0].observation_space.shape), dtype=actions.dtype)
        rews = np.empty((actions.shape[0],), dtype=actions.dtype)
        dones = np.empty((actions.shape[0],), dtype=bool)
        infos = [] if return_info else None

        for env_idx in range(actions.shape[0]):
            next_ob, rew, done, info = self.envs[env_idx].step(actions[env_idx])
            next_obs[env_idx] = next_ob
            rews[env_idx] = rew
            dones[env_idx] = done
            if return_info:
                infos.append(info)

        return next_obs, rews, dones, infos



def world_model_transition_eval(
    model: WorldModel,
    dynamics_env: Optional[EvalVecMujocoEnv],
    transitions: Optional[Transition],
    state_gen: Optional[Callable[[], State]],
    action_gen: Optional[Policy],
    aggregates: Iterable[Callable[[TransitionPredResult], Any]] = (
        lambda result: result,
    ),
    exclude_current_positions_from_observation: bool = False,
    device: str="cuda"
) -> Tuple[Any]:
    if state_gen or action_gen:
        state = state_gen() if state_gen is not None else transitions.observation
        action = action_gen(state) if action_gen is not None else transitions.action

        dynamics_env.set_state(state)
        next_state, reward, done, _ = dynamics_env.step(action)

        if exclude_current_positions_from_observation:
            state = torch.from_numpy(state[..., 1:]).to(device)
        else:
            state = torch.from_numpy(state).to(device)
        action = torch.from_numpy(action).to(device)
        next_state = torch.from_numpy(next_state).to(device)
        reward = torch.from_numpy(reward).to(device)
    else:
        next_state = transitions.next_observation
        reward = transitions.reward

    next_state_pred, reward_pred, done_pred = model.predict(state, action)

    result = TransitionPredResult(
        state, action, next_state_pred, next_state, reward, reward_pred, done, done_pred
    )
    return [aggr(result) for aggr in aggregates]



def world_model_trajectory_eval(
    model: WorldModel,
    env: Optional[EvalVecMujocoEnv],
    trajectories: Optional[Trajectory] = None,
    state_gen: Optional[Callable[[], State]] = None,  # Only initial state
    action_gen: Optional[Policy] = None,
    aggregates: Iterable[Callable[[TrajectoryPredResult], Any]] = (lambda x: x, ),
    horizon: int = 50
) -> FloatArray:
    result = TrajectoryPredResult()
    return [aggr(result) for aggr in aggregates]


# Eval on...
# Offline Dset Train/Val, Single Step/Action Seq Deviation 평가
# Online env에 같은 Action Set / 같은 Policy로 평가

from tryallshifts.model.world_model import WorldModel

def eval_on_rollout(world_model: WorldModel, trajectory: Trajectory) -> None:
    first_obs = trajectory.observations[:, 0]
    
    pred_obs, pred_rews, pred_dones = world_model.rollout_actions(first_obs, trajectory.actions)

    obs_error = trajectory.observations - pred_obs
