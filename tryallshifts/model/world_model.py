from typing import Tuple, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from tryallshifts.common.types import *
from tryallshifts.common.types import InfoDict, Trajectory, Tuple, torch
from tryallshifts.model.net import construct_mlp
from tryallshifts.utils.misc import shape_flattener



class WorldModel(nn.Module):
    name: str = None
    loss_names: List[str] = []
    train_names: List[str] = []
    optim: torch.optim.Optimizer

    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
    
    def predict(self, observation: Observation, action: Action, **kwargs) -> Tuple[Observation, Reward, Done]: ...
    def rollout(self, observation: Observation, policy: Policy, detach: bool=False, **kwargs) -> Trajectory: ...
    def rollout_actions(self, observation: Observation, actions: TrajAction, detach: bool=False, **kwargs) -> Trajectory: ...
    def train(self, data: Dict[str, Any], **kwargs) -> InfoDict: ...
    def train_trans(self, transitions: Transition, **kwargs) -> InfoDict: ...
    def train_traj(self, trajectories: Trajectory, **kwargs) -> InfoDict: ...
    def calc_wm_loss(self, pred_traj: Trajectory, true_traj: Trajectory, 
                     **kwargs) -> Tuple[torch.Tensor, InfoDict]: ...


class SSMWorldModel(WorldModel):
    def encode_states(self, obs: Observation) -> State: ...
    def decode_states(self, states: State) -> Observation: ...
    def rssm_imagine(self, prev_rssm_state: State, prev_action: Action, done: Done) -> State: ...
    def rssm_rollout(self, prev_rssm_state: State, horizon: int, actor: nn.Module) -> Trajectory: ...
    def rssm_rollout_actions(self, prev_rssm_state: State) -> Trajectory: ...



class DynamicsWorldModel(WorldModel):
    def predict(self, observation: Observation, action: Action, dynamics: Dynamics
    ) -> Tuple[Observation, Reward, Done]: ...
    def rollout(self, observation: Observation, policy: Policy, dynamics: Dynamics, steps: int, detach: bool=False) -> Trajectory: ...
    def rollout_actions(self, observation: Observation, actions: TrajAction, dynamics: Dynamics, detach: bool=False) -> Trajectory: ...
    def train(self, transitions: Transition, dynamics: Dynamics) -> InfoDict: ...
    def train_traj(self, trajectories: Trajectory, dynamics: Dynamics) -> InfoDict: ...
    def calc_wm_loss(self, pred_traj: Trajectory, true_traj: Trajectory, 
                     dynamics: Dynamics,**kwargs) -> Tuple[torch.Tensor, InfoDict]: ...


class DynamicsSSMWorldModel(DynamicsWorldModel):
    def encode_states(self, obs: Observation) -> State: ...
    def decode_states(self, states: State) -> Observation: ...
    def rssm_imagine(self, prev_rssm_state: State, dynamics: Dynamics, prev_action: Action, done: Done) -> State: ...
    def rssm_rollout(self, prev_rssm_state: State, dynamics: Dynamics, horizon: int, actor: nn.Module) -> Trajectory: ...
    def rssm_rollout_actions(self, prev_rssm_state: State) -> Trajectory: ...



class MLPWorldModel(WorldModel):
    name = "mlp_wm"

    def __init__(self, config: OmegaConf):
        super().__init__(config)

        self.enc = construct_mlp(
            [
                shape_flattener(config.observation_shape) + shape_flattener(config.action_shape),
            ] + config.world_model.enc_arch, activation=nn.SiLU
        )
        
        self.recon_head = construct_mlp(
            config.world_model.enc_arch[-2:-1] + config.world_model.recon_arch + [shape_flattener(config.observation_shape),],
            activation=nn.SiLU
        )
        self.reward_head = construct_mlp(config.world_model.enc_arch[-2:-1] + config.world_model.reward_arch + [1,], nn.SiLU)
        self.done_head = nn.Sequential(
            construct_mlp(config.world_model.enc_arch[-2:-1] + config.world_model.done_arch + [1,], nn.SiLU),
            nn.Sigmoid()
        )

        self.optim: torch.optim.Optimizer = eval(config.world_model.optim_cls)(self.parameters(), **config.world_model.optim_kwargs)
    

    def forward(self, observation: Observation, action: Action, **_) -> Tuple[Observation, Reward, Done]:
        batch_size = observation.shape[0]
        state = self.enc(torch.concat((observation.reshape(batch_size, -1), action.reshape(batch_size, -1)), dim=1))

        return self.recon_head(state).reshape(batch_size, *self.config.observation_shape), self.reward_head(state), self.done_head(state)


    def predict(self, observation: Observation, action: Action, **_) -> Tuple[Observation, Reward, Done]:
        return self.forward(observation, action)
    

    def rollout(self, observation: Observation, policy: ObservationPolicy, steps: int=8, **_) -> Trajectory:
        observations = [observation,]
        actions = [policy(observation),]
        rewards = []
        dones = []

        batch_size = observation.shape[0]

        for _ in range(steps - 1):
            state = self.enc(torch.concat((observations[-1].view(batch_size, -1), actions[-1].view(batch_size, -1)), dim=1))
            observations.append(self.recon_head.forward(state).view(batch_size, *self.config.observation_shape))
            actions.append(policy(observations[-1]).view(batch_size, *self.config.action_shape))
            rewards.append(self.reward_head.forward(state))
            dones.append(self.done_head.forward(state))
        
        return Trajectory(observations, actions, None, rewards, None, dones)


    def rollout_actions(self, observation: Observation, actions: TrajAction, **_) -> Trajectory:
        observations = [observation,]
        rewards = []
        dones = []

        batch_size = observation.shape[0]

        for step_idx in range(actions.shape[1]):
            state = self.enc(torch.concat((observations[-1].view(batch_size, -1), actions[:, step_idx].view(batch_size, -1)), dim=1))
            observations.append(self.recon_head.forward(state).view(batch_size, *self.config.observation_shape))
            rewards.append(self.reward_head.forward(state))
            dones.append(self.done_head.forward(state))
        
        return Trajectory(observations, actions, None, rewards, None, dones)


    def _calc_loss(self, transitions: Transition) -> Tuple[torch.Tensor, InfoDict]:
        next_obs, rew, done = self.predict(transitions.observation, transitions.action)
        obs_loss = F.mse_loss(next_obs, transitions.next_observation)
        rew_loss = F.mse_loss(rew.squeeze(), transitions.reward)
        done_loss = F.binary_cross_entropy(done.squeeze(), transitions.terminal)

        loss = obs_loss + rew_loss + done_loss
        return loss, {'wm/loss': loss.item(), 'wm/obs_loss': obs_loss.item(),
                       'wm/rew_loss': rew_loss.item(), 'wm/done_loss': done_loss.item()}


    def train(self, transitions: Transition) -> InfoDict:
        self.optim.zero_grad()
        loss, info = self._calc_loss(transitions)

        loss.backward()
        self.optim.step()

        return info



class DynamicsMLPWorldModel(DynamicsWorldModel):
    name = "mlp_dwm"

    def __init__(self, config: OmegaConf):
        super().__init__(config)
        self.enc = construct_mlp(
            [
                config.dyn_encoder.dyn_dim + shape_flattener(config.observation_shape) + shape_flattener(config.action_shape),
            ] + config.world_model.enc_arch, 
            activation=nn.SiLU
        )
        
        self.recon_head = construct_mlp(
            [config.world_model.enc_arch[-1],] + config.world_model.recon_arch + [shape_flattener(config.observation_shape),],
            activation=nn.SiLU
        )
        self.reward_head = construct_mlp([config.world_model.enc_arch[-1],] + config.world_model.reward_arch + [1,], nn.SiLU)
        self.done_head = nn.Sequential(
            construct_mlp([config.world_model.enc_arch[-1],] + config.world_model.done_arch + [1,], nn.SiLU),
            nn.Sigmoid()
        )

        self.optim: torch.optim.Optimizer = eval(config.world_model.optim_cls)(self.parameters(), **config.world_model.optim_kwargs)


    def forward(self, observation: Observation, action: Action, dynamics: Dynamics) -> torch.Tensor:
        """
        observation: (B, Obs) / action: (B, Act) / dynamics: (B, Dyn)
        Returns predicted next step's state.
        """
        batch_size = observation.shape[0]
        latent = self.enc(torch.concat(
            (observation.view(batch_size, -1), action.view(batch_size, -1), dynamics), dim=1
        ))

        return latent


    def predict(self, observation: Observation, action: Action, dynamics: Dynamics) -> Tuple[Observation, Reward, Done]:
        latent = self.forward(observation, action, dynamics)
        return self.recon_head(latent), self.reward_head(latent), self.done_head(latent)


    # What about dynamics update every step?
    def rollout(self, observation: Observation, policy: ObservationPolicy,
                dynamics: Dynamics, steps: int = 8, detach: bool=False, **_) -> Trajectory:
        observations = [observation,]
        actions = []
        rewards = []
        dones = []

        batch_size = observation.shape[0]

        for _ in range(steps - 1):
            action = policy(observations[-1]).view(batch_size, *self.config.action_shape)
            state = self.forward(observations[-1], action, dynamics)

            next_ob = self.recon_head.forward(state).view(batch_size, *self.config.observation_shape)
            rew = self.reward_head.forward(state)
            done = self.done_head.forward(state)

            if detach:
                actions.append(action)
                observations.append(next_ob.detach())
                rewards.append(rew.detach())
                dones.append(done.detach())
        
        return Trajectory(
            observations=torch.stack(observations, dim=1),  # List[(B, O)] -> (B, T, O)
            actions=torch.stack(actions, dim=1),  # List[(B, A)] -> (B, T, A)
            rewards=torch.concat(rewards, dim=1),  # List[(B)] -> (B, T)
            terminals=torch.concat(dones, dim=1)   # List[(B)] -> (B, T)
        )


    def rollout_actions(self, observation: Observation, actions: TrajAction, dynamics: Dynamics, detach: bool=False) -> Trajectory:
        observations = [observation,]
        rewards = []
        dones = []

        batch_size = observation.shape[0]

        for t in range(actions.shape[1]):
            state = self.forward(observations[-1], actions[:, t], dynamics)
            observations.append(self.recon_head.forward(state).view(batch_size, *self.config.observation_shape))
            rewards.append(self.reward_head.forward(state))
            dones.append(self.done_head.forward(state))
        
        return Trajectory(
            observations=torch.stack(observations, dim=1),
            actions=actions,
            rewards=torch.concat(rewards, dim=1),
            terminals=torch.concat(dones, dim=1)
        )

    # TODO: config에 맞게 수정
    def _calc_loss(self, transitions: Transition, dynamics: Dynamics) -> Tuple[torch.Tensor, InfoDict]:
        next_obs, rew, done = self.predict(transitions.observation, transitions.action, dynamics)
        obs_loss = F.mse_loss(next_obs, transitions.next_observation)
        rew_loss = F.mse_loss(rew.squeeze(), transitions.reward)
        done_loss = F.binary_cross_entropy(done.squeeze(), transitions.terminal)

        loss = obs_loss + rew_loss + done_loss
        return loss, {'wm/loss': loss.item(), 'wm/obs_loss': obs_loss.item(), 'wm/rew_loss': rew_loss.item(), 'wm/done_loss': done_loss.item()}


    def calc_wm_loss(self, pred_traj: Trajectory, true_traj: Trajectory, **kwargs) -> Tuple[torch.Tensor, InfoDict]:
        obs_loss = F.mse_loss(pred_traj.observations[:, 1:], true_traj.observations[:, 1:])
        rew_loss = F.mse_loss(pred_traj.rewards, true_traj.rewards)
        done_loss = F.binary_cross_entropy(pred_traj.terminals, true_traj.terminals)

        loss = obs_loss.mean() + rew_loss.mean() * 1.0 + done_loss.mean() * 1.0

        return loss, {'wm/loss': loss.item(), 'wm/obs_loss': obs_loss.item(), 'wm/rew_loss': rew_loss.item(), 'wm/done_loss': done_loss.item()}



class SSMModel(WorldModel):
    name = "ssm_wm"
    def __init__(self, config: OmegaConf):
        super().__init__(config)

        self.config = config

        self.encoder = construct_mlp(
            [shape_flattener(config.observation_shape),] + config.world_model.enc_arch, activation=nn.SiLU
        )
        self.mlp = construct_mlp([
                config.world_model.hidden_dim + shape_flattener(config.action_shape),
            ] + config.world_model.net_arch, activation=nn.SiLU
        )

        self.optim: torch.optim.Optimizer = eval(config.world_model.optim_cls)(self.parameters(), **config.world_model.optim_kwargs)


    def forward(self, state: State, action: Action) -> Tuple[State, Observation, Reward, Done]:
        # State (B, #obs) + Action (B, #act)
        next_s = self.mlp(torch.concatenate((state, action), dim=1))

        return next_s, self.recon_head(next_s), self.reward_head(next_s), self.done_head(next_s)
    
    
    def encode(self, observation: Observation) -> State:
        return self.encoder(observation)


    # TODO: wrong
    def predict(self, state: State, action: Action) -> Tuple[State, Reward]:
        # State (B, #obs) + Action (B, #act)
        return self.forward(state, action)


    def rollout(self, policy: Callable[[State], Action], state: State, steps: int) -> Trajectory:
        batch_size = state.shape[0]
        dtype = state.dtype

        traj_state = torch.empty((batch_size, steps + 1, *state.shape[1:]), dtype=dtype)
        traj_action = torch.empty((batch_size, steps, shape_flattener(self.config.action_shape)), dtype=dtype)
        traj_reward = torch.empty((batch_size, steps), dtype=dtype)

        traj_state[:, 0] = state

        for step in range(steps):
            action = policy(traj_state[:, step])
            traj_action[:, step] = action
            next_state, reward = self.predict(traj_state[:, step], traj_action[:, step])
            traj_state[:, step + 1] = next_state
            traj_reward[:, step] = reward

        return (traj_state, traj_action, traj_reward, None, None)



class EnsembleTransitionMLP(nn.Module):
    def __init__(self, config: OmegaConf, n_ensemble:int = 50):
        nn.Module.__init__(self)
        self.config = config
        self.n_ensemble = n_ensemble

        self.nets = None #nn.ModuleList(
        #    (TransitionMLPModel(config) for _ in range(n_ensemble))
        #)
        self.optim: torch.optim.Optimizer = eval(config.world_model.optim_cls)(self.parameters(), **config.world_model.optim_kwargs)


    def forward(self, state: State, action: Action) -> Tuple[State, Reward]:
        # State (B, #obs) + Action (B, #act)  ->  (B, #E, #obs),  (B, #E)
        state_action = torch.concatenate((state, action), dim=1)

        preds = torch.empty(
            (state_action.shape[0], self.n_ensemble, state.shape[1] + 1), dtype=state.dtype, device=state.device
        )

        for ens_idx in range(self.n_ensemble):
            pred = self.nets[ens_idx].mlp(state_action)
            preds[:, ens_idx, :] = pred

        return preds[..., :-1], preds[..., -1]


    def predict(self, state: State, action: Action) -> Tuple[State, Reward]:
        return self.forward(state, action)


    def rollout(self, policy: Policy, state: State, steps: int) -> Trajectory:
        batch_size = state.shape[0]
        dtype = state.dtype
        device = state.device

        traj_state = torch.empty((batch_size, self.n_ensemble, steps + 1, *state.shape[1:]), dtype=dtype, device=device)
        traj_action = torch.empty((batch_size, self.n_ensemble, steps, shape_flattener(self.config.action_shape)), dtype=dtype, device=device)
        traj_reward = torch.empty((batch_size, self.n_ensemble, steps), dtype=dtype, device=device)

        traj_state[:, :, 0] = state

        for step in range(steps):
            action = policy(traj_state[:, step])
            traj_action[:, :, step] = action
            next_state, reward = self.predict(traj_state[:, step], traj_action[:, step])
            traj_state[:, :, step + 1] = next_state
            traj_reward[:, :, step] = reward

        return (traj_state, traj_action, traj_reward, None, None)


    def skill_rollout(self, skill: Sequence[Action], state: State, **kwargs) -> Trajectory:
        pass
