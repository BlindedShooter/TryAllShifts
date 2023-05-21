from typing import Tuple, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from tryallshifts.common.types import *
from tryallshifts.model.net import construct_mlp
from tryallshifts.utils.misc import shape_flattener



class WorldModel(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
    
    def predict(self, observation: Observation, action: Action) -> Tuple[Observation, Reward, Done]: ...
    def rollout(self, observation: Observation, policy: Policy) -> Tuple[TrajObservation, TrajActionDist, TrajReward, TrajDone]: ...
    def rollout_actions(self, observation: Observation, actions: TrajAction) -> Tuple[TrajObservation, TrajReward, TrajDone]: ...
    def train(self, observation: Observation, action: Action, reward: Reward, next_observation: Observation) -> InfoDict: ...
    def train_traj(self, observations: TrajObservation, actions: TrajAction, 
                    rewards: TrajReward, next_observations: TrajObservation
    ) -> InfoDict: ...


class SSMWorldModel(WorldModel):
    def encode_states(self, obs: Observation) -> State: ...
    def decode_states(self, states: State) -> Observation: ...
    def rssm_imagine(self, prev_rssm_state: State, prev_action: Action, done: Done) -> State: ...
    def rssm_rollout(self, prev_rssm_state: State, horizon: int, actor: nn.Module
    ) -> Tuple[TrajState, TrajActionDist, TrajReward, TrajDone]: ...
    def rssm_rollout_actions(self, prev_rssm_state: State) -> Tuple[TrajState, TrajReward, TrajDone]: ...



class DynamicsWorldModel(WorldModel):
    def predict(self, observation: Observation, action: Action, dynamics: Dynamics
    ) -> Tuple[Observation, Reward, Done]: ...
    def rollout(self, observation: Observation, policy: Policy, dynamics: Dynamics
    ) -> Tuple[TrajObservation, TrajActionDist, TrajReward, TrajDone]: ...
    def rollout_actions(self, observation: Observation, actions: TrajAction, dynamics: Dynamics
    ) -> Tuple[TrajObservation, TrajReward, TrajDone]: ...
    def train(self, observation: Observation, action: Action, reward: Reward, 
                next_observation: Observation, dynamics: Dynamics
    ) -> InfoDict: ...
    def train_traj(self, observations: TrajObservation, actions: TrajAction, rewards: TrajReward, 
                next_observations: TrajObservation, dynamics: Dynamics
    ) -> InfoDict: ...


class DynamicsSSMWorldModel(DynamicsWorldModel):
    def encode_states(self, obs: Observation) -> State: ...
    def decode_states(self, states: State) -> Observation: ...
    def rssm_imagine(self, prev_rssm_state: State, dynamics: Dynamics, prev_action: Action, done: Done) -> State: ...
    def rssm_rollout(self, prev_rssm_state: State, dynamics: Dynamics, horizon: int, actor: nn.Module
    ) -> Tuple[TrajState, TrajActionDist, TrajReward, TrajDone]: ...
    def rssm_rollout_actions(self, prev_rssm_state: State) -> Tuple[TrajState, TrajReward, TrajDone]: ...



class MLPWorldModel(WorldModel):
    def __init__(self, config: OmegaConf):
        super().__init__(config)

        self.enc = construct_mlp(
            [
                shape_flattener(config.observation_shape) + shape_flattener(config.action_shape),
            ] + config.world_model.enc_arch, 
            activation=nn.SiLU
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
    

    def forward(self, observation: Observation, action: Action) -> Tuple[Observation, Reward, Done]:
        batch_size = observation.shape[0]
        state = self.enc(torch.concat((observation.reshape(batch_size, -1), action.reshape(batch_size, -1)), dim=1))

        return self.recon_head(state).reshape(batch_size, *self.config.observation_shape), self.reward_head(state), self.done_head(state)
    

    def predict(self, observation: Observation, action: Action) -> Tuple[Observation, Reward, Done]:
        return self.forward(observation, action)
    

    def rollout(self, observation: Observation, policy: ObservationPolicy, steps: int=8):
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
        
        return observations, actions, rewards, dones


    def rollout_actions(self, observation: Observation, actions: TrajAction) -> Tuple[TrajObservation, TrajReward, TrajDone]:
        observations = [observation,]
        rewards = []
        dones = []

        batch_size = observation.shape[0]

        for action in actions:
            state = self.enc(torch.concat((observations[-1].view(batch_size, -1), action.view(batch_size, -1)), dim=1))
            observations.append(self.recon_head.forward(state).view(batch_size, *self.config.observation_shape))
            rewards.append(self.reward_head.forward(state))
            dones.append(self.done_head.forward(state))
        
        return observations, rewards, dones



class DynamicsMLPWorldModel(WorldModel):
    def __init__(self, config: OmegaConf):
        super().__init__(config)
        self.enc = construct_mlp(
            [
                config.dyn_encoder.dyn_dim + shape_flattener(config.observation_shape) + shape_flattener(config.action_shape),
            ] + config.world_model.enc_arch, 
            activation=nn.SiLU
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


    def forward(self, observation: Observation, action: Action, dynamics: Dynamics) -> torch.Tensor:
        """
        observation: (B, Obs)
        action: (B, Act)
        dynamics: (B, Dyn)
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
    def rollout(self, observation: Observation, policy: ObservationPolicy, dynamics: Dynamics, steps: int = 8):
        observations = [observation,]
        actions = [policy(observation),]
        rewards = []
        dones = []

        batch_size = observation.shape[0]

        for _ in range(steps - 1):
            state = self.forward(observations[-1], actions[-1], dynamics)
            observations.append(self.recon_head.forward(state).view(batch_size, *self.config.observation_shape))
            actions.append(policy(observations[-1]).view(batch_size, *self.config.action_shape))
            rewards.append(self.reward_head.forward(state))
            dones.append(self.done_head.forward(state))
        
        return observations, actions, rewards, dones


    def rollout_actions(self, observation: Observation, actions: TrajAction, dynamics: Dynamics) -> Tuple[TrajObservation, TrajReward, TrajDone]:
        observations = [observation,]
        rewards = []
        dones = []

        batch_size = observation.shape[0]

        for action in actions:
            state = self.forward(observations, actions, dynamics)
            observations.append(self.recon_head.forward(state).view(batch_size, *self.config.observation_shape))
            rewards.append(self.reward_head.forward(state))
            dones.append(self.done_head.forward(state))
        
        return observations, actions, rewards, dones
    
    


class SSMModel(WorldModel):
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
