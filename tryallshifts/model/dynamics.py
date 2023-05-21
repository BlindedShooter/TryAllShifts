import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from tryallshifts.common.types import Sequence, Transition, Dynamics, Trajectory, Observation
from tryallshifts.model.net import construct_mlp
from tryallshifts.utils.misc import shape_flattener



class DynamicsEncoder:
    def __init__(self, config: OmegaConf):
        self.config = config
    

    def transitions_encode(self, transitions: Sequence[Transition]) -> Dynamics:  # (B, Obs), (B, Act), (B, rew=1), (B, done=1), (B, Obs) -> (Dyn,)
        raise NotImplementedError
    
    def trajectory_encode(self, trajectories: Trajectory) -> Dynamics:  # (B, T, Obs), (B, T, Act), (B, T, rew=1), (B, T, done=1), (B, T, Obs) -> (Dyn,)
        raise NotImplementedError



class MLPDynamicsEncoder(DynamicsEncoder, nn.Module):
    def __init__(self, config: OmegaConf):  # (B, ...)
        DynamicsEncoder.__init__(self, config)
        nn.Module.__init__(self)

        self.dyn_enc = construct_mlp(
            [2 * shape_flattener(config.observation_shape)+ shape_flattener(config.action_shape) + 2,]\
            + config.dyn_encoder.arch\
            + [config.dyn_encoder.dyn_dim,],
            nn.SiLU
        )
    

    def transitions_encode(self, transitions: Transition) -> Dynamics: 
        batch_size = transitions.observation.shape[0]  # Technically not batches
        encoded_transitions = self.dyn_enc(torch.concat((
            transitions.observation.view(batch_size, -1),
            transitions.action.view(batch_size, -1),
            transitions.reward.view(batch_size, -1),
            transitions.terminal.view(batch_size, -1),
            transitions.next_observation.view(batch_size, -1)), dim=1
        ))  # (B, Dyn)

        return torch.mean(encoded_transitions, dim=0)  # mean??
    

    def trajectory_encode(self, trajectories: Trajectory) -> Dynamics:  
        return super().trajectory_encode()

