import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from tryallshifts.common.config import TASConfig
from tryallshifts.common.types import Sequence, Transition, Dynamics, Trajectory, Observation, Union
from tryallshifts.model.net import construct_mlp
from tryallshifts.utils.misc import shape_flattener



class DynamicsEncoder(nn.Module):
    name: str = None
    optim: torch.optim.Optimizer

    def __init__(self, config: TASConfig):
        super().__init__()
        self.config = config


    def _transition_concat(self, trs: Union[Transition, Trajectory]) -> torch.Tensor:
        to_concat = []

        trs_view = trs.flattened_view()

        for k in self.config.dyn_encoder.concat:
            to_concat.append(trs_view[k])
        
        return torch.concatenate(to_concat, dim=-1)


    def transitions_encode(self, transitions: Transition) -> Dynamics:  
        # (..., Obs), (..., Act), (..., rew=1), (..., done=1), (..., Obs) -> (..., Dyn,)
        raise NotImplementedError
    
    def trajectory_encode(self, trajectories: Trajectory) -> Dynamics:  
        # (..., T, Obs), (..., T, Act), (..., T, rew=1), (..., T, done=1), (..., T, Obs) -> (..., Dyn,)
        raise NotImplementedError



class MLPDynamicsEncoder(DynamicsEncoder):
    name = "mlp_de"
    def __init__(self, config: TASConfig):  # (B, ...)
        super().__init__(config)

        self.dyn_enc = construct_mlp(
            [2 * shape_flattener(config.observation_shape)+ shape_flattener(config.action_shape) + 2,]\
            + config.dyn_encoder.arch\
            + [config.dyn_encoder.dyn_dim,],
            nn.SiLU
        )

        self.optim = eval(config.dyn_encoder.optim_cls)(self.parameters(), **config.dyn_encoder.optim_kwargs)
    

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



class RNNDynamicsEncoder(DynamicsEncoder):
    name = "rnn_de"
    def __init__(self, config: TASConfig):
        super().__init__(config)

        self.hidden = nn.Parameter(torch.zeros(
                (len(config.dyn_encoder.arch), config.batch_size, config.dyn_encoder.dyn_dim,)
            ), requires_grad=config.dyn_encoder.learnable_init_state
        )  # start from true dynamics = [0, 0, ...]?

        self.rnn = nn.RNN(
            input_size=config.dyn_encoder.inp_size,
            hidden_size=config.dyn_encoder.dyn_dim, nonlinearity="relu",
            num_layers=len(config.dyn_encoder.arch), batch_first=True
        )

        self.optim = eval(config.dyn_encoder.optim_cls)(self.parameters(), **config.dyn_encoder.optim_kwargs)

    # 정보가 리니어하게 늘어난다고 생각?
    def trajectory_encode(self, trajectories: Trajectory) -> Dynamics:
        inp = self._transition_concat(trajectories)
        rnn_out, _ = self.rnn(inp, self.hidden)

        return rnn_out  # (B, T, D)
