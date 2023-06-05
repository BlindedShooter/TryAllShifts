from typing import Tuple, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tryallshifts.common.config import Tuple

from tryallshifts.common.types import *
from tryallshifts.common.config import *
from tryallshifts.common.types import ActionDist, InfoDict, Tuple
from tryallshifts.model.net import construct_mlp
from tryallshifts.utils.misc import shape_flattener



class TASPolicy(nn.Module):
    name: Optional[str] = None

    def __init__(self, config: TASConfig) -> None:
        super().__init__()
        self.config = config
    
    def forward(self): ...
    def predict(self, transition: Transition) -> Tuple[ActionDist, InfoDict]: ...
    def train_transition(self, transition: Transition) -> InfoDict: ...
    def train_trajectory(self, trajectory: Trajectory) -> InfoDict: ...


class TASMLPPolicy(TASPolicy):
    name = "mlp_po"
    def __init__(self, config: TASConfig) -> None:
        super().__init__(config)

        self.value_fn = construct_mlp(
            [shape_flattener(config.observation_shape),] \
                + config.policy.value_arch\
                + [1,]
        )
        self.actor = construct_mlp(
            [shape_flattener(config.observation_shape),] \
                + config.policy.actor_arch\
                + [shape_flattener(config.action_shape),]
        )
    

    def predict(self, transition: Transition) -> Tuple[ActionDist, InfoDict]:
        return super().predict()