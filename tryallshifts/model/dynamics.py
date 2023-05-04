import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from tryallshifts.common.types import *


class DynamicsEncoder(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
    
    def transitions_encode(obs: Observation) -> Dynamics: ...
    def trajectory_encode(traj_obs: TrajObservation) -> Dynamics: ...



class MLPDynamicsEncoder(DynamicsEncoder):
    def __init__(self, config: OmegaConf):
        super().__init__(config)

