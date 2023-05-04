from typing import Optional
import numpy as np
import torch
from tryallshifts.common.types import *



class TransitionBuffer:
    def __init__(self, backend="numpy"):
        pass


    def store(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        timeout: Optional[Done] = None,
        terminal: Optional[Done] = None
    ):
        pass


    def sample(self, n_samples: int) -> Transition:
        pass


class TASEnsembleTransitionBuffer:
    def __init__(
        self,
        backend="numpy",
        n_models: int = 1
    ):
        self.backend = backend
        pass

    def store_ensemble():
        pass
