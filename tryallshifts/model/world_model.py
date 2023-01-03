from typing import Tuple, Callable
from abc import ABCMeta, abstractmethod
import numpy as np
from tryallshifts.common.types import Transition, Trajectory


class WorldModel(metaclass=ABCMeta):
    @abstractmethod
    def predict(
        self,
        state: np.ndarray[np.float32],
        action: np.ndarray[np.float32],
        **kwargs
        ) -> Tuple[np.ndarray[np.float32], np.float32]:
        """
        a
        """


    @abstractmethod
    def rollout(
        self,
        policy: Callable[[np.ndarray[np.float32]], np.ndarray[np.float32]],
        transition: Transition,
        steps: int,
        **kwargs
        ) -> Trajectory:

        """
        """
