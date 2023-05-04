from typing import Tuple, Iterable
from functools import reduce
import operator

import numpy as np
import gym



def shape_flattener(shape: Tuple[int]) -> int:
    return reduce(operator.mul, shape, 1)


def d4rl_visualize(env: gym.Env, obs: Iterable[np.ndarray]):
    pass