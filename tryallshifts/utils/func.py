import torch
import numpy as np


def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log(np.abs(x) + 1)
