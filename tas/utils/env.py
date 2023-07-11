import gym
import numpy as np
import torch


def mjenv_set_state(env: gym.Env, state: np.ndarray):
    nq, nv = env.model.nq, env.model.nv

    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()
    state = np.squeeze(state)

    if len(state) == nq + nv:
        qpos = state[:-nv]
        qvel = state[-nv:]
    elif len(state) == nq + nv - 1:
        qpos = np.concatenate(((0,), state[:-nv]))
        qvel = state[-nv:]
    else:
        raise ValueError(f"Invalid state shape: {state.shape}")
    
    env.set_state(qpos, qvel)