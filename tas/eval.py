import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from tas.typing import *
from tas.utils import *
from tas.model import *
from tas.losses import *


# 이거 그냥 loss 계산 아님?
def eval_wm_on_traj(wm: WorldModel, traj: Trajectory) -> InfoDict:
    obs, act, next_obs, rew, dynamics = traj.observations, traj.actions, traj.next_observations, traj.rewards, traj.dynamics
    if isinstance(dyn, DynEncoder):
        dyn = dyn(traj)

    next_obs_pred, rew_pred, done_pred = wm(obs, act, dynamics=dynamics)
    dyn_loss = F.mse_loss(next_obs_pred, next_obs)
    rew_loss = F.mse_loss(rew_pred, rew)
    wm_loss = dyn_loss + rew_loss

    return {'wm_loss': wm_loss.item(), 'dyn_loss': dyn_loss.item(), 'rew_loss': rew_loss.item()}


@torch.no_grad()
def eval_wm_on_traj_oracledyn(wm: WorldModel, dyn_enc: DynEncoder, traj: Trajectory):
    obs, act, next_obs, rew, done = traj.observations, traj.actions, traj.next_observations, traj.rewards, traj.dones
    dynamics = dyn_enc(traj).unsqueeze(0).repeat(obs.shape[0], 1, 1)

    next_obs_pred, rew_pred, done_pred = wm(obs, act, dynamics=dynamics)
    dyn_loss = F.mse_loss(next_obs_pred, next_obs)
    rew_loss = F.mse_loss(rew_pred, rew)
    wm_loss = dyn_loss + rew_loss

    return {'wm_loss': wm_loss.item(), 'dyn_loss': dyn_loss.item(), 'rew_loss': rew_loss.item()}



def get_oracle_traj(env: SettableEnv, imag_traj: Trajectory, policy: Actor, horizon: int = 50) -> Trajectory:
    init_obs = env.reset()
    imag_traj = seq_to_numpy(imag_traj)

    true_obss = [np.squeeze(imag_traj.observations[0])]
    true_rews = []

    for obs, act in zip(imag_traj.observations, imag_traj.actions):
        obs, act = np.squeeze(obs), np.squeeze(act)
        mjenv_set_state(env, obs)
        true_next_obs, true_reward, done, _ = env.step(act)
        # done 처리?
        true_obss.append(true_next_obs)
        true_rews.append(true_reward)

    true_obss = np.stack(true_obss, axis=0)
    true_rews = np.stack(true_rews, axis=0)

    return Trajectory(
        all_observations=true_obss,
        actions=imag_traj.actions,
        rewards=true_rews,
        dones=imag_traj.dones,
    )


def eval_wm_on_env(wm: WorldModel, dynamics: Dynamics, env: gym.Env, policy: Actor,
                   eval_fns: Callable[[Trajectory, Trajectory], InfoDict], horizon: int=50) -> InfoDict:
    init_obs = env.reset()  # VecEnv?
    wm_device = next(wm.parameters()).device
    
    imag_traj = wm.rollout_policy(tensorify(init_obs, device=wm_device).unsqueeze(0), policy, dynamics=dynamics, horizon=horizon)
    true_traj = get_oracle_traj(env, imag_traj, policy, horizon=horizon)

    info = {}
    for fn in eval_fns:
        info.update(fn(imag_traj, true_traj))
    
    return info


def eval_policy_on_env(policy: Actor, env: gym.Env) -> InfoDict:
    # dynenc 쓰도록 해야함
    obs, done = env.reset(), False
    obss = [obs]
    acts = []
    rews = []
    steps = 0

    while not done:
        act = policy(obss[-1])
        obs, rew, done, _ = env.step(act)
        obss.append(obs)
        acts.append(act)
        rews.append(rew)
        steps += 1

    return {'true_steps': steps, 'true_return': sum(rews), 'NOLOG-true_obss': obss, 'NOLOG-true_acts': acts, 'NOLOG-true_rews': rews}


# initial state dist.도 학습을 해야하나?
def eval_policy_on_wm(policy: Actor, wm: WorldModel, dynamics: Dynamics, env: gym.Env,
                       done_func: Callable[[Observation], Done]) -> InfoDict:
    obs = env.reset()
    wm_device = next(wm.parameters()).device

    obss = [tensorify(obs, device=wm_device).unsqueeze(0)]
    acts = []
    rews = []
    steps = 0

    while not done_func(obss[-1]).any().item():
        act = policy(obss[-1], dynamics=dynamics)
        obs, rew = wm.forward(obss[-1], act, dynamics=dynamics)
        obss.append(obs)
        acts.append(act)
        rews.append(rew)
        steps += 1

    return {'imag_steps': steps, 'imag_return': sum(rews), 'NOLOG-imag_obss': obss, 
            'NOLOG-imag_acts': acts, 'NOLOG-imag_rews': rews}


# Trajectory evaluation functions
def calc_traj_mse(imag_traj: Trajectory, true_traj: Trajectory) -> InfoDict:
    info = {}
    imag_traj = seq_to_numpy(imag_traj)
    true_traj = seq_to_numpy(true_traj)

    step_dyn_mse = ((np.squeeze(imag_traj.next_observations) - np.squeeze(true_traj.next_observations)) ** 2).mean(axis=-1)
    step_rew_mse = ((np.squeeze(imag_traj.rewards) - np.squeeze(true_traj.rewards)) ** 2)

    info['dyn_mse_mean'] = step_dyn_mse.mean()
    info['dyn_mse_std'] = step_dyn_mse.std()
    info['rew_mse_mean'] = step_rew_mse.mean()
    info['rew_mse_std'] = step_rew_mse.std()

    if len(step_dyn_mse.shape) == 2:  # (T, B)
        info['dyn_mse_min'] = step_dyn_mse.min(axis=0).mean()
        info['dyn_mse_max'] = step_dyn_mse.max(axis=0).mean()
        info['rew_mse_min'] = step_rew_mse.min(axis=0).mean()
        info['rew_mse_max'] = step_rew_mse.max(axis=0).mean()
    else:  # (T,)
        info['dyn_mse_min'] = step_dyn_mse.min()
        info['dyn_mse_max'] = step_dyn_mse.max()
        info['rew_mse_min'] = step_rew_mse.min()
        info['rew_mse_max'] = step_rew_mse.max()

    info['NOLOG-dyn_mse_step'] = step_dyn_mse
    info['NOLOG-rew_mse_step'] = step_rew_mse

    return info
