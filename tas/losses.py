from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

from tas.typing import *
from tas.configs import *
from tas.utils import *
from tas.model import *

######################
# World Model Losses #
######################
def calc_wm_loss(tran: Transition, wm: WorldModel, dynamics: Dynamics) -> Tuple[torch.Tensor, InfoDict]:
    obs, act, next_obs, rew, done = tran.observations, tran.actions, tran.next_observations, tran.rewards, tran.dones
    
    next_obs_pred, rew_pred, done_pred, _ = wm(obs, act, dynamics=dynamics)
    dyn_loss = F.mse_loss(next_obs_pred, next_obs)
    rew_loss = F.mse_loss(rew_pred, rew)
    wm_loss = dyn_loss + rew_loss

    return wm_loss, {'wm_loss': wm_loss.item(), 'dyn_loss': dyn_loss.item(), 'rew_loss': rew_loss.item()}


def calc_stoch_wm_loss(tran: Transition, wm: WorldModel, dynamics: Dynamics) -> Tuple[torch.Tensor, InfoDict]:
    obs, act, next_obs, reward, done = tran.observations, tran.actions, tran.next_observations, tran.rewards, tran.dones
    
    next_obs_pred, rew_pred, done_pred,\
        (next_obs_mu, next_obs_logvar, reward_mu, reward_logvar) = wm(obs, act, dynamics=dynamics)
    dyn_loss = gaussian_mse_loss(next_obs_mu, next_obs_logvar, next_obs)
    rew_loss = gaussian_mse_loss(reward_mu.squeeze(), reward_logvar.squeeze(), reward)
    logvar_loss = 0.01 * wm.obs_max_logvar.sum() - 0.01 * wm.obs_min_logvar.sum() + 0.01 * wm.reward_max_logvar.sum() - 0.01 * wm.reward_min_logvar.sum()

    wm_loss = dyn_loss + rew_loss + logvar_loss
    
    with torch.no_grad():
        dyn_mse_loss = F.mse_loss(next_obs_pred, next_obs)
        rew_mse_loss = F.mse_loss(rew_pred.squeeze(), reward)
    
    return wm_loss, {'wm_loss': wm_loss.item(), 'dyn_loss': dyn_loss.item(), 'rew_loss': rew_loss.item(),
                      'dyn_mse_loss': dyn_mse_loss.item(), 'rew_mse_loss': rew_mse_loss.item(), 'logvar_loss': logvar_loss.item()}


def calc_adaptive_wm_loss(traj: Trajectory, wm: WorldModel, dyn_enc: DynEncoder, stoch: bool=False) -> Tuple[torch.Tensor, InfoDict]:
    wm_loss = 0.0
    infos = defaultdict(list)

    for i in range(1, len(traj)):
        with torch.no_grad:
            dyn = dyn_enc(traj[:i])  # (B, D)

        if stoch:
            loss, info = calc_stoch_wm_loss(traj[i], wm, dyn)
        else:
            loss, info = calc_wm_loss(traj[i], wm, dyn)
        
        wm_loss += loss
        for k, v in info.items():
            infos[k].append(v)
    
    for k, v in infos.items():
        infos[k] = sum(v) / len(v)
    
    return wm_loss, infos


###########################
# Dynamics Encoder Losses #
###########################
def calc_dyn_enc_randdyn_loss(
        traj: Trajectory, dyn_enc: DynEncoder, wm: WorldModel, actor: ActorType,
        true_dyn: torch.Tensor, deterministic: bool=False, train_wm: bool=False, policy_imagine: bool=False
    ) -> Tuple[torch.Tensor, InfoDict]:
    rand_dynamics = avg_l1_norm(torch.empty((
        1, traj.observations.shape[1], dyn_enc.config.dyn_dim
    ), requires_grad=False, device=dyn_enc.config.device).normal_(-1, 1).expand(traj.observations.shape[0], -1, -1))

    if policy_imagine:
        imag_traj, _ = wm.rollout_policy(
            traj.observations[0], actor.predict, dynamics=rand_dynamics, horizon=len(traj), deterministic=True
        )
    else:
        imag_traj, _ = wm.rollout_actions(
            traj.observations[0], traj.actions, dynamics=rand_dynamics, horizon=len(traj), deterministic=True
        )
    #traj = Trajectory(traj.all_observations, traj.actions, traj.rewards, traj.dones, dynamics=true_dyn)
    traj.dynamics = true_dyn
    
    imag_traj = Trajectory(
        imag_traj.all_observations.detach(), imag_traj.actions.detach(), 
        imag_traj.rewards.detach(), imag_traj.dones.detach(), dynamics=imag_traj.dynamics.detach()
    )
    cat_traj = traj.torch_cat(imag_traj)

    pred_dyn = dyn_enc(cat_traj)
    dyn_enc_loss = F.mse_loss(pred_dyn, true_dyn[-1])  # predict only the last dynamics. (hmm)

    return dyn_enc_loss, {'dyn_enc_loss': dyn_enc_loss.item()}


def calc_dyn_enc_persistent_loss(
        traj: Trajectory, dyn_enc: DynEncoder, wm: WorldModel, policies: Sequence[ActorType], 
        deterministic: bool=False, train_wm: bool=True, train_wm_sup: bool=True
    ) -> Tuple[torch.Tensor, InfoDict]:
    traj1, traj2 = split_trajectory(traj)
    info = {}

    # no_grad needed?
    with torch.no_grad():
        target_dyn = dyn_enc(traj1)  # (B, D)

    traj2.dynamics = target_dyn.expand(len(traj2), -1, -1)

    if train_wm_sup:
        wm_loss, wm_info = calc_stoch_wm_loss(traj2, wm, traj2.dynamics)
        info.update(('persist_dyn_' + k, v) for k, v in wm_info.items())
    else:
        wm_loss = 0.0
    
    rollouts = []
    with torch.set_grad_enabled(train_wm):
        for policy in policies:
            rollouts.append(wm.rollout_policy(
                traj2.all_observations[0], policy, dynamics=target_dyn.expand(len(traj2), -1, -1),
                horizon=len(traj2), deterministic=deterministic
            ))
    imag_traj: Trajectory = stack_trajectory(rollouts + [traj2,])

    imag_pred_dyns = dyn_enc(imag_traj)
    dyn_enc_loss = F.mse_loss(imag_pred_dyns, imag_traj.dynamics[-1])  # predict only the last dynamics. (hmm)

    loss = dyn_enc_loss + wm_loss
    info['dyn_enc_loss'] = dyn_enc_loss.item()
    
    return loss, info


################
# Actor Losses #
################
def calc_actor_bc_loss(tran: Transition, actor: Actor) -> Tuple[torch.Tensor, InfoDict]:
    obs, act = tran.observations, tran.actions

    act_pred = actor.rsample(obs, dynamics=tran.dynamics)
    act_loss = F.mse_loss(act_pred, act)

    return act_loss, {'act_bc_loss': act_loss.item()}


def calc_actor_loss(tran: Transition, actor: Actor, critic: Critic, 
               log_alpha: Optional[torch.Tensor], entropy_bonus: bool=True) -> Tuple[torch.Tensor, InfoDict]:
    obs = tran.observations

    # Update Actor
    pred_act, log_prob = actor.get_action_w_prob(obs, dynamics=tran.dynamics)
    
    if entropy_bonus:  # this can be inferred from log_alpha is None...? don't think torch will compile that
        entropy = -log_alpha.exp().detach() * log_prob
    else:
        entropy = 0.0

    min_q, _ = critic(obs, pred_act, dynamics=tran.dynamics).min(dim=-1)
    actor_loss = (-min_q - entropy).mean()

    return actor_loss, {'actor_loss': actor_loss.item(), 'NOLOG-log_prob_w_grad': log_prob}


def calc_critic_loss(tran: Transition, actor: Actor, critic: Critic, target_critic: Critic, gamma: float,
                log_alpha: Optional[torch.Tensor], entropy_bonus: bool=True) -> Tuple[torch.Tensor, InfoDict]:
    obs, act, next_obs, rew, done = tran.observations, tran.actions, tran.next_observations, tran.rewards, tran.dones

    with torch.no_grad():
        next_act, log_prob = actor.get_action_w_prob(next_obs, dynamics=tran.dynamics)
        if entropy_bonus:
            entropy = -log_alpha.exp().detach() * log_prob
        else:
            entropy = 0.0
        next_q, _ = target_critic(next_obs, next_act, dynamics=tran.dynamics).min(dim=-1)
        target_q = rew + (1 - done) * gamma * (next_q + entropy)

    q = critic(obs, act, dynamics=tran.dynamics)
    critic_loss = F.smooth_l1_loss(q, target_q.unsqueeze(-1).expand_as(q))

    return critic_loss, {'critic_loss': critic_loss.item(), 'q': q.mean().item(), 'target_q': target_q.mean().item()}


def calc_alpha_loss(log_alpha: torch.Tensor, log_prob: torch.Tensor, target_entropy: float) -> Tuple[torch.Tensor, InfoDict]:
    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()

    return alpha_loss, {'alpha_loss': alpha_loss.item()}