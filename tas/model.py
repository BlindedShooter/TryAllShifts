from dataclasses import asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tas.configs import DynEncoderConfig

from tas.typing import *
from tas.configs import *
from tas.utils import *
from tas.distributions.continuous import TanhNormal, IndependentNormal
import tas.net as net
from tas.utils.done_funcs import DONE_FUNCS, not_done_func


#maybe_compile = torch.compile
maybe_compile = lambda x: x  # compile makes things slower on Titan RTX


class Actor(nn.Module):
    def forward(self, obs: torch.Tensor, dynamics: torch.Tensor) -> ActionDist: ...
    def get_action_w_prob(self, obs: torch.Tensor, dynamics: torch.Tensor) -> Tuple[Action, torch.Tensor]: ...
    def rsample(self, obs: torch.Tensor, dynamics: torch.Tensor) -> Action: ...
    def sample(self, obs: torch.Tensor, dynamics: torch.Tensor) -> Action: ...



class StochActor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.config = config
        self.num_dist_params: Final[int] = config.num_dist_params  # Typically 2 for Gaussian mean and std
        self.net: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(net, config.net_config.net_cls)(**asdict(config.net_config))
        self.distribution = TanhNormal(
            loc=torch.zeros(1), scale=torch.ones(1), upscale=self.config.upscale,
            min=self.config.min, max=self.config.max, event_dims=self.config.event_dims, tanh_loc=True
        )

    @maybe_compile
    def forward(self, obs: torch.Tensor, dynamics: torch.Tensor) -> ActionDist:
        mu, log_std = self.net(obs, dynamics).chunk(self.num_dist_params, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)  # clipping as in EDAC paper, same as SAC-RND
        self.distribution.update(loc=mu, scale=log_std.exp())

        return self.distribution


    def get_action_w_prob(self, obs: torch.Tensor, dynamics: torch.Tensor) -> Tuple[Action, torch.Tensor]:
        dist = self.forward(obs, dynamics)
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    

    def rsample(self, obs: torch.Tensor, dynamics: torch.Tensor) -> Action:
        self.forward(obs, dynamics)
        return self.distribution.rsample()
    

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, dynamics: torch.Tensor, deterministic: bool = False) -> Action:
        self.forward(obs, dynamics)

        if deterministic:
            return self.distribution.loc
        else:
            return self.distribution.sample()  # 어차피 여기서 grad 끊어지나..?



class Critic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        self.critics = nn.ModuleList([getattr(net, config.net_config.net_cls)(**asdict(config.net_config)) for _ in range(self.config.num_critics)])
    
    @maybe_compile
    def forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor] = None) -> Value:
        return torch.cat([critic(torch.cat([obs, act], dim=-1), dynamics) for critic in self.critics], dim=-1)


    def single_forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor] = None, idx: int = 0) -> Value:
        return self.critics[idx](torch.cat([obs, act], dim=-1), dynamics)



class DynModel(nn.Module):
    def __init__(self, config: DynModelConfig):
        super().__init__()
        self.config = config
        self.net: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(net, config.net_config.net_cls)(**asdict(config.net_config))
    
    @maybe_compile
    def forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor] = None) -> State:
        return self.net(torch.cat([obs, act], dim=-1), dynamics)



class RewardModel(nn.Module):
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        self.net: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(net, config.net_config.net_cls)(**asdict(config.net_config))
    
    @maybe_compile
    def forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor]=None) -> Reward:
        return self.net(torch.cat([obs, act], dim=-1), dynamics)



class WorldModel(nn.Module):
    obs_min_logvar: nn.Parameter
    obs_max_logvar: nn.Parameter
    reward_min_logvar: nn.Parameter
    reward_max_logvar: nn.Parameter

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        if config.env_name.split("-")[0] in DONE_FUNCS:
            self.done_func: Callable[[torch.Tensor,], torch.Tensor] = DONE_FUNCS[config.env_name.split("-")[0]]
        else:
            print(f'No done function found for {config.env_name}. Using default done function.')
            self.done_func = not_done_func

        self.dyn_model = DynModel(config.dyn_model_config)
        if config.use_separate_reward_model:
            self.reward_model = RewardModel(config.reward_model_config)

        if self.config.stochastic:
            self.obs_dist = IndependentNormal(loc=torch.zeros(1), scale=torch.ones(1))
            self.reward_dist = IndependentNormal(loc=torch.zeros(1), scale=torch.ones(1))

            obs_shape = self.config.dyn_model_config.net_config.out_dim // self.config.dyn_model_config.num_dist_params\
                        - (0 if self.config.use_separate_reward_model else 1)
            
            self.register_parameter('obs_min_logvar', nn.Parameter(torch.full((obs_shape,), -10.0, requires_grad=True, device=config.device)))
            self.register_parameter('obs_max_logvar', nn.Parameter(torch.full((obs_shape,), 0.5, requires_grad=True, device=config.device)))

            self.register_parameter('reward_min_logvar', nn.Parameter(torch.full((1,), -10.0, requires_grad=True, device=config.device)))
            self.register_parameter('reward_max_logvar', nn.Parameter(torch.full((1,), 0.5, requires_grad=True, device=config.device)))

    @maybe_compile
    def forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor] = None,
                 deterministic: bool = False) -> Tuple[State, Reward]:
        if self.config.use_separate_reward_model:
            if self.config.stochastic:
                next_obs_mu, next_obs_logvar = self.dyn_model(obs, act, dynamics=dynamics).chunk(2, dim=-1)
                reward_mu, reward_logvar = self.reward_model(obs, act, dynamics=dynamics).chunk(2, dim=-1)
                reward_mu, reward_logvar = reward_mu, reward_logvar

                next_obs_logvar = self.obs_max_logvar - F.softplus(self.obs_max_logvar - next_obs_logvar)
                next_obs_logvar = self.obs_min_logvar + F.softplus(next_obs_logvar - self.obs_min_logvar)
                reward_logvar = self.reward_max_logvar - F.softplus(self.reward_max_logvar - reward_logvar)
                reward_logvar = self.reward_min_logvar + F.softplus(reward_logvar - self.reward_min_logvar)

                self.obs_dist.update(loc=next_obs_mu, scale=next_obs_logvar.exp().sqrt())
                self.reward_dist.update(loc=reward_mu, scale=reward_logvar.exp().sqrt())

                if deterministic:
                    next_obs, reward = next_obs_mu, reward_mu
                else:
                    next_obs = self.obs_dist.sample()  # rsample?
                    reward = self.reward_dist.sample()
            else:
                next_obs_mu, next_obs_logvar = self.dyn_model(obs, act, dynamics=dynamics), 0.0
                reward_mu, reward_logvar = self.reward_model(obs, act, dynamics=dynamics), 0.0

                next_obs, reward = next_obs_mu, reward_mu
        else:
            if self.config.stochastic:
                all_mu, all_logvar = self.dyn_model(obs, act, dynamics=dynamics).chunk(2, dim=-1)
                next_obs_mu, reward_mu = all_mu[..., :-1], all_mu[..., -1:]
                next_obs_logvar, reward_logvar = all_logvar[..., :-1], all_logvar[..., -1:]
                reward_mu, reward_logvar = reward_mu, reward_logvar

                next_obs_logvar = self.obs_max_logvar - F.softplus(self.obs_max_logvar - next_obs_logvar)
                next_obs_logvar = self.obs_min_logvar + F.softplus(next_obs_logvar - self.obs_min_logvar)
                reward_logvar = self.reward_max_logvar - F.softplus(self.reward_max_logvar - reward_logvar)
                reward_logvar = self.reward_min_logvar + F.softplus(reward_logvar - self.reward_min_logvar)

                self.obs_dist.update(loc=next_obs_mu, scale=next_obs_logvar.exp().sqrt())
                self.reward_dist.update(loc=reward_mu, scale=reward_logvar.exp().sqrt())

                if deterministic:
                    next_obs, reward = next_obs_mu, reward_mu
                else:
                    next_obs = self.obs_dist.sample()
                    reward = self.reward_dist.sample()
            else:
                all_mu = self.dyn_model(obs, act, dynamics=dynamics)
                next_obs_mu, reward_mu = all_mu[:, :-1], all_mu[:, -1:]
                next_obs_logvar, reward_logvar = 0.0, 0.0

                next_obs, reward = next_obs_mu, reward_mu

        done = self.done_func(next_obs).unsqueeze(-1)

        return next_obs, reward, done, (next_obs_mu, next_obs_logvar, reward_mu, reward_logvar)
    
    
    @maybe_compile
    def rollout_actions(self, obs: torch.Tensor, actions: torch.Tensor, dynamics: Optional[torch.Tensor] = None,
                         horizon: int = 1, rew_pen: bool = False, deterministic: bool = False) -> Tuple[Trajectory, InfoDict]:
        obss, rewards, dones = [obs], [], []
        info = {}

        for i in range(horizon):
            n_obs, reward, done, _ = self.forward(obss[-1], actions[i], dynamics[i], deterministic)
            obss.append(n_obs)
            rewards.append(reward)
            dones.append(done)

        return Trajectory(
            all_observations=torch.stack(obss, dim=0),
            actions=actions,
            rewards=torch.stack(rewards, dim=0),
            dones=torch.stack(dones, dim=0),
            dynamics=dynamics
        ), info
    

    @maybe_compile
    def rollout_policy(self, obs: torch.Tensor, policy: Callable[[Observation, Dynamics], Action], 
                        dynamics: Optional[torch.Tensor] = None, horizon: int = 1,
                        rew_pen: bool = False, lam: float=1.0, deterministic: bool = False) -> Tuple[Trajectory, InfoDict]:
        obss, actions, rewards, dones = [obs], [], [], []
        info = {}
        if rew_pen:
            info['rew_pen'] = 0.0
            info['rew_pen_max'] = 0.0

        for i in range(horizon):
            action = policy(obss[-1], dynamics[i])
            n_obs, reward, done, (n_obs_mu, n_obs_logvar, rew_mu, rew_logvar) = self.forward(obss[-1], action, dynamics[i], deterministic)
            
            if rew_pen:
                mopo_pen = n_obs_logvar.exp().sqrt().norm(p=2, dim=-1, keepdim=True)
                info['rew_pen'] += mopo_pen.mean().item()
                info['rew_pen_max'] = max(info['rew_pen_max'], mopo_pen.max().item())
                reward = reward - lam * mopo_pen
            obss.append(n_obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        if rew_pen:
            info['rew_pen'] /= horizon

        return Trajectory(
            all_observations=torch.stack(obss, dim=0),
            actions=torch.stack(actions, dim=0),
            rewards=torch.stack(rewards, dim=0),
            dones=torch.stack(dones, dim=0),
            dynamics=dynamics
        ), info



class DynEncoder(nn.Module):
    def __init__(self, config: DynEncoderConfig):
        super().__init__()
        self.config = config
    
    def forward(self, traj: Trajectory) -> Dynamics:
        raise NotImplementedError



class MHDynEncoder(DynEncoder):
    """
    Inspired by "ATTENTIVE NEURAL PROCESSES", ICLR 2019 (https://arxiv.org/pdf/1901.05761.pdf)
    """
    def __init__(self, config: DynEncoderConfig):
        super().__init__(config)
        self.encoder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(net, config.enc_config.net_cls)(**asdict(config.enc_config))
        self.obs_encoder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(net, config.obs_enc_config.net_cls)(**asdict(config.obs_enc_config))
        self.query = nn.Parameter(torch.randn(1, 1, config.dyn_dim))  # (D,)
        self.mh_attn = nn.MultiheadAttention(embed_dim=config.dyn_dim, num_heads=1, add_bias_kv=True, batch_first=False)
        self.ff = nn.Sequential(nn.SiLU(), nn.Linear(config.dyn_dim, config.dyn_dim))
    
    @maybe_compile
    def forward(self, traj: Trajectory, only_last: bool=True) -> Dynamics:
        # RNN? Attention? Transformer?
        value = self.encoder(torch.cat(list(getattr(traj, k) for k in self.config.enc_config.inp_keys), dim=-1))  # (T, B, D)
        key = self.obs_encoder(traj.observations)  # (T, B, D)
        
        query = self.query.expand((-1 if only_last else traj.observations.shape[0]), traj.observations.shape[1], -1)  # (1, B, D)
        attn_output, _ = self.mh_attn.forward(query, key, value, need_weights=False)  # (1, B, D)

        if only_last:
            attn_output = attn_output.squeeze(0)
        
        return avg_l1_norm(attn_output)
