from dataclasses import asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tas.configs import DynEncoderConfig

from tas.typing import *
from tas.configs import *
from tas.utils import *
from tas.distributions.continuous import TanhNormal
import tas.net as net
from tas.utils.done_funcs import DONE_FUNCS, not_done_func



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
        self.net = getattr(net, config.net_config.net_cls)(**asdict(config.net_config))
        self.distribution = TanhNormal(
            loc=torch.zeros(1), scale=torch.ones(1), upscale=self.config.upscale,
            min=self.config.min, max=self.config.max, event_dims=self.config.event_dims
        )


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
    

    def forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor] = None) -> Value:
        return torch.cat([critic(torch.cat([obs, act], dim=-1), dynamics) for critic in self.critics], dim=-1)


    def single_forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor] = None, idx: int = 0) -> Value:
        return self.critics[idx](torch.cat([obs, act], dim=-1), dynamics)



class DynModel(nn.Module):
    def __init__(self, config: DynModelConfig):
        super().__init__()
        self.config = config
        self.net = getattr(net, config.net_config.net_cls)(**asdict(config.net_config))
    

    def forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor] = None) -> State:
        return self.net(torch.cat([obs, act], dim=-1), dynamics)



class RewardModel(nn.Module):
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        self.net = getattr(net, config.net_config.net_cls)(**asdict(config.net_config))
    

    def forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor]=None) -> Reward:
        return self.net(torch.cat([obs, act], dim=-1), dynamics)



class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        if config.env_name.split("-")[0] in DONE_FUNCS:
            self.done_func = DONE_FUNCS[config.env_name.split("-")[0]]
        else:
            print(f'No done function found for {config.env_name}. Using default done function.')
            self.done_func = not_done_func

        self.dyn_model = DynModel(config.dyn_model_config)
        self.reward_model = RewardModel(config.reward_model_config)
    

    def forward(self, obs: torch.Tensor, act: torch.Tensor, dynamics: Optional[torch.Tensor] = None) -> Tuple[State, Reward]:
        next_obs = self.dyn_model(obs, act, dynamics=dynamics)
        #next_obs, reward = next_obs_rew[..., :-1], next_obs_rew[..., -2:-1]
        reward = self.reward_model(obs, act, dynamics=dynamics)
        done = self.done_func(next_obs).unsqueeze(-1)

        return next_obs, reward, done
    

    def rollout_actions(self, obs: torch.Tensor, actions: torch.Tensor, dynamics: Optional[torch.Tensor] = None, horizon: int = 1) -> Trajectory:
        obss, rewards, dones = [obs], [], []

        for i in range(horizon):
            n_obs, reward, done = self.forward(obss[-1], actions[i], dynamics[i])
            obss.append(n_obs)
            rewards.append(reward)
            dones.append(done)

        return Trajectory(
            all_observations=torch.stack(obss, dim=0),
            actions=actions,
            rewards=torch.stack(rewards, dim=0),
            dones=torch.stack(dones, dim=0),
            dynamics=dynamics
        )
    

    def rollout_policy(self, obs: torch.Tensor, policy: Callable[[Observation, Dynamics], Action], 
                        dynamics: Optional[torch.Tensor] = None, horizon: int = 1) -> Trajectory:
        obss, actions, rewards, dones = [obs], [], [], []

        for i in range(horizon):
            action = policy(obss[-1], dynamics[i])
            n_obs, reward, done = self.forward(obss[-1], action, dynamics[i])
            obss.append(n_obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        return Trajectory(
            all_observations=torch.stack(obss, dim=0),
            actions=torch.stack(actions, dim=0),
            rewards=torch.stack(rewards, dim=0),
            dones=torch.stack(dones, dim=0),
            dynamics=dynamics
        )



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
        self.encoder = getattr(net, config.enc_config.net_cls)(**asdict(config.enc_config))
        self.obs_encoder = getattr(net, config.obs_enc_config.net_cls)(**asdict(config.obs_enc_config))
        self.query = nn.Parameter(torch.randn(1, 1, config.dyn_dim))  # (D,)
        self.mh_attn = nn.MultiheadAttention(embed_dim=config.dyn_dim, num_heads=1, add_bias_kv=True, batch_first=False)
    

    def forward(self, traj: Trajectory) -> Dynamics:
        # RNN? Attention? Transformer?
        value = self.encoder(torch.cat([traj.observations, traj.actions, traj.rewards, traj.next_observations], dim=-1))  # (T, B, D)
        key = self.obs_encoder(traj.observations)  # (T, B, D)
        query = self.query.expand(-1, traj.observations.shape[1], -1)  # (1, B, D)
        attn_output, _ = self.mh_attn.forward(query, key, value, need_weights=False)  # (1, B, D)

        return avg_l1_norm(attn_output.squeeze(0))