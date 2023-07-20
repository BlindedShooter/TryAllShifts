from dataclasses import dataclass, field
from typing import Dict, Any
import gym
import d4rl
import numpy as np


@dataclass
class ContextMLPConfig:
    net_cls: str = 'ConcatFirstMLP'
    inp_keys: list = field(default_factory=lambda: ['observation'])
    out_keys: list = field(default_factory=lambda: ['action'])
    inp_dim: int = 17  # Can be automatically inferred from config.inp_keys
    out_dim: int = 6  # Can be automatically inferred from config.out_keys
    context_dim: int = 0
    arch: list = field(default_factory=lambda: [256, 256])
    actv: str = 'SiLU'
    out_actv: str = 'Identity'
    layer_norm: bool = False
    bias: bool = True



@dataclass
class ModelConfig:
    net_config: ContextMLPConfig = field(default_factory=lambda: ContextMLPConfig())
    optim_kwargs: dict = field(default_factory=lambda: {'lr': 1e-3, 'weight_decay': 1e-5})
    grad_clip_norm: float = 10.0
    stochastic: bool = False
    num_dist_params: int = 2  # ignored if deterministic
    device: str = None  # will be set on post_init



@dataclass
class ActorConfig(ModelConfig):
    stochastic: bool = True
    num_dist_params: int = 2

    # for stochastic actors. ignored if deterministic.  TODO: to dist_kwargs?
    upscale: int = 5
    min: float = -1.0
    max: float = 1.0
    event_dims: int = 1
    net_config: ContextMLPConfig = field(default_factory=lambda: ContextMLPConfig(
        inp_keys=['observation'], out_keys=['action']
    ))



@dataclass
class CriticConfig(ModelConfig):
    num_critics: int = 2
    target_tau: float = 0.005
    net_config: ContextMLPConfig = field(default_factory=lambda: ContextMLPConfig(
        inp_keys=['observation', 'action'], out_keys=['value'], layer_norm=True
    ))



@dataclass
class DynModelConfig(ModelConfig):
    net_config: ContextMLPConfig = field(default_factory=lambda: ContextMLPConfig(
        inp_keys=['observation', 'action'], out_keys=['next_observation'], actv='SiLU', arch=[256, 256, 256]
    ))



@dataclass
class RewardModelConfig(ModelConfig):
    net_config: ContextMLPConfig = field(default_factory=lambda: ContextMLPConfig(
        inp_keys=['observation', 'action'], out_keys=['reward']
    ))



@dataclass
class WorldModelConfig(ModelConfig):
    env_name: str = ''
    use_separate_reward_model: bool = False  # will add reward dimension automatically to the output of dynamics model
    dyn_model_config: DynModelConfig = field(default_factory=DynModelConfig)
    reward_model_config: RewardModelConfig = field(default_factory=RewardModelConfig)
    stochastic: bool = True

    def __post_init__(self):
        self.dyn_model_config.stochastic = self.stochastic
        self.reward_model_config.stochastic = self.stochastic



@dataclass
class DynEncoderConfig(ModelConfig):
    model_cls: str = "MHDynEncoder"

    net_config = None  # unused
    enc_config: ContextMLPConfig = field(default_factory=lambda: ContextMLPConfig(
        # out_keys name can overlap for now... only used for shape inference.
        net_cls="PlainMLP", inp_keys=['observations', 'actions', 'next_observations'], out_keys=['dynamics']  
    ))
    obs_enc_config: ContextMLPConfig = field(default_factory=lambda: ContextMLPConfig(
        net_cls="PlainMLP", inp_keys=['observation'], out_keys=['dynamics']
    ))  # hmm...
    max_traj_len: int = 1000
    dyn_dim: int = 32



@dataclass
class TASConfig:
    actor_config: ActorConfig = field(default_factory=ActorConfig)
    critic_config: CriticConfig = field(default_factory=CriticConfig)
    wm_config: WorldModelConfig = field(default_factory=WorldModelConfig)
    dyn_enc_config: DynEncoderConfig = field(default_factory=DynEncoderConfig)

    device: str = 'cuda'
    seed: int = 0  # Currently unused
    env_name: str = 'halfcheetah-medium-replay-v2'
    val_env_name: str = 'halfcheetah-expert-v2'
    
    log_dir: str = 'logs'
    debug: bool = False
    exp_prefix: str = ''
    proj_name: str = 'TAS'
    info_polyak: float = 0.995
    log_freq: int = 200 # WM or Critic Updates
    save_freq: int = 10000  # WM or Critic Updates
    eval_freq: int = 1000  # WM or Critic Updates
    eval_episodes: int = 1
    eval_save_gif: bool = False
    eval_save_video: bool = False
    eval_save_dir: str = 'eval'

    wm_train_steps: int = 100000
    policy_train_steps: int = 1000000
    dynenc_update_freq: int = 16
    gamma: float = 0.99
    target_entropy: float = -1.0
    init_alpha: float = 0.5
    # For dynamics encoder
    imag_horizon: int = 64
    #
    policy_rollout_len: int = 16
    use_dyn: bool = True
    use_mopo_pen: bool = False  # Only available if world model is stochastic
    mopo_pen_lam: float = 1.0
    warmup_steps: int = 10000
    buffer_size: int = 1000000

    batch_size: int = 256
    lr_alpha: float = 1e-4

    dim_dict: dict = field(default_factory=lambda: {})


    def __post_init__(self):
        self.wm_config.env_name = self.env_name
        self.wm_config.device = self.device
        self.dyn_enc_config.device = self.device
        self.actor_config.device = self.device
        self.critic_config.device = self.device



    def infer_shapes(self):
        """
        Automatically fills out the inp_dim and out_dim fields of all ContextMLPConfigs,
        based on the config itself and the environment's observation and action spaces.
        """
        env = gym.make(self.env_name)
        dim_dict = {
            'observation': int(np.prod(env.observation_space.shape)),
            'next_observation': int(np.prod(env.observation_space.shape)),
            'action': int(np.prod(env.action_space.shape)),
            'reward': 1,
            'done': 1,
            'value': 1,
            'multi_value': self.critic_config.num_critics,
            'dynamics': self.dyn_enc_config.dyn_dim if self.use_dyn else 0,
        }

        dim_dict.update({k + 's' : v for k, v in dim_dict.items()})
        self.dim_dict = dim_dict
        self.actor_config.event_dims = len(env.action_space.shape)
        
        if not self.use_dyn:
            self.disable_dynamics()

        # Iterate through all ContextMLPConfigs and infer shapes
        for config in [
            self.actor_config, self.critic_config,
            self.wm_config.dyn_model_config, self.wm_config.reward_model_config, 
        ]:
            config.net_config.inp_dim = sum([dim_dict[k] for k in config.net_config.inp_keys])
            config.net_config.out_dim = sum([dim_dict[k] for k in config.net_config.out_keys])
            config.net_config.context_dim = dim_dict['dynamics']

            if config.stochastic:
                config.net_config.out_dim *= config.num_dist_params

        for config in (self.dyn_enc_config.obs_enc_config, self.dyn_enc_config.enc_config):
            config.inp_dim = sum([dim_dict[k] for k in config.inp_keys])
            config.out_dim = sum([dim_dict[k] for k in config.out_keys])
            config.context_dim = dim_dict['dynamics']
        
        if not self.wm_config.use_separate_reward_model:
            self.wm_config.dyn_model_config.net_config.out_dim += self.wm_config.dyn_model_config.num_dist_params

    

    def disable_dynamics(self):
        for config in [self.actor_config.net_config, self.critic_config.net_config, 
                        self.wm_config.dyn_model_config.net_config, self.wm_config.reward_model_config.net_config, 
                        self.dyn_enc_config.enc_config, self.dyn_enc_config.obs_enc_config]:
            config.net_cls = "PlainMLP"
    

    @property
    def exp_name(self):
        exp_name = f"{self.exp_prefix}{'_' if self.exp_prefix else ''}{'dyn' if self.use_dyn else 'nodyn'}_{self.env_name}"
        
        net_name_replace_dict = {
            'Concat': 'C',
            'Plain': 'P',
            'Bilinear': 'B',
            'FiLM': 'F',
            'MLP': 'M',
            'First': 'F',
            'Last': 'L',
            'Full': 'A',
        }

        for model_name, model_net_conf in [
            ('act', self.actor_config.net_config),
            ('cri', self.critic_config.net_config),
            ('wm', self.wm_config.dyn_model_config.net_config),
            ('dyn', self.dyn_enc_config.enc_config),
        ]:
            net_names = model_net_conf.net_cls
            for k, v in net_name_replace_dict.items():
                net_names = net_names.replace(k, v)
            exp_name += f"_{model_name}_{net_names}"

        return exp_name
    

    def get_tags(self):
        tags = ['train_' + self.env_name, 'val_' + self.val_env_name, ('dyn' if self.use_dyn else 'nodyn')]
        
        net_name_replace_dict = {
            'Concat': 'C',
            'Plain': 'P',
            'Bilinear': 'B',
            'FiLM': 'F',
            'MLP': 'M',
            'First': 'F',
            'Last': 'L',
            'Full': 'A',
        }

        for model_name, model_net_conf in [
            ('act', self.actor_config.net_config),
            ('cri', self.critic_config.net_config),
            ('wm', self.wm_config.dyn_model_config.net_config),
            ('dyn', self.dyn_enc_config.enc_config),
        ]:
            net_names = model_net_conf.net_cls
            for k, v in net_name_replace_dict.items():
                net_names = net_names.replace(k, v)
            tags.append(f"{model_name}_{net_names}")

        return tags