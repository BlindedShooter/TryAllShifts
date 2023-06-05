# From TD-MPC Github code by Nicklas Hansen, https://github.com/nicklashansen/tdmpc/blob/main/src/cfg.py
import os
import re
from typing import Tuple, Dict, Any
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING



def parse_config(config_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(config_path / 'default.yaml')
    cli = OmegaConf.from_cli()
    for key, value in cli.items():
        if value is None:
            cli[key] = True  # pylint: disable=E1137:unsupported-assignment-operation
    base.merge_with(cli)

    # Modality config
    #if cli.get('modality', base.modality) not in {'state', 'pixels'}:
    #    raise ValueError(
    #        f"Invalid modality: {cli.get('modality', base.modality)}")
    #modality = cli.get('modality', base.modality)
    #if modality != 'state':
    #    mode = OmegaConf.load(config_path / f'{modality}.yaml')
    #    base.merge_with(mode, cli)

    # Task config
    try:
        domain, _ = base.task.split('-', 1)
    except Exception as exc:
        raise ValueError(f'Invalid task name: {base.task}') from exc
    domain_path = config_path / 'tasks' / f'{domain}.yaml'
    if not os.path.exists(domain_path):
        domain_path = config_path / 'tasks' / 'default.yaml'
    domain_cfg = OmegaConf.load(domain_path)
    base.merge_with(domain_cfg, cli)

    # Algebraic expressions
    for k, v in base.items():
        if isinstance(v, str):
            match = re.match(r'(\d+)([+\-*/])(\d+)', v)
            if match:
                base[k] = eval(match.group(1) + match.group(2) + match.group(3))  # pylint: disable=W0123:eval-used
                if isinstance(base[k], float) and base[k].is_integer():
                    base[k] = int(base[k])

    # Convenience
    base.task_title = base.task.replace('-', ' ').title()
    base.device = 'cuda'  # if base.modality == 'state' else 'cpu'
    base.exp_name = str(base.get('exp_name', 'default'))

    return base



@dataclass
class ModelConfig:
    cls_name: str = MISSING
    optim_cls: str = "torch.optim.AdamW"
    optim_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    loss_weights: Dict[str, float] = field(default_factory=lambda: {})

    def get_cls(self, cls_dict):
        return cls_dict[self.cls_name]



@dataclass
class TASWorldModelConfig(ModelConfig):
  enc_arch: Tuple[int] = (512, 512, 512)  # (Observation, Action -> State)
  recon_arch: Tuple[int] = (256,)   # (State -> Observation)   if empty, one Linear Layer is applied
  reward_arch: Tuple[int] = (256,) # (State -> reward)   if empty, one Linear Layer is applied
  done_arch: Tuple[int] = (256,)    # (State -> Terminated)   if empty, one Linear Layer is applied



@dataclass
class TASDynamicsConfig(ModelConfig):
  use_trajectory: bool = False
  concat: str = "OAR"  # O: Observation, A: Action, R: Reward, T: Done, N: Next Obs, S: State, M: Next State, B: Action Dist, D: Dynamics

  dyn_dim: int = 32  # final output size
  learnable_init_state: bool = False
  inp_size: int = 24  # hmm.. auto calculate from .concat?

  arch: Tuple[int] = (256, 128)  # if RNN, dimensions are ignored, only number of layers will be used.



@dataclass
class TASPolicyConfig(ModelConfig):
    value_arch: Tuple[int] = (256, 256)
    actor_arch: Tuple[int] = (256, 256)



@dataclass
class TASConfig:
    exp_name: str = MISSING
    task: str = MISSING
    behavior: str = MISSING
    val_behavior: str = MISSING
    discount: float = MISSING
    observation_shape: Tuple[int] = MISSING
    action_shape: Tuple[int] = MISSING
    device: str = "cpu"

    batch_size: int = 64
    rollout_len: int = 16
    epochs: int = 100
    val_freq: int = 8

    ckpt_freq: int = 5000
    
    world_model: TASWorldModelConfig = TASWorldModelConfig()
    dyn_encoder: TASDynamicsConfig = TASDynamicsConfig()
    policy: TASPolicyConfig = TASPolicyConfig()
