# From TD-MPC Github code by Nicklas Hansen, https://github.com/nicklashansen/tdmpc/blob/main/src/cfg.py
import os
import re
from omegaconf import OmegaConf


def parse_config(config_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(config_path / 'default.yaml')
    cli = OmegaConf.from_cli()
    for key, value in cli.items():
        if value is None:
            cli[key] = True  # pylint: disable=E1137:unsupported-assignment-operation
    base.merge_with(cli)

       # Modality config
    if cli.get('modality', base.modality) not in {'state', 'pixels'}:
        raise ValueError(
            f"Invalid modality: {cli.get('modality', base.modality)}")
    modality = cli.get('modality', base.modality)
    if modality != 'state':
        mode = OmegaConf.load(config_path / f'{modality}.yaml')
        base.merge_with(mode, cli)

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
    base.device = 'cuda' if base.modality == 'state' else 'cpu'
    base.exp_name = str(base.get('exp_name', 'default'))

    return base
