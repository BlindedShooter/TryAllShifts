from dataclasses import asdict
from pathlib import Path

from tensorboardX import SummaryWriter
import wandb
import pyrallis as pyls

from tas.typing import *
from tas.utils.data import numpify
from tas.configs import TASConfig



class Logger:
    def __init__(self, config: TASConfig, log_wandb: bool=True):
        self.config = config
        self.log_dir = config.log_dir
        self.log_freq = config.log_freq
        self.log_wandb = log_wandb
        self.last_log_step = 0

        for i in range(1000):
            if not (tb_log_dir := Path(f'{self.log_dir}/{config.exp_name}_{i}')).exists():
                tb_log_dir.mkdir(parents=True)
                break
        self.tb_log_dir = tb_log_dir
        self.writer = SummaryWriter(log_dir=tb_log_dir, write_to_disk=(not config.debug))
        self.writer.add_text('config', str(asdict(config)), self.last_log_step)
        if log_wandb:
            wandb.init(
                name=config.exp_prefix, project='tas', dir=self.log_dir, config=asdict(config),
                tags=config.get_tags(), save_code=True, mode=('online' if not config.debug else 'offline')
            )
        self.info_dict = {}


    def log(self, info_dict: InfoDict, step: Optional[int]=None):
        # polyak update if value is float
        for k, v in info_dict.items():
            if isinstance(v, float):
                self.info_dict[k] = self.info_dict.get(k, v) * self.config.info_polyak + v * (1 - self.config.info_polyak)
            else:
                self.info_dict[k] = v
        
        if step is not None:
            if step - self.last_log_step >= self.log_freq: 
                self.last_log_step = step
                self._log()


    def _log(self):
        for k, v in self.info_dict.items():
            if 'NOLOG' in k:
                continue
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, self.last_log_step)
                if self.log_wandb:
                    wandb.log({k: v}, step=self.last_log_step)
            elif hasattr(v, 'shape'):
                v = numpify(v)
                if len(v.shape) == 1:  # histogram
                    self.writer.add_histogram(k, v, self.last_log_step)
                    if self.log_wandb:
                        wandb.log({k: wandb.Histogram(v)}, step=self.last_log_step)
                elif len(v.shape) == 2 or len(v.shape) == 3:  # image  (H, W) or (H, W, C)
                    self.writer.add_images(k, v, self.last_log_step)
                    if self.log_wandb:
                        wandb.log({k: [wandb.Image(v)]}, step=self.last_log_step)
                elif len(v.shape) == 4:  # video  (T, H, W, C)
                    self.writer.add_video(k, v, self.last_log_step)
                    if self.log_wandb:
                        wandb.log({k: wandb.Video(v)}, step=self.last_log_step)
                else:
                    raise NotImplementedError
        self.info_dict = {}
        self.writer.flush()


    def close(self):
        self.writer.close()
        if self.log_wandb:
            wandb.join()
