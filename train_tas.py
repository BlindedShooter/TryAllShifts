from pprint import pprint
import numpy as np
import torch
import gym
import d4rl
import tqdm
import pyrallis

from tas.typing import *
from tas.utils import *
from tas.losses import *
from tas.configs import *
from tas.model import *
from tas.replay import *
from tas.trainer import *
from tas.utils.torch_funcs import shrink_and_perturb, weight_reset

from tensorboardX import SummaryWriter
import time
from collections import defaultdict

config: TASConfig = pyrallis.parse(config_class=TASConfig)
config.infer_shapes()
trainer = TASTrainer(config)
trainer.learn(config.wm_train_steps, config.policy_train_steps)
trainer.logger.close()