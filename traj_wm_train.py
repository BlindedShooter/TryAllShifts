import os
import pathlib
from pathlib import Path
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import time
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import d4rl
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
import tqdm

from tryallshifts import *
from tryallshifts.utils.func import *


# Configs
config: TASConfig = OmegaConf.load('./config/default.yaml')
config.merge_with(OmegaConf.from_cli())
config.merge_with(OmegaConf.load(f"./config/task/{config.task}.yaml"))


# Logging
dir_inc = 0
if "exp_name" in config:
    logdir=f"./logs/{config.exp_name}"
else:
    logdir=f"./logs/{int(time.time())}"

while Path(f"{logdir}_{dir_inc}").exists():
    dir_inc += 1
else:
    logdir = Path(f"{logdir}_{dir_inc}")
    logdir.mkdir(parents=True, exist_ok=False)

summary_writer = SummaryWriter(logdir=logdir, write_to_disk=True)


# Env & Dataset
env = gym.make(config.task + "-" + config.behavior)
dset = env.get_dataset()
for k, v in dset.items():
    if v.dtype == np.float32 or v.dtype == np.float64:
        dset[k] = symlog(v)
dloader = D4RLTransitionLoader(**dset, batch_size=256)
traj_dloader = D4RLTrajectoryLoader(**dset, batch_size=config.batch_size, trajectoy_len=16)

val_env = gym.make(config.task + "-" + config.val_behavior)
val_dset = val_env.get_dataset()
for k, v in val_dset.items():
    if v.dtype == np.float32 or v.dtype == np.float64:
        val_dset[k] = symlog(v)
val_dloader = D4RLTransitionLoader(**val_dset, batch_size=512)
val_traj_dloader = D4RLTrajectoryLoader(**val_dset, batch_size=32, trajectoy_len=16)

# Models
world_model: DynamicsWorldModel = MODEL_MAPPING[config.world_model.cls_name](config).to(config.device)
dyn_encoder: DynamicsEncoder = MODEL_MAPPING[config.dyn_encoder.cls_name](config).to(config.device)

dyn_latent = torch.empty((config.batch_size, config.dyn_encoder.dyn_dim)).to(config.device)
policy = lambda obs: torch.from_numpy(
    np.random.uniform(-1, 1, (obs.shape[0], *config.action_shape))
).to(dtype=torch.float32, device=config.device)


# Train
pbar = tqdm.trange(config.epochs)
policy: Policy

step = 0

for epoch in pbar:
    for np_traj in iter(traj_dloader):
        info = {}
        dyn_latent.normal_()
        th_traj = np_traj.to_tensor(device=config.device)
        
        pred_traj = world_model.rollout_actions(th_traj.observations[:, 0], th_traj.actions, dyn_latent)
        wm_loss, wm_info = world_model.calc_wm_loss(pred_traj, th_traj)

        info.update(wm_info)

        dyn_latent.normal_()
        imag_traj = world_model.rollout(th_traj.observations[:, 0], policy, dyn_latent, 32, detach=True)
        
        dyn_pred = dyn_encoder.trajectory_encode(imag_traj)
        dyn_loss = F.mse_loss(dyn_pred[:, -1, :], dyn_latent)

        info['dynenc/loss'] = dyn_loss.item()

        loss = wm_loss + dyn_loss * 0.5
        world_model.optim.zero_grad()
        dyn_encoder.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            chain(world_model.parameters(), dyn_encoder.parameters()), 5
        )
        world_model.optim.step()
        dyn_encoder.optim.step()

        for k, v in info.items():
            summary_writer.add_scalar(k, v, step)
        step += 1