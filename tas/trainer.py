import os
from copy import deepcopy
from collections import defaultdict

import tqdm

from tas.typing import *
from tas.configs import *
from tas.utils import *
from tas.model import *
import tas.model  # for getattr
from tas.net import *
from tas.eval import *
from tas.d4rl_dataset import *
from tas.losses import *
from tas.replay import TorchReplay, NumpyReplay



class TASTrainer:
    def __init__(self, config: TASConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.seed = config.seed
        self.train_states: Dict[str, Any] = {
            'critic_grad_updates': 0,
            'actor_grad_updates': 0,
            'wm_grad_updates': 0,
            'dynenc_grad_updates': 0,
            'imag_steps': 0,
        }
        self.logger = Logger(config)

        # Env & Dataset
        self.env = gym.make(config.env_name)
        self.val_env = gym.make(config.val_env_name)
        self.raw_dset = self.env.get_dataset()
        self.val_raw_dset = self.val_env.get_dataset()

        self.tran_dloader = D4RLTransitionLoader(**self.raw_dset, batch_size=config.batch_size)
        self.tran_gen = infinite_generator(self.tran_dloader)
        self.traj_dloader = D4RLTrajectoryLoader(**self.raw_dset, batch_size=config.batch_size, trajectory_len=config.imag_horizon)
        self.traj_gen = infinite_generator(self.traj_dloader)
        self.val_traj_dloader = D4RLTrajectoryLoader(**self.val_raw_dset, batch_size=config.batch_size, trajectory_len=config.imag_horizon)
        self.val_traj_gen = infinite_generator(self.val_traj_dloader)

        # Models
        self.wm = WorldModel(config.wm_config).to(self.device)
        self.actor = StochActor(config.actor_config).to(self.device)
        self.log_alpha = torch.tensor(np.log(config.init_alpha), requires_grad=True, device=self.device)
        self.critic = Critic(config.critic_config).to(self.device)
        self.target_critic = deepcopy(self.critic)
        self.dyn_encoder: DynEncoder = getattr(tas.model, config.dyn_enc_config.model_cls)(config.dyn_enc_config).to(self.device)
        
        self.true_dynamics = avg_l1_norm(torch.ones(
            config.imag_horizon, config.batch_size, config.dyn_enc_config.dyn_dim#, requires_grad=False
        ).to(self.device))# / np.sqrt(config.dyn_enc_config.dyn_dim)
        
        self.buffer: TorchReplay = TorchReplay(
            config.dim_dict['observation'], config.dim_dict['action'], 
            config.dyn_enc_config.dyn_dim, config.buffer_size, config.device
        )
        
        # Optimizers
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), **config.actor_config.optim_kwargs)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), **config.critic_config.optim_kwargs)
        self.dyn_encoder_optim = torch.optim.AdamW(self.dyn_encoder.parameters(), **config.dyn_enc_config.optim_kwargs)
        self.wm_optim = torch.optim.AdamW(self.wm.parameters(), **config.wm_config.optim_kwargs)
        self.log_alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)
    

    @property
    def steps(self):
        return self.train_states['critic_grad_updates'] + self.train_states['wm_grad_updates']


    def update_target_critic(self, tau: Optional[float]=None):
        if tau is None:
            tau = self.config.critic_config.target_tau
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    

    def train_wm(self, tran: Transition) -> InfoDict:
        self.wm_optim.zero_grad()

        wm_loss, wm_info = calc_wm_loss(tran, self.wm, dynamics=self.true_dynamics[0])  # hmm
        
        loss = wm_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), self.config.wm_config.grad_clip_norm)
        self.wm_optim.step()
        self.train_states['wm_grad_updates'] += 1

        return wm_info


    def train_dynenc(self, traj: Trajectory, train_wm: bool=True, policy_imagine: bool=False) -> InfoDict:
        self.dyn_encoder_optim.zero_grad()

        rand_dynamics = avg_l1_norm(torch.empty((
            1, traj.observations.shape[1], self.config.dyn_enc_config.dyn_dim
        ), requires_grad=False, device=self.device).normal_(-1, 1).expand(traj.observations.shape[0], -1, -1))

        if policy_imagine:
            imag_traj = self.wm.rollout_policy(
                traj.observations[0], self.actor.predict, dynamics=rand_dynamics, horizon=self.config.imag_horizon
            )
        else:
            imag_traj = self.wm.rollout_actions(
                traj.observations[0], traj.actions, dynamics=rand_dynamics, horizon=self.config.imag_horizon
            )
        traj = Trajectory(traj.all_observations, traj.actions, traj.rewards, traj.dones, dynamics=self.true_dynamics)
        
        if train_wm:
            self.wm_optim.zero_grad()
        else:
            imag_traj = Trajectory(
                imag_traj.all_observations.detach(), imag_traj.actions.detach(), 
                imag_traj.rewards.detach(), imag_traj.dones.detach(), dynamics=imag_traj.dynamics.detach()
            )
        cat_traj = traj.torch_cat(imag_traj)

        dyn_enc_loss, dyn_enc_info = calc_dyn_enc_loss(cat_traj, self.dyn_encoder)
        dyn_enc_loss.backward()
        nn.utils.clip_grad_norm_(self.dyn_encoder.parameters(), self.config.dyn_enc_config.grad_clip_norm)
        self.train_states['dynenc_grad_updates'] += 1

        if train_wm:
            nn.utils.clip_grad_norm_(self.wm.parameters(), self.config.wm_config.grad_clip_norm)
            self.wm_optim.step()
        self.dyn_encoder_optim.step()

        return dyn_enc_info


    # Train behavior cloning actor
    def train_actor_bc(self, tran: Transition) -> InfoDict:
        self.actor_optim.zero_grad()
        tran = Transition(
            observations=tran.observations, 
            actions=tran.actions,
            next_observations=tran.next_observations,
            rewards=tran.rewards,
            dones=tran.dones,
            dynamics=self.true_dynamics[0]
        )

        actor_loss, actor_info = calc_actor_bc_loss(tran, self.actor)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.actor_config.grad_clip_norm)
        self.actor_optim.step()

        return actor_info
    

    def train_critic(self, tran: Transition) -> InfoDict:
        critic_loss, critic_info = calc_critic_loss(
            tran, self.actor, self.critic, self.target_critic, self.config.gamma, self.log_alpha
        )

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.critic_config.grad_clip_norm)
        self.critic_optim.step()
        self.train_states['critic_grad_updates'] += 1

        return critic_info
    

    def train_actor(self, tran: Transition) -> InfoDict:
        actor_loss, actor_info = calc_actor_loss(tran, self.actor, self.critic, self.log_alpha)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.actor_config.grad_clip_norm)
        self.actor_optim.step()
        self.train_states['actor_grad_updates'] += 1

        alpha_loss, alpha_info = calc_alpha_loss(self.log_alpha, actor_info['NOLOG-log_prob_w_grad'], target_entropy=self.config.target_entropy)
        actor_info.update(alpha_info)
        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()

        return actor_info
    

    @torch.no_grad()
    def collect_experience(self, imag_horizon: Optional[int] = None) -> InfoDict:
        if imag_horizon is None:
            imag_horizon = self.config.imag_horizon
        # adapt하는 과정을 담아야 하는 것이 아닌가? 그냥 true dyn을 넣어줘도 되나?  
        tran: Transition = next(self.tran_gen).to_tensor(self.device)  # for wm initial obs. need first obs generation?
        rand_dynamics = avg_l1_norm(torch.empty((
            1, tran.observations.shape[0], self.config.dyn_enc_config.dyn_dim
        ), requires_grad=False, device=self.device).normal_(-1, 1).expand(imag_horizon, -1, -1))

        imag_traj = self.wm.rollout_policy(tran.observations, self.actor.predict, rand_dynamics, horizon=imag_horizon)
        self.train_states['imag_steps'] += imag_traj.observations.shape[0] * imag_traj.observations.shape[1]
        self.buffer.store_trajectory(imag_traj)

        return {'policy_eval/imag_rew_mean': imag_traj.rewards.mean().item()}


    def learn_wm_and_dynenc(self) -> InfoDict:
        tran = next(self.tran_gen).to_tensor(self.device)

        info = {}
        info.update(self.train_wm(tran))

        if self.config.use_dyn:
            traj = next(self.traj_gen).to_tensor(self.device)
            info.update(self.train_dynenc(traj))

        return info


    def learn_policy(self) -> InfoDict:
        tran = self.buffer.sample(self.config.batch_size)
        info = {}
        info.update(self.train_critic(tran))
        info.update(self.train_actor(tran))
        with torch.no_grad():
            self.update_target_critic()

        return info
    

    def eval_wm(self) -> InfoDict:
        exp_traj = next(self.val_traj_gen).to_tensor(self.device)

        oracle_info = eval_wm_on_env(self.wm, self.true_dynamics[:, 0:1], self.env, self.actor.predict, [calc_traj_mse], horizon=self.config.imag_horizon)
        info = {f'wm_eval/oracle_{k}': v for k, v in oracle_info.items()}

        if self.config.use_dyn:
            transfer_info = eval_wm_on_traj_oracledyn(self.wm, self.dyn_encoder, exp_traj)
            info.update({f'wm_eval/transfer_{k}': v for k, v in transfer_info.items()})
        
        return info
    

    
    def eval_policy(self) -> InfoDict:
        infos = defaultdict(float)

        @torch.no_grad()
        def actor_predict(obs):
            obs = tensorify(obs, device=self.device).unsqueeze(0)
            act = self.actor.predict(obs, dynamics=self.true_dynamics[0, 0:1], deterministic=True)
            return numpify(act)[0]

        for _ in range(self.config.eval_episodes):
            info = eval_policy_on_env(actor_predict, self.env)
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    infos[k] += v
        
        return {f'policy_eval/{k}': v / self.config.eval_episodes for k, v in infos.items()}


    def learn(self, wm_steps: int = 20000, sac_steps: int = 1000000):
        # WM & DynEnc
        wm_pbar = tqdm.trange(wm_steps, desc='WM')
        for _ in wm_pbar:
            info = self.learn_wm_and_dynenc()
            #wm_pbar.set_postfix(**dict((k, v) for k, v in info.items() if isinstance(v, float)), refresh=False)
            self.logger.log({f'train/wm/{k}': v for k, v in info.items()}, self.steps)

            if self.steps % self.config.eval_freq == 0:
                self.logger.log(self.eval_wm(), self.steps)

            if self.steps % self.config.save_freq == 0:
                self.save_models()
        wm_pbar.close()

        # SAC
        while len(self.buffer) < self.config.warmup_steps:
            self.collect_experience()
        
        sac_pbar = tqdm.trange(sac_steps, desc='SAC')
        for _ in sac_pbar:
            info = self.learn_policy()
            if self.train_states['critic_grad_updates'] % self.config.imag_horizon == 0:
                info.update(self.collect_experience())
            #info.update(self.train_dynenc(next(self.traj_gen).to_tensor(self.device), train_wm=False, policy_imagine=True))

            #sac_pbar.set_postfix(**dict((k, v) for k, v in info.items() if isinstance(v, float)), refresh=False)
            self.logger.log({f'train/policy/{k}': v for k, v in info.items()}, self.steps)

            if self.steps % self.config.eval_freq == 0:
                self.logger.log(self.eval_policy(), self.steps)
            
            if self.steps % self.config.save_freq == 0:
                self.save_models()


    def save_models(self, suffix: Optional[str] = None):
        if suffix is None:
            suffix = f'_{self.steps}'
        
        save_dict = {
            'wm': self.wm.state_dict(),
            'wm_optim': self.wm_optim.state_dict(),
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'dyn_encoder': self.dyn_encoder.state_dict(),
            'dynenc_optim': self.dyn_encoder_optim.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'log_alpha_optim': self.log_alpha_optim.state_dict(),
            'config': self.config,
            'train_states': self.train_states
        }
        torch.save(save_dict, os.path.join(self.logger.tb_log_dir, f'{self.config.exp_prefix}_step{suffix}.pt'))
    

    def load_models(self, path: str):
        load_dict = torch.load(path)
        self.wm.load_state_dict(load_dict['wm'])
        self.wm_optim.load_state_dict(load_dict['wm_optim'])
        self.actor.load_state_dict(load_dict['actor'])
        self.actor_optim.load_state_dict(load_dict['actor_optim'])
        self.critic.load_state_dict(load_dict['critic'])
        self.critic_optim.load_state_dict(load_dict['critic_optim'])
        self.dyn_encoder.load_state_dict(load_dict['dyn_encoder'])
        self.dyn_encoder_optim.load_state_dict(load_dict['dynenc_optim'])
        self.log_alpha = torch.tensor(load_dict['log_alpha'], requires_grad=True, device=self.device)
        self.log_alpha_optim.load_state_dict(load_dict['log_alpha_optim'])
        self.config = load_dict['config']
        self.train_states = load_dict['train_states']
