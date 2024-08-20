import copy
import time

import numpy as np
import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.wm_qmix import WMQMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num

WORLD_MODELING_AGENTS = ["wm_hyper_rnn"]

def calculate_target_q(target_mac, batch, enable_parallel_computing=False, thread_num=4):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)
    with th.no_grad():
        # Set target mac to testing mode
        target_mac.set_evaluation_mode()
        target_mac_out = []
        target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _, _, _ = target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
        return target_mac_out


def calculate_n_step_td_target(target_mixer, target_max_qvals, batch, rewards, terminated, mask, gamma, td_lambda,
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        target_max_qvals, _, _ = target_mixer(target_max_qvals, batch["state"])

        if q_lambda:
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()


class WMLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":  # 31.521K
            self.mixer = Mixer(args)
        elif args.mixer == "wm_qmix":
            self.mixer = WMQMixer(args)
        else:
            raise "mixer error"

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0
        self.wm_latent_dim = args.rnn_hidden_dim // args.wm_latent_divider

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        wm_latent_obs = []
        reconstruction_hidden_state = []
        original_hidden_state = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, wm_latent_obs_outs, hidden_state_outs, original_hiddent_state_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            wm_latent_obs.append(wm_latent_obs_outs)
            reconstruction_hidden_state.append(hidden_state_outs)
            original_hidden_state.append(original_hiddent_state_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        wm_latent_obs = th.stack(wm_latent_obs, dim=1)  # Concat over time
        reconstruction_hidden_state = th.stack(reconstruction_hidden_state, dim=1)  # Concat over time
        original_hidden_state = th.stack(original_hidden_state, dim=1)  # Concat over time


        # TODO: double DQN action, COMMENT: do not need copy
        mac_out[avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = calculate_target_q(self.target_mac, batch)

            # Max over target Q-Values/ Double q learning
            # mac_out_detach = mac_out.clone().detach()
            # TODO: COMMENT: do not need copy
            mac_out_detach = mac_out
            # mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]

            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            assert getattr(self.args, 'q_lambda', False) == False
            targets = calculate_n_step_td_target(
                self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                self.args.td_lambda
            )

        # Set mixing net to training mode
        self.mixer.train()
        # Mixer
        original_state = batch["state"][:, :-1]
        chosen_action_qvals, wm_state_latent, reconstruction_state = self.mixer(chosen_action_qvals, original_state)

        # Calculate DQN loss
        td_error = (chosen_action_qvals - targets)
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        mask_elems = mask.sum()
        dqn_loss = masked_td_error.sum() / mask_elems

        # Calculate world model loss
        last_batch_size = batch.batch_size * (batch.max_seq_length - 1)
        wm_latent_obs = wm_latent_obs.view(-1, self.args.n_agents, self.wm_latent_dim)[:last_batch_size, :]
        reconstruction_hidden_state = reconstruction_hidden_state.view(-1, self.args.n_agents, self.args.rnn_hidden_dim)[:last_batch_size, :]
        original_hidden_state = original_hidden_state.view(-1, self.args.n_agents, self.args.rnn_hidden_dim)[:last_batch_size, :]
        wm_state_latent = wm_state_latent.view(-1, 1, self.wm_latent_dim).repeat(1, self.args.n_agents, 1)[:last_batch_size, :]
        reconstruction_state = reconstruction_state.view(-1, self.args.state_shape)
        original_state = original_state.reshape(-1, self.args.state_shape)

        obs_mse_loss = self.args.obs_rl_lambda * th.mean((reconstruction_hidden_state - original_hidden_state) ** 2)
        state_mse_loss = self.args.state_rl_lambda * th.mean((reconstruction_state - original_state) ** 2)
        latent_mse_loss = self.args.latent_rl_lambda * th.mean((wm_latent_obs - wm_state_latent) ** 2)

        # Total loss
        loss = dqn_loss + obs_mse_loss + state_mse_loss + latent_mse_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        # print("Avg cost {} seconds".format(self.avg_time))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # For log
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("loss_dqn", dqn_loss.item(), t_env)
            self.logger.log_stat("loss_obs_mse", obs_mse_loss.item(), t_env)
            self.logger.log_stat("loss_state_mse", state_mse_loss.item(), t_env)
            self.logger.log_stat("loss_latent_mse", latent_mse_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("target_mean", target_mean, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

