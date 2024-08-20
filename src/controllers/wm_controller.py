#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch as th

from .basic_controller import BasicMAC


class DataParallelAgent(th.nn.DataParallel):
    def init_hidden(self):
        # make hidden states on same device as model
        return self.module.init_hidden()


# This multi-agent controller shares parameters between agents
class WMMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(WMMAC, self).__init__(scheme, groups, args)
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, wm_latent_state, reconstruct_hidden_state, original_embedding = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, wm_latent_state, reconstruct_hidden_state, original_embedding # [batch, n_actions], [batch, n_agents, wm_latent_dim]

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, wm_latent_state, self.hidden_states, original_embedding = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), wm_latent_state, self.hidden_states, original_embedding

    # Add new func
    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim), (
            enemy_feats_dim, ally_feats_dim)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        obs_component_dim, _ = self._get_obs_component_dim()
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        move_feats_t, enemy_feats_t, ally_feats_t, own_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents * self.n_enemies,
                                              -1)  # [bs * n_agents * n_enemies, fea_dim]
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]
        # merge move features and own features to simplify computation.
        context_feats = [move_feats_t, own_feats_t]  # [batch, agent_num, own_dim]
        own_context = th.cat(context_feats, dim=2).reshape(bs * self.n_agents, -1)  # [bs * n_agents, own_dim]

        embedding_indices = []
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents]
            embedding_indices.append(th.arange(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1))
        if self.args.obs_last_action:
            # action-id indices, [bs, n_agents]
            if t == 0:
                embedding_indices.append(None)
            else:
                embedding_indices.append(batch["actions"][:, t - 1].squeeze(-1))

        return bs, own_context, enemy_feats_t, ally_feats_t, embedding_indices

    def _get_input_shape(self, scheme):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component
        own_context_dim = move_feats_dim + own_feats_dim
        return own_context_dim, enemy_feats_dim, ally_feats_dim
