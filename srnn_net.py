"""Standalone inference reimplementation of the GenSafeNav (Ours_GST) policy.

Reconstructs *only* the test-time forward path of the upstream conformal-RL policy
(``rl/networks/{model,selfAttn_srnn_temp_node,srnn_model,distributions,
network_utils}.py``) so the committed checkpoint loads with ``strict=True`` outside
the upstream ``crowd_sim`` simulator and without the openai-baselines / RVO2 stack.

Module and parameter names match upstream exactly so the saved ``state_dict`` keys
(``base.*``, ``dist.fc_mean.*``) line up one-for-one. The policy is the
``selfAttn_merge_srnn`` base with ``config.policy.aci_input=True``: per-human
spatial edges (12 dims: current + 5 predicted relative positions) are concatenated
with 5 adaptive-conformal-inference conformity scores -> 17-dim attention input.

Action is holonomic (vx, vy) in world frame; the bridge applies no diff-drive
projection (handled by the edge node for differential_drive, but this planner is
omnidirectional). Deterministic eval uses ``dist.mode()`` (the mean), so the
constant unit log-std (``constant_std=True``, no learnable ``logstd`` parameter in
the checkpoint) never affects the action.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# --- upstream rl/networks/network_utils.py ---------------------------------
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    """constant_std=True branch: bias is a fixed (non-parameter) zero tensor.

    Upstream hardcodes ``.cuda()`` here; we keep it device-agnostic so CPU
    inference works. It is excluded from the state_dict either way.
    """

    def __init__(self, bias):
        super().__init__()
        self._bias = bias.unsqueeze(1)

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias.to(x.device)


# --- upstream rl/networks/distributions.py ---------------------------------
class FixedNormal(torch.distributions.Normal):
    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size(), device=x.device)
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


# --- upstream rl/networks/srnn_model.py (RNNBase, reshapeT) ----------------
def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))


class RNNBase(nn.Module):
    def __init__(self, args, edge):
        super().__init__()
        self.args = args
        if edge:
            self.gru = nn.GRU(args.human_human_edge_embedding_size, args.human_human_edge_rnn_size)
        else:
            self.gru = nn.GRU(args.human_node_embedding_size * 2, args.human_node_rnn_size)

    def _forward_gru(self, x, hxs, masks):
        # acting model only: input shape[0] == hidden state shape[0]
        seq_len, nenv, agent_num, _ = x.size()
        x = x.view(seq_len, nenv * agent_num, -1)
        mask_agent_num = masks.size()[-1]
        hxs_times_masks = hxs * (masks.view(seq_len, nenv, mask_agent_num, 1))
        hxs_times_masks = hxs_times_masks.view(seq_len, nenv * agent_num, -1)
        x, hxs = self.gru(x, hxs_times_masks)
        x = x.view(seq_len, nenv, agent_num, -1)
        hxs = hxs.view(seq_len, nenv, agent_num, -1)
        return x, hxs


# --- upstream rl/networks/selfAttn_srnn_temp_node.py -----------------------
class SpatialEdgeSelfAttn(nn.Module):
    """Human-human multi-head self attention."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        # CrowdSimPredRealGST-v0 with aci_input=True -> input is 12 (spatial) + 5 (conformity).
        self.input_size = 17
        self.num_attn_heads = 8
        self.attn_size = 512

        self.embedding_layer = nn.Sequential(
            nn.Linear(self.input_size, 128), nn.ReLU(),
            nn.Linear(128, self.attn_size), nn.ReLU(),
        )
        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)
        self.multihead_attn = torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        device = each_seq_len.device
        mask = torch.zeros(seq_len * nenv, max_human_num + 1, device=device)
        mask[torch.arange(seq_len * nenv, device=device), each_seq_len.long()] = 1.0
        mask = torch.logical_not(mask.cumsum(dim=1))
        mask = mask[:, :-1].unsqueeze(-2)
        return mask

    def forward(self, inp, each_seq_len):
        seq_len, nenv, max_human_num, _ = inp.size()
        # sort_humans=True: each_seq_len is the detected-human count.
        attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)
        attn_mask = attn_mask.squeeze(1)

        input_emb = self.embedding_layer(inp).view(seq_len * nenv, max_human_num, -1)
        input_emb = torch.transpose(input_emb, 0, 1)
        q = self.q_linear(input_emb)
        k = self.k_linear(input_emb)
        v = self.v_linear(input_emb)
        z, _ = self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask))
        z = torch.transpose(z, 0, 1)
        return z


class EdgeAttention_M(nn.Module):
    """Robot-human attention module."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size
        self.temporal_edge_layer = nn.ModuleList()
        self.spatial_edge_layer = nn.ModuleList()
        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))
        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        device = each_seq_len.device
        mask = torch.zeros(seq_len * nenv, max_human_num + 1, device=device)
        mask[torch.arange(seq_len * nenv, device=device), each_seq_len.long()] = 1.0
        mask = torch.logical_not(mask.cumsum(dim=1))
        mask = mask[:, :-1].unsqueeze(-2)
        return mask

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        seq_len, nenv, num_edges, h_size = h_spatials.size()
        attn = temporal_embed * spatial_embed
        attn = torch.sum(attn, dim=3)
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)
        attn = attn.view(seq_len, nenv, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(
            seq_len * nenv * self.agent_num, self.human_num, h_size
        ).permute(0, 2, 1)
        attn = attn.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)
        weighted_value = torch.bmm(h_spatials, attn)
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)
        return weighted_value, attn

    def forward(self, h_temporal, h_spatials, each_seq_len):
        seq_len, nenv, max_human_num, _ = h_spatials.size()
        self.human_num = max_human_num // self.agent_num
        weighted_value_list, attn_list = [], []
        for i in range(self.num_attention_head):
            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            spatial_embed = self.spatial_edge_layer[i](h_spatials)
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)
            attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)
            attn_mask = attn_mask.squeeze(-2).view(seq_len, nenv, max_human_num)
            weighted_value, attn = self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)
        return weighted_value_list[0], attn_list[0]


class EndRNN(RNNBase):
    """GRU node RNN for the robot node."""

    def __init__(self, args):
        super().__init__(args, edge=False)
        self.args = args
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size
        self.encoder_linear = nn.Linear(256, self.embedding_size)
        self.relu = nn.ReLU()
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, robot_s, h_spatial_other, h, masks):
        encoded_input = self.relu(self.encoder_linear(robot_s))
        h_edges_embedded = self.relu(self.edge_attention_embed(h_spatial_other))
        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)
        x, h_new = self._forward_gru(concat_encoded, h, masks)
        outputs = self.output_linear(x)
        return outputs, h_new


class selfAttn_merge_SRNN(nn.Module):
    """The conformal-RL policy network (selfAttn_merge_srnn base)."""

    def __init__(self, obs_space_dict, args, config, infer=False):
        super().__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args = args
        self.config = config
        self.human_num = obs_space_dict["spatial_edges"].shape[0]
        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        self.humanNodeRNN = EndRNN(args)
        self.attn = EdgeAttention_M(args)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        num_inputs = hidden_size = self.output_size
        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
        )
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        robot_size = 9
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU())
        self.human_node_final_linear = init_(nn.Linear(self.output_size, 2))

        self.spatial_attn = SpatialEdgeSelfAttn(args)
        self.spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())

        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num + 1)

    def forward(self, inputs, rnn_hxs, masks, infer=True):
        # inference path only: seq_length = 1, nenv = self.nenv (set to 1 by caller)
        seq_length = 1
        nenv = self.nenv

        robot_node = reshapeT(inputs["robot_node"], seq_length, nenv)
        temporal_edges = reshapeT(inputs["temporal_edges"], seq_length, nenv)
        spatial_edges = reshapeT(inputs["spatial_edges"], seq_length, nenv)
        conformity_scores = reshapeT(inputs["conformity_scores"], seq_length, nenv)

        detected_human_num = inputs["detected_human_num"].squeeze(-1).int()

        hidden_states_node_RNNs = reshapeT(rnn_hxs["human_node_rnn"], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        all_hidden_states_edge_RNNs = Variable(
            torch.zeros(1, nenv, 1 + self.human_num, rnn_hxs["human_human_edge_rnn"].size()[-1],
                        device=spatial_edges.device)
        )

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)

        # aci_input=True: concat conformity scores onto the spatial edge features.
        spatial_edges = torch.cat((spatial_edges, conformity_scores), dim=-1)

        spatial_attn_out = self.spatial_attn(spatial_edges, detected_human_num).view(
            seq_length, nenv, self.human_num, -1
        )
        output_spatial = self.spatial_linear(spatial_attn_out)

        hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, detected_human_num)

        outputs, h_nodes = self.humanNodeRNN(
            robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks
        )

        rnn_hxs["human_node_rnn"] = h_nodes
        rnn_hxs["human_human_edge_rnn"] = all_hidden_states_edge_RNNs

        x = outputs[:, :, 0, :]
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs


# --- upstream rl/networks/model.py (Policy) --------------------------------
class Policy(nn.Module):
    def __init__(self, obs_shape, action_dim, args, config):
        super().__init__()
        self.base = selfAttn_merge_SRNN(obs_shape, args, config)
        self.srnn = True
        self.dist = DiagGaussian(self.base.output_size, action_dim)

    def act(self, inputs, rnn_hxs, masks, deterministic=True):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)
        dist = self.dist(actor_features)
        action = dist.mode() if deterministic else dist.sample()
        return value, action, rnn_hxs
