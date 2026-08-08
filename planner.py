"""SoNIC (SoNIC_GST, constrained-RL + ACI safety + GST prediction) wrapper for arena_planners.

SoNIC shares the GenSafeNav lineage: the same ``selfAttn_merge_srnn`` policy
network (with ``aci_input`` conformity scores) and the same GST human-trajectory
predictor; only the trained policy checkpoint differs (SoNIC_GST/05207.pt).

Replicates the deterministic eval/inference path of upstream ``test.py`` /
``rl/evaluation.py`` outside the ``crowd_sim`` simulator:

1. A 5-frame per-pedestrian position history feeds the GST human-trajectory
   predictor (``gst_updated`` st_model, same wiring as the attngraph planner) to
   produce 5 predicted future positions per ped.
2. Per ped, an adaptive-conformal-inference (DtACI) bank of 5 quantile predictors
   (``dt_aci/one_step_dtai.py``) yields per-step conformity scores. These are the
   inference-time conformal/safety step: upstream's ``baseEnv.talk2Env`` runs the
   same online ACI update from the GST predictions and the realized ped positions,
   then feeds ``obs['conformity_scores']`` into the policy. We track each ped's
   predicted-vs-realized nonconformity online and replicate it deterministically
   (eval noise std/clip are set to 0 in upstream test.py).
3. The selfAttn_merge_srnn policy (``srnn_net.py``) consumes
   robot_node + temporal_edges + spatial_edges(+conformity) and outputs a
   holonomic (vx, vy) twist (deterministic = distribution mean).

The bridge applies no diff-drive projection (this planner is omnidirectional).
Checkpoints shipped in ``model/``: ``policy.pt`` (SoNIC_GST/05207.pt),
``gst_predictor.pt`` + ``gst_args.pickle`` (GST_predictor_rand epoch_100).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np
import torch

from arena_planners.sdk import load_manifest, main_loop

# Vendored upstream code (gst_updated) lives flat under this dir; expose on sys.path
# so its `from gst_updated...` absolute imports resolve.
sys.path.insert(0, os.path.dirname(__file__))

from srnn_net import Policy  # noqa: E402

_HERE = pathlib.Path(__file__).parent
_MODEL_DIR = _HERE / "model"
_POLICY_WEIGHTS = _MODEL_DIR / "policy.pt"
_GST_WEIGHTS = _MODEL_DIR / "gst_predictor.pt"
_GST_ARGS = _MODEL_DIR / "gst_args.pickle"

_V_PREF = 1.0
_RADIUS = 0.3
_MAX_HUMAN_NUM = 20          # config.sim.human_num
_PREDICT_STEPS = 5          # config.sim.predict_steps
_OBS_SEQ_LEN = 5            # GST obs window
_PRED_HORIZON_ACI = 5      # human.pred_horizon_aci
_ACI_ALPHA = 0.1            # config.aci_related.alpha
_INVALID_POS = -999.0
_PRED_VALID_DIST = 0.1     # talk2Env: prediction valid if start within 0.1 of human pos
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- DtACI (inlined from upstream dt_aci/one_step_dtai.py, sans matplotlib) --
def _vec_zero_min(x):
    return np.minimum(x, 0)


def _pinball(u, alpha):
    return alpha * u - _vec_zero_min(u)


class DtACI:
    """Adaptive Conformal Inference quantile tracker (dynamically-tuned ACI)."""

    def __init__(self, alpha=0.1, gammas=np.array([0.05, 0.1, 0.2]),
                 sigma=1 / 1000, eta=2.72, initial_pred=0.5):
        self.alpha = alpha
        self.gammas = gammas
        self.sigma = sigma
        self.eta = eta
        self.true_values = []
        self.num_experts = len(gammas)
        self.initial_pred = initial_pred
        self.expert_predictions = np.full(self.num_experts, initial_pred)
        self.expert_weights = np.ones(self.num_experts)
        self.current_expert = np.random.choice(np.arange(self.num_experts))
        self.expert_probs = np.full(self.num_experts, 1 / self.num_experts)
        self.last_prediction = initial_pred

    def make_prediction(self):
        if len(self.true_values) == 0:
            self.last_prediction = self.initial_pred
            return self.initial_pred
        prediction = self.expert_predictions[self.current_expert]
        self.last_prediction = prediction
        return prediction

    def update_true_value(self, new_true_value):
        self.true_values.append(new_true_value)
        truth = new_true_value
        expert_losses = _pinball(truth - self.expert_predictions, self.alpha)
        self.expert_predictions -= self.gammas * (self.alpha - (self.expert_predictions < truth).astype(float))
        if self.eta < float("inf"):
            expert_bar_weights = self.expert_weights * np.exp(-self.eta * expert_losses)
            expert_next_weights = (1 - self.sigma) * expert_bar_weights / np.sum(expert_bar_weights) \
                + self.sigma / self.num_experts
            self.expert_probs = expert_next_weights / np.sum(expert_next_weights)
            self.current_expert = np.random.choice(np.arange(self.num_experts), p=self.expert_probs)
            self.expert_weights = expert_next_weights


# --- args for the policy net -----------------------------------------------
def _build_args() -> argparse.Namespace:
    a = argparse.Namespace()
    a.human_node_rnn_size = 128
    a.human_human_edge_rnn_size = 256
    a.human_node_input_size = 3
    a.human_human_edge_input_size = 2
    a.human_node_output_size = 256
    a.human_node_embedding_size = 64
    a.human_human_edge_embedding_size = 256
    a.attention_size = 64
    a.seq_length = 30
    a.num_processes = 1
    a.num_mini_batch = 1
    a.use_self_attn = True
    a.sort_humans = True
    a.no_cuda = _DEVICE.type == "cpu"
    a.env_name = "CrowdSimPredRealGST-v0"
    return a


class _Config:
    """Minimal config exposing only the fields the net reads."""

    class _Policy:
        aci_input = True
        constant_std = True

    policy = _Policy()


class _Runner:
    def __init__(self) -> None:
        self.args = _build_args()
        self.config = _Config()
        self.human_num = _MAX_HUMAN_NUM

        obs_space = {"spatial_edges": _ShapeStub((_MAX_HUMAN_NUM, 2 * (_PREDICT_STEPS + 1)))}
        self.policy = Policy(obs_space, action_dim=2, args=self.args, config=self.config)
        self.policy.base.nenv = 1
        state = torch.load(str(_POLICY_WEIGHTS), map_location=_DEVICE)
        if isinstance(state, (list, tuple)):
            state = state[0]
        self.policy.load_state_dict(state, strict=True)
        self.policy.to(_DEVICE)
        self.policy.eval()

        self.gst = self._build_gst()
        self.reset_state()

    @staticmethod
    def _build_gst():
        from gst_updated.src.gumbel_social_transformer.st_model import st_model

        with open(_GST_ARGS, "rb") as fh:
            gst_args = pickle.load(fh)
        model = st_model(gst_args, device=_DEVICE).to(_DEVICE)
        ckpt = torch.load(str(_GST_WEIGHTS), map_location=_DEVICE)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model

    def reset_state(self) -> None:
        n = self.human_node_rnn_size = self.args.human_node_rnn_size
        e = self.args.human_human_edge_rnn_size
        self.rnn_hxs = {
            "human_node_rnn": torch.zeros(1, 1, n, device=_DEVICE),
            "human_human_edge_rnn": torch.zeros(1, 1 + _MAX_HUMAN_NUM, e, device=_DEVICE),
        }
        self.masks = torch.zeros(1, 1, device=_DEVICE)
        # per-id rolling GST history + ACI state
        self.traj_buffer = [np.full((_MAX_HUMAN_NUM, 2), _INVALID_POS, dtype=np.float32) for _ in range(_OBS_SEQ_LEN)]
        self.mask_buffer = [np.zeros(_MAX_HUMAN_NUM, dtype=bool) for _ in range(_OBS_SEQ_LEN)]
        self.ped_id_slot: dict[int, int] = {}
        # per-id ACI state: bank of predictors + history of predictions / gt positions
        self.aci: dict[int, dict] = {}

    # -- pedestrian slotting (stable per-id rows) --
    def _slot_for(self, ped_id: int):
        if ped_id in self.ped_id_slot:
            return self.ped_id_slot[ped_id]
        used = set(self.ped_id_slot.values())
        for s in range(_MAX_HUMAN_NUM):
            if s not in used:
                self.ped_id_slot[ped_id] = s
                return s
        return None

    def _aci_for(self, ped_id: int) -> dict:
        if ped_id not in self.aci:
            self.aci[ped_id] = {
                "predictors": [DtACI(alpha=_ACI_ALPHA, initial_pred=(i + 1) / 10) for i in range(_PRED_HORIZON_ACI)],
                "predictions": [],   # list of (pred_horizon+1, 2) absolute predicted traj
                "gt": [],            # list of (2,) realized positions
            }
        return self.aci[ped_id]

    # -- GST forward (mirrors attngraph _run_gst / vec_pretext_normalize) --
    def run_gst(self, traj_hist: np.ndarray, mask_hist: np.ndarray):
        in_traj = torch.from_numpy(traj_hist).to(_DEVICE).float().unsqueeze(0)
        in_mask = torch.from_numpy(mask_hist.astype(np.float32)).to(_DEVICE).unsqueeze(0).unsqueeze(-1)
        obs_traj = in_traj.permute(0, 1, 3, 2)  # (1, n, 2, 5)
        n_env, num_peds = obs_traj.shape[:2]
        loss_mask_obs = in_mask[:, :, :, 0]
        loss_mask_rel_obs = loss_mask_obs[:, :, :-1] * loss_mask_obs[:, :, -1:]
        loss_mask_rel_obs = torch.cat((loss_mask_obs[:, :, :1], loss_mask_rel_obs), dim=2)
        loss_mask_rel_pred = torch.ones((n_env, num_peds, _PREDICT_STEPS), device=_DEVICE) * loss_mask_rel_obs[:, :, -1:]
        loss_mask_rel = torch.cat((loss_mask_rel_obs, loss_mask_rel_pred), dim=2)
        loss_mask_rel_obs_p = loss_mask_rel_obs.permute(0, 2, 1).reshape(n_env * _OBS_SEQ_LEN, num_peds)
        attn_mask_obs = torch.bmm(loss_mask_rel_obs_p.unsqueeze(2), loss_mask_rel_obs_p.unsqueeze(1))
        attn_mask_obs = attn_mask_obs.reshape(n_env, _OBS_SEQ_LEN, num_peds, num_peds)
        obs_traj_rel = obs_traj[:, :, :, 1:] - obs_traj[:, :, :, :-1]
        obs_traj_rel = torch.cat((torch.zeros(n_env, num_peds, 2, 1, device=_DEVICE), obs_traj_rel), dim=3)
        obs_traj_rel = _INVALID_POS * torch.ones_like(obs_traj_rel) * (1 - loss_mask_rel_obs.unsqueeze(2)) \
            + obs_traj_rel * loss_mask_rel_obs.unsqueeze(2)
        v_obs = obs_traj_rel.permute(0, 3, 1, 2)
        seq_p = obs_traj.permute(0, 3, 1, 2)
        a_obs = seq_p.unsqueeze(3) - seq_p.unsqueeze(2)
        with torch.no_grad():
            gaussian_params, _, _ = self.gst(v_obs, a_obs, attn_mask_obs, loss_mask_rel,
                                             tau=0.03, hard=True, sampling=False, device=_DEVICE)
        mu = gaussian_params[0]                       # (1, pred_seq_len, num_peds, 2) displacements
        mu = mu.cumsum(1) + obs_traj.permute(0, 3, 1, 2)[:, -1:]   # absolute positions
        mu = mu.squeeze(0).permute(1, 0, 2).cpu().numpy()         # (num_peds, pred_seq_len, 2)
        valid = loss_mask_rel_pred[0, :, 0].cpu().numpy().astype(bool)
        return mu, valid

    def act(self, obs):
        with torch.no_grad():
            _, action, self.rnn_hxs = self.policy.act(obs, self.rnn_hxs, self.masks, deterministic=True)
        self.masks = torch.ones(1, 1, device=_DEVICE)
        return action.reshape(-1).cpu().numpy()


class _ShapeStub:
    def __init__(self, shape):
        self.shape = shape


_runner: _Runner | None = None


def _get_runner() -> _Runner:
    global _runner
    if _runner is None:
        _runner = _Runner()
    return _runner


def step(features: dict) -> list[float]:
    runner = _get_runner()

    robot_pose = features.get("robot_pose")
    robot_state = features.get("robot_state")
    if robot_pose is None:
        return [0.0, 0.0]
    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    if robot_state is not None and len(robot_state) >= 4:
        vx, vy = float(robot_state[2]), float(robot_state[3])
    else:
        vx, vy = 0.0, 0.0

    goal_pose = features.get("goal_pose")
    target = None
    if goal_pose is not None:
        target = (float(goal_pose[0]), float(goal_pose[1]))
    if target is None:
        return [0.0, 0.0]
    gx, gy = target

    # -- slot pedestrians into stable per-id rows --
    peds = features.get("pedestrians")
    if peds is None:
        peds = []
    cur_pos = np.full((_MAX_HUMAN_NUM, 2), _INVALID_POS, dtype=np.float32)
    cur_mask = np.zeros(_MAX_HUMAN_NUM, dtype=bool)
    slot_to_id: dict[int, int] = {}
    seen_slots: set[int] = set()
    for ped in peds[:_MAX_HUMAN_NUM]:
        pid = int(ped[0])
        slot = runner._slot_for(pid)
        if slot is None:
            continue
        cur_pos[slot] = (float(ped[1]), float(ped[2]))
        cur_mask[slot] = True
        slot_to_id[slot] = pid
        seen_slots.add(slot)
    for pid, slot in list(runner.ped_id_slot.items()):
        if slot not in seen_slots:
            del runner.ped_id_slot[pid]
            runner.aci.pop(pid, None)

    runner.traj_buffer.append(cur_pos)
    runner.mask_buffer.append(cur_mask)
    del runner.traj_buffer[0]
    del runner.mask_buffer[0]
    traj_hist = np.stack(runner.traj_buffer, axis=1)  # (max_human_num, 5, 2)
    mask_hist = np.stack(runner.mask_buffer, axis=1)  # (max_human_num, 5)

    pred_pos, pred_valid = runner.run_gst(traj_hist, mask_hist)  # (max_human_num, 5, 2)

    # -- per-ped online ACI update + conformity-score prediction (talk2Env) --
    conformity_full = np.zeros((_MAX_HUMAN_NUM, _PRED_HORIZON_ACI), dtype=np.float32)
    for slot in range(_MAX_HUMAN_NUM):
        if not cur_mask[slot]:
            continue
        pid = slot_to_id[slot]
        st = runner._aci_for(pid)
        # prediction valid only if the predictor's first step matches the current pos.
        valid = bool(pred_valid[slot])
        # full predicted traj incl current frame as step 0 (talk2Env predictions[0..predict_steps]).
        pred_traj = np.zeros((_PREDICT_STEPS + 1, 2), dtype=np.float32)
        pred_traj[0] = cur_pos[slot]
        for k in range(_PREDICT_STEPS):
            pred_traj[1 + k] = pred_pos[slot, k]
        if valid:
            st["predictions"].append(pred_traj.copy())
        # online ACI update: compare past i-step-ahead prediction to realized pos (update_aci).
        st["gt"].append(cur_pos[slot].copy())
        if len(st["gt"]) >= 2 and len(st["predictions"]) >= 1:
            curr = st["gt"][-1]
            num = min(_PRED_HORIZON_ACI, len(st["gt"]) - 1, len(st["predictions"]))
            for i in range(num):
                past_pred = st["predictions"][-(i + 1)]
                predicted_curr = past_pred[i + 1]
                nonconf = float(np.linalg.norm(curr - predicted_curr))
                st["predictors"][i].update_true_value(nonconf)
        # read out current conformity scores (clipped to [0,1]).
        if valid:
            for i, predr in enumerate(st["predictors"]):
                conformity_full[slot, i] = float(np.clip(predr.make_prediction(), 0.0, 1.0))

    # -- build obs: spatial_edges (current rel + predicted rel), conformity, masks --
    spatial = np.full((_MAX_HUMAN_NUM, 2 * (_PREDICT_STEPS + 1)), 15.0, dtype=np.float32)
    visible = np.zeros(_MAX_HUMAN_NUM, dtype=bool)
    for i in range(_MAX_HUMAN_NUM):
        if not cur_mask[i]:
            continue
        spatial[i, 0] = cur_pos[i, 0] - px
        spatial[i, 1] = cur_pos[i, 1] - py
        if pred_valid[i]:
            for k in range(_PREDICT_STEPS):
                spatial[i, 2 + 2 * k] = pred_pos[i, k, 0] - px
                spatial[i, 3 + 2 * k] = pred_pos[i, k, 1] - py
        else:
            for k in range(_PREDICT_STEPS):
                spatial[i, 2 + 2 * k] = spatial[i, 0]
                spatial[i, 3 + 2 * k] = spatial[i, 1]
        visible[i] = True

    # sort by distance to robot (matches talk2Env + process_obs_rew ordering).
    dists = np.linalg.norm(spatial[:, :2], axis=1)
    order = np.argsort(dists)
    spatial = spatial[order]
    visible = visible[order]
    conformity = conformity_full[order]
    detected_n = max(int(visible.sum()), 1)

    obs = {
        "robot_node": torch.tensor([[px, py, _RADIUS, gx, gy, _V_PREF, theta]],
                                   dtype=torch.float32, device=_DEVICE).unsqueeze(0),
        "temporal_edges": torch.tensor([[vx, vy]], dtype=torch.float32, device=_DEVICE).unsqueeze(0),
        "spatial_edges": torch.from_numpy(spatial).to(_DEVICE).unsqueeze(0),
        "conformity_scores": torch.from_numpy(conformity).to(_DEVICE).unsqueeze(0),
        "visible_masks": torch.from_numpy(visible).to(_DEVICE).unsqueeze(0),
        "detected_human_num": torch.tensor([[float(detected_n)]], device=_DEVICE),
    }

    raw = runner.act(obs)  # holonomic (vx, vy) world frame
    speed = float(np.linalg.norm(raw))
    if speed > _V_PREF:
        raw = raw / speed * _V_PREF
    return [float(raw[0]), float(raw[1])]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    _get_runner().reset_state()


if __name__ == "__main__":
    manifest = load_manifest(_HERE / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
