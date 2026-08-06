import os
import time
import pickle
import random
import math
import threading
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import panda_mujoco_gym
import pyspacemouse
from scipy.signal import savgol_filter

# Import wrappers from smooth_env.py
from smooth_env import (
    FlattenGoalEnv, 
    RelativeGoalWrapper, 
    FixDoneWrapper
)

# ===== ACT TUNABLE PARAMETERS & CONFIG =====
CONTROL_HZ = 100
EE_Z_OFFSET = 0.068  # Exact 6.8cm offset from hand frame to fingertip center
SENSITIVITY = 8.0  

CHUNK_SIZE = 30       # Predict 30 future steps at once (0.3s trajectory horizon)
LATENT_DIM = 16       # CVAE Latent variable dimension
HIDDEN_DIM = 128      # Transformer hidden dimension (Optimized for CPU/GPU)
KL_WEIGHT = 10.0      # Weight for CVAE KL-Divergence loss

TASK_CONFIG = {
    "push": {
        "env_id": "FrankaPushSparse-v0",
        "act_dim": 3,
        "default_data_path": "operator_data_push_spacemouse.pkl",
    },
    "pick_and_place": {
        "env_id": "FrankaPickAndPlaceSparse-v0",
        "act_dim": 4,
        "default_data_path": "operator_data_pick_and_place_spacemouse.pkl",
    },
}


# =====================================================================
# 1. SMART BINARY GRIPPER WRAPPER (WITH PROXIMITY LATCH)
# =====================================================================

class SmartBinaryGripperActionWrapper(gym.ActionWrapper):
    """Action wrapper with close threshold shift (0.2) and exact EE-to-block distance latching."""
    def __init__(self, env, close_thresh=0.2, open_thresh=0.6):
        super().__init__(env)
        self.close_thresh = close_thresh
        self.open_thresh = open_thresh
        self.is_grasped = False

    def reset(self, **kwargs):
        self.is_grasped = False
        return self.env.reset(**kwargs)

    def action(self, action):
        if action.shape[0] < 4:
            return action

        act = np.array(action, copy=True)
        gripper_cmd = act[3]

        # Reset latch if operator explicitly commands OPEN (> 0.8) during takeover
        if gripper_cmd > 0.8:
            self.is_grasped = False

        # Close if latched or command is below close_thresh
        if self.is_grasped or gripper_cmd < self.close_thresh:
            act[3] = -1.0
        else:
            act[3] = 1.0

        return act

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        
        # Calculate EXACT 3D distance from EE to block using relative obs[:3]
        raw_obs = obs[-25:] if obs.shape[0] > 50 else obs
        dist_ee_block = info.get("dist_ee_block", 1.0)

        # PROXIMITY LATCH: If within 3.5cm of block and action shows intent to close (< 0.5), latch closed
        if dist_ee_block < 0.035 and action[3] < 0.5:
            self.is_grasped = True

        return obs, reward, terminated, truncated, info


# =====================================================================
# 2. ACT MODEL ARCHITECTURE (CVAE + TRANSFORMER DECODER)
# =====================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100):
        super().__init__()
        pe = th.zeros(max_len, dim)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)].unsqueeze(1)


class CVAEEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_size, hidden_dim=128, latent_dim=16):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, 128)
        self.act_proj = nn.Linear(chunk_size * act_dim, 128)
        
        self.net = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs, action_chunk):
        batch_size = obs.size(0)
        flat_act = action_chunk.view(batch_size, -1)
        
        h_obs = F.relu(self.obs_proj(obs))
        h_act = F.relu(self.act_proj(flat_act))
        
        h = th.cat([h_obs, h_act], dim=-1)
        feat = self.net(h)
        
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar


class ACTPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim=4, chunk_size=CHUNK_SIZE, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, nheads=8, num_decoder_layers=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.cvae_encoder = CVAEEncoder(obs_dim, act_dim, chunk_size, hidden_dim, latent_dim)

        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len=chunk_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=nheads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=0.1, 
            activation='relu', 
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, action_chunk=None):
        batch_size = obs.size(0)

        if self.training and action_chunk is not None:
            mu, logvar = self.cvae_encoder(obs, action_chunk)
            std = th.exp(0.5 * logvar)
            eps = th.randn_like(std)
            z = mu + eps * std
        else:
            mu, logvar = None, None
            z = th.zeros(batch_size, self.latent_dim, device=obs.device)

        h_obs = self.obs_proj(obs)
        h_z = self.latent_proj(z)
        memory_token = h_obs + h_z
        memory = memory_token.unsqueeze(0)

        queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        queries = self.pos_encoder(queries)

        decoded_feats = self.transformer_decoder(tgt=queries, memory=memory)
        decoded_feats = decoded_feats.transpose(0, 1)
        pred_actions = self.action_head(decoded_feats)

        return pred_actions, mu, logvar


# =====================================================================
# 3. TEMPORAL ENSEMBLING (WITH CRISP GRIPPER EXECUTION)
# =====================================================================

class TemporalEnsemble:
    """Ensembles overlapping XYZ predictions while keeping binary gripper commands crisp."""
    def __init__(self, act_dim=4, max_steps=1000, chunk_size=CHUNK_SIZE, exp_weight=0.01):
        self.act_dim = act_dim
        self.max_steps = max_steps
        self.chunk_size = chunk_size
        self.exp_weight = exp_weight
        
        buffer_len = max_steps + chunk_size + 100
        self.all_predictions = np.zeros((buffer_len, act_dim), dtype=np.float32)
        self.weights = np.zeros((buffer_len, 1), dtype=np.float32)
        self.latest_gripper_pred = 1.0

    def update(self, t, predicted_chunk):
        if self.act_dim == 4:
            # Store immediate step 0 gripper prediction from the latest chunk
            self.latest_gripper_pred = float(predicted_chunk[0, 3])

        for i in range(self.chunk_size):
            step_idx = t + i
            w = np.exp(-self.exp_weight * i)
            self.all_predictions[step_idx] += w * predicted_chunk[i]
            self.weights[step_idx] += w

    def get_action(self, t):
        if self.weights[t] == 0:
            return np.zeros(self.act_dim, dtype=np.float32)
        
        # Smooth spatial XYZ actions across overlapping chunks
        spatial_act = self.all_predictions[t][:3] / self.weights[t]
        
        if self.act_dim == 4:
            # Crisp gripper execution (no temporal smoothing blur)
            gripper_act = -1.0 if self.latest_gripper_pred < 0.2 else 1.0
            return np.append(spatial_act, gripper_act)
        
        return spatial_act


# =====================================================================
# 4. SPACEMOUSE THREAD & TAKEOVER SOURCE
# =====================================================================

class SpaceMouseThread:
    def __init__(self, device):
        self.device = device
        self.latest_state = None
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def _poll_loop(self):
        while self.running:
            try:
                state = self.device.read()
                if state is not None:
                    self.latest_state = state
            except Exception:
                pass
            time.sleep(0.001)

    def get_state(self):
        return self.latest_state

    def stop(self):
        self.running = False


class SpaceMouseTakeoverSource:
    GLITCH_FILTER_ALPHA = 0.55
    DEADZONE = 0.05

    def __init__(self):
        print("\nConnecting to SpaceMouse USB HID...")
        devices = pyspacemouse.get_all_hid_devices()
        if not devices:
            print("❌ ERROR: No SpaceMouse detected!")
            
        self._cm = pyspacemouse.open()
        device = self._cm.__enter__()
        self.sm_thread = SpaceMouseThread(device)
        self.filtered_raw = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.gripper_val = 1.0  
        self._abort = False
        print("✅ SpaceMouse Connected & Polling!\n")

    def sync_gripper_state(self, current_env_gripper):
        state = self.sm_thread.get_state()
        if state is not None:
            left, right = state.buttons
            if not (left or right):
                self.gripper_val = current_env_gripper

    def get_human_action(self, actual_dt: float, act_dim: int):
        state = self.sm_thread.get_state()
        if state is None:
            if act_dim == 4:
                return np.array([0.0, 0.0, 0.0, self.gripper_val], dtype=np.float32), False
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), False

        left, right = state.buttons
        if left and right:
            self._abort = True
            print("\n[INFO] Episode aborted by operator.")
        elif left:
            self.gripper_val = -1.0
            print("   [SPACEMOUSE] Gripper -> CLOSED")
        elif right:
            self.gripper_val = 1.0
            print("   [SPACEMOUSE] Gripper -> OPEN")

        a = self.GLITCH_FILTER_ALPHA
        self.filtered_raw["x"] = (1 - a) * self.filtered_raw["x"] + a * state.x
        self.filtered_raw["y"] = (1 - a) * self.filtered_raw["y"] + a * state.y
        self.filtered_raw["z"] = (1 - a) * self.filtered_raw["z"] + a * state.z

        def smooth_deadzone(val):
            if abs(val) < self.DEADZONE:
                return 0.0
            return np.sign(val) * (abs(val) - self.DEADZONE) / (1.0 - self.DEADZONE)

        raw_x = smooth_deadzone(self.filtered_raw["x"])
        raw_y = smooth_deadzone(self.filtered_raw["y"])
        raw_z = smooth_deadzone(self.filtered_raw["z"])

        dx = raw_x * SENSITIVITY * actual_dt
        dy = raw_y * SENSITIVITY * actual_dt
        dz = raw_z * SENSITIVITY * actual_dt

        human_active = (abs(raw_x) > 0.0 or abs(raw_y) > 0.0 or abs(raw_z) > 0.0 or left or right)

        if act_dim == 4:
            action = np.array([dy, dx, dz, self.gripper_val], dtype=np.float32)
        else:
            action = np.array([dy, dx, dz], dtype=np.float32)

        return action, human_active

    def should_abort(self):
        return self._abort

    def reset_state(self):
        self.filtered_raw = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.gripper_val = 1.0
        self._abort = False

    def close(self):
        self.sm_thread.stop()
        self._cm.__exit__(None, None, None)


# =====================================================================
# 5. ENVIRONMENT FACTORY
# =====================================================================

def make_env(env_id="FrankaPickAndPlaceSparse-v0", max_steps=1000, render_mode=None):
    raw_env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_steps)
    flat_env = FlattenGoalEnv(raw_env)
    grip_env = SmartBinaryGripperActionWrapper(flat_env, close_thresh=0.2, open_thresh=0.6)
    rel_env = RelativeGoalWrapper(grip_env, ee_z_offset=EE_Z_OFFSET)
    fixed_env = FixDoneWrapper(rel_env)
    return fixed_env


# =====================================================================
# 6. DATASET FILTERING & ACTION CHUNK FORMATTING
# =====================================================================

def filter_trajectories(trajectories):
    filtered = []
    for traj in trajectories:
        raw_obs = traj["obs"]
        raw_acts = traj["acts"]

        if len(raw_acts) < 10:
            continue

        aligned_obs = np.array(raw_obs, dtype=np.float32)
        aligned_acts = np.array(raw_acts, dtype=np.float32)

        if len(aligned_acts) >= 5 and aligned_acts.shape[1] >= 3:
            window_len = min(11, len(aligned_acts) if len(aligned_acts) % 2 != 0 else len(aligned_acts) - 1)
            if window_len > 3:
                aligned_acts[:, :3] = savgol_filter(aligned_acts[:, :3], window_length=window_len, polyorder=3, axis=0)

        filtered.append({
            "obs": aligned_obs,
            "acts": aligned_acts,
            "terminal": traj["terminal"],
        })
    return filtered


def format_dataset_for_act(raw_trajectories, rel_transformer, act_dim=4, chunk_size=CHUNK_SIZE):
    obs_samples = []
    chunk_samples = []

    for traj in raw_trajectories:
        raw_obs = traj["obs"]
        raw_acts = traj["acts"]
        N = len(raw_acts)

        if N < 10:
            continue

        close_idx = None
        if act_dim == 4:
            for idx, act in enumerate(raw_acts):
                if len(act) >= 4 and act[3] < 0.2:
                    close_idx = idx
                    break

        transformed_obs = []
        clean_acts = []
        for i in range(N):
            base_obs = raw_obs[i][:25].copy()
            g_state = -1.0 if (close_idx is not None and i >= close_idx) else 1.0
            obs_i = rel_transformer.observation(base_obs, override_gripper=g_state)
            
            target_act = np.array(raw_acts[i], copy=True)
            if act_dim == 4:
                target_act[3] = -1.0 if (close_idx is not None and i >= close_idx) else 1.0

            transformed_obs.append(obs_i)
            clean_acts.append(target_act)

        transformed_obs = np.array(transformed_obs, dtype=np.float32)
        clean_acts = np.array(clean_acts, dtype=np.float32)

        for t in range(N):
            obs_t = transformed_obs[t]
            chunk = clean_acts[t : t + chunk_size]
            if len(chunk) < chunk_size:
                pad_count = chunk_size - len(chunk)
                last_act = clean_acts[-1]
                padding = np.tile(last_act, (pad_count, 1))
                chunk = np.vstack([chunk, padding])

            obs_samples.append(obs_t)
            chunk_samples.append(chunk)

    obs_tensor = th.tensor(np.array(obs_samples), dtype=th.float32)
    chunks_tensor = th.tensor(np.array(chunk_samples), dtype=th.float32)
    return obs_tensor, chunks_tensor


# =====================================================================
# 7. ACT TRAINING LOOP
# =====================================================================

def train_act_model(
    model,
    train_obs,
    train_chunks,
    val_obs,
    val_chunks,
    max_epochs=30,
    patience=5,
    lr=1e-4,
    batch_size=64,
    temp_save_path="best_act_model.pt"
):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = model.to(device)

    train_obs, train_chunks = train_obs.to(device), train_chunks.to(device)
    val_obs, val_chunks = val_obs.to(device), val_chunks.to(device)

    optimizer = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    num_samples = len(train_obs)
    num_val_samples = len(val_obs)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n--- Training ACT Model ({num_samples} train | {num_val_samples} val | Device: {device}) ---")

    for epoch in range(1, max_epochs + 1):
        model.train()
        indices = th.randperm(num_samples)
        train_l1_losses = []
        train_kl_losses = []

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            b_obs = train_obs[batch_idx]
            b_chunks = train_chunks[batch_idx]

            pred_chunks, mu, logvar = model(b_obs, b_chunks)

            l1_loss = F.l1_loss(pred_chunks, b_chunks)
            kl_loss = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b_obs.size(0)

            loss = l1_loss + KL_WEIGHT * kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_l1_losses.append(l1_loss.item())
            train_kl_losses.append(kl_loss.item())

        mean_l1 = np.mean(train_l1_losses)
        mean_kl = np.mean(train_kl_losses)

        # VALIDATION (Mini-batched for CPU)
        model.eval()
        val_l1_losses = []
        with th.no_grad():
            for v_start in range(0, num_val_samples, batch_size):
                v_obs = val_obs[v_start : v_start + batch_size]
                v_chunks = val_chunks[v_start : v_start + batch_size]
                pred_v_chunks, _, _ = model(v_obs, None)
                v_loss = F.l1_loss(pred_v_chunks, v_chunks).item()
                val_l1_losses.append(v_loss)

        val_l1_loss = np.mean(val_l1_losses)

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train L1: {mean_l1:.4f} | KL: {mean_kl:.4f} | Val L1: {val_l1_loss:.4f}")

        if val_l1_loss < best_val_loss:
            best_val_loss = val_l1_loss
            patience_counter = 0
            th.save(model.state_dict(), temp_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"✋ Early stopping triggered at Epoch {epoch}! Best Val L1 Loss: {best_val_loss:.4f}")
                break

    model.load_state_dict(th.load(temp_save_path, map_location=device))
    return model


# =====================================================================
# 8. INTERACTIVE DAGGER ROLLOUTS WITH TEMPORAL ENSEMBLING
# =====================================================================

def collect_spacemouse_act_interventions(model, sm_source, task_name="pick_and_place", num_episodes=3, max_steps=1000):
    cfg = TASK_CONFIG[task_name]
    env_id = cfg["env_id"]
    act_dim = cfg["act_dim"]

    print("\n" + "="*65)
    print(f" 🖱️  ACT TRANSFORMER DAGGER ROLLOUTS ({task_name.upper()})")
    print(" Controls:")
    print("   - Push SpaceMouse Joystick : Move Franka End-Effector (X, Y, Z)")
    if act_dim == 4:
        print("   - Left Button              : CLOSE Gripper")
        print("   - Right Button             : OPEN Gripper")
    print("   - Both Buttons             : Abort Episode")
    print("="*65)

    eval_env = make_env(env_id, max_steps=max_steps, render_mode="human")
    device = next(model.parameters()).device
    model.eval()

    new_trajectories = []

    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        sm_source.reset_state()
        
        ensemble = TemporalEnsemble(act_dim=act_dim, max_steps=max_steps, chunk_size=CHUNK_SIZE)

        done = False
        step = 0
        
        ep_obs = [obs]
        ep_acts = []
        interventions_count = 0

        print(f"\n---> ACT DAgger Episode {ep + 1}/{num_episodes} Starting...")
        last_tick = time.perf_counter()

        while not done:
            if sm_source.should_abort():
                print("Aborting episode per user request...")
                break

            current_tick = time.perf_counter()
            actual_dt = current_tick - last_tick
            last_tick = current_tick

            if act_dim == 4:
                current_g_state = float(obs[-1]) if len(obs) >= 1 else 1.0
                sm_source.sync_gripper_state(current_g_state)

            obs_tensor = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
            with th.no_grad():
                pred_chunk, _, _ = model(obs_tensor, None)
                pred_chunk = pred_chunk.squeeze(0).cpu().numpy()

            ensemble.update(step, pred_chunk)
            bot_action = ensemble.get_action(step)

            human_action, human_active = sm_source.get_human_action(actual_dt, act_dim)

            if human_active:
                active_action = human_action
                interventions_count += 1
                status = "🔴 SPACEMOUSE TAKEOVER"
            else:
                active_action = bot_action
                status = "🤖 ACT TRANSFORMER"

            active_action = np.clip(active_action, eval_env.action_space.low, eval_env.action_space.high)

            next_obs, reward, terminated, truncated, info = eval_env.step(active_action)
            done = bool(terminated) or bool(truncated)
            step += 1

            ep_obs.append(next_obs)
            ep_acts.append(active_action)

            obs = next_obs

            if step % 30 == 0 or done:
                dist_ee_block = info.get("dist_ee_block", 0.0)
                print(f"Step {step:03d} | Status: {status:22s} | Dist EE->Block: {dist_ee_block*100:.2f} cm | Act XYZ: {active_action[:3].round(3)}")

            time.sleep(0.001)

        print(f"Episode {ep + 1} ended after {step} steps. Interventions: {interventions_count}/{step} frames.")

        if len(ep_acts) > 5 and not sm_source.should_abort():
            new_trajectories.append({
                "obs": np.array(ep_obs, dtype=np.float32),
                "acts": np.array(ep_acts, dtype=np.float32),
                "terminal": True
            })

    eval_env.close()
    return filter_trajectories(new_trajectories)


# =====================================================================
# 9. MAIN PIPELINE
# =====================================================================

def run_act_dagger_pipeline(
    task_name="pick_and_place",
    data_path=None,
    aggregated_save_path="operator_data_dagger_act_aggregated.pkl",
    model_save_path="act_transformer_model.pt",
    dagger_rounds=3,
    intervene_episodes_per_round=3,
    max_epochs_per_round=30
):
    cfg = TASK_CONFIG[task_name]
    env_id = cfg["env_id"]
    act_dim = cfg["act_dim"]

    if data_path is None:
        data_path = cfg["default_data_path"]

    dummy_raw = gym.make(env_id)
    dummy_flat = FlattenGoalEnv(dummy_raw)
    dummy_grip = SmartBinaryGripperActionWrapper(dummy_flat, close_thresh=0.2, open_thresh=0.6)
    rel_transformer = RelativeGoalWrapper(dummy_grip, ee_z_offset=EE_Z_OFFSET)
    dummy_raw.close()

    if os.path.exists(aggregated_save_path):
        with open(aggregated_save_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded existing aggregated dataset ({len(all_raw_trajectories)} trajectories).")
    elif os.path.exists(data_path):
        with open(data_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded base dataset '{data_path}' ({len(all_raw_trajectories)} trajectories).")
    else:
        fallback_path = "operator_data_pick_and_place_kb.pkl" if act_dim == 4 else "operator_data_push_kb.pkl"
        with open(fallback_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded fallback dataset '{fallback_path}' ({len(all_raw_trajectories)} trajectories).")

    sample_obs = rel_transformer.observation(all_raw_trajectories[0]["obs"][0][:25])
    obs_dim = sample_obs.shape[0]

    act_model = ACTPolicy(
        obs_dim=obs_dim, 
        act_dim=act_dim, 
        chunk_size=CHUNK_SIZE, 
        hidden_dim=HIDDEN_DIM, 
        latent_dim=LATENT_DIM
    )

    sm_source = SpaceMouseTakeoverSource()

    try:
        for dagger_round in range(1, dagger_rounds + 1):
            print(f"\n" + "🚀"*30)
            print(f"   STARTING ACT TRANSFORMER DAGGER ROUND {dagger_round}/{dagger_rounds}")
            print(f"   Current Dataset Size: {len(all_raw_trajectories)} Trajectories")
            print("🚀"*30)

            obs_tensor, chunks_tensor = format_dataset_for_act(
                all_raw_trajectories, 
                rel_transformer, 
                act_dim=act_dim, 
                chunk_size=CHUNK_SIZE
            )

            num_samples = len(obs_tensor)
            val_size = max(1, int(num_samples * 0.2))
            indices = th.randperm(num_samples)

            val_idx = indices[:val_size]
            train_idx = indices[val_size:]

            train_obs, train_chunks = obs_tensor[train_idx], chunks_tensor[train_idx]
            val_obs, val_chunks = obs_tensor[val_idx], chunks_tensor[val_idx]

            act_model = train_act_model(
                model=act_model,
                train_obs=train_obs,
                train_chunks=train_chunks,
                val_obs=val_obs,
                val_chunks=val_chunks,
                max_epochs=max_epochs_per_round,
                patience=5,
                lr=1e-4,
                batch_size=64
            )

            th.save(act_model.state_dict(), model_save_path)
            print(f"Saved updated ACT model to '{model_save_path}'.")

            new_trajs = collect_spacemouse_act_interventions(
                model=act_model,
                sm_source=sm_source,
                task_name=task_name,
                num_episodes=intervene_episodes_per_round,
                max_steps=1000
            )

            all_raw_trajectories.extend(new_trajs)

            with open(aggregated_save_path, "wb") as f:
                pickle.dump(all_raw_trajectories, f)
            print(f"💾 Aggregated dataset saved to '{aggregated_save_path}' ({len(all_raw_trajectories)} trajectories).")

    finally:
        sm_source.close()

    print("\n" + "🎉"*30)
    print(f" ACT Transformer Pipeline Complete! Saved to '{model_save_path}'.")
    print("🎉"*30)


if __name__ == "__main__":
    run_act_dagger_pipeline(
        task_name="pick_and_place",
        data_path="operator_data_pick_and_place_spacemouse.pkl",
        aggregated_save_path="operator_data_dagger_act_aggregated.pkl",
        model_save_path="act_transformer_model.pt",
        dagger_rounds=3,
        intervene_episodes_per_round=3,
        max_epochs_per_round=30
    )