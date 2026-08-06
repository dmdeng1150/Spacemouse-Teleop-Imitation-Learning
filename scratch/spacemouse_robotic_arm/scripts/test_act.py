import os
import time
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import panda_mujoco_gym

# Import wrappers from smooth_env.py
from smooth_env import (
    FlattenGoalEnv, 
    RelativeGoalWrapper, 
    FixDoneWrapper
)

# ===== ACT MODEL TUNABLE PARAMETERS =====
EE_Z_OFFSET = 0.0     # obs[0:3] is already at fingertip center
CHUNK_SIZE = 30       # Predict 30 future steps at once
LATENT_DIM = 16       # CVAE Latent variable dimension
HIDDEN_DIM = 256      # Matches your saved checkpoint (256)


# =====================================================================
# 1. SMART BINARY GRIPPER WRAPPER (UNBLOCKED PROXIMITY TRIGGER)
# =====================================================================

class SmartBinaryGripperActionWrapper(gym.Wrapper):
    """Action wrapper with threshold shift, 2.0cm grasp latch, and smooth Z lift-off."""
    def __init__(self, env, close_thresh=0.3, open_thresh=0.6, grasp_dist_thresh=0.020, ee_z_offset=0.0, clamp_steps=5, lift_boost=0.20):
        super().__init__(env)
        self.close_thresh = close_thresh
        self.open_thresh = open_thresh
        self.grasp_dist_thresh = grasp_dist_thresh  # Set to your preferred 2.0cm threshold
        self.ee_z_offset = ee_z_offset
        self.clamp_steps = clamp_steps
        self.lift_boost = lift_boost              # Gentle 20% upward speed (Down from 80%)
        self.is_grasped = False
        self.grasp_counter = 0

    def reset(self, **kwargs):
        self.is_grasped = False
        self.grasp_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        act = np.array(action, copy=True)

        if len(act) >= 4:
            # If operator manually presses OPEN (> 0.8), reset latch
            if act[3] > 0.8:
                self.is_grasped = False
                self.grasp_counter = 0

            # If grasped, enforce closed gripper + smooth lift-off
            if self.is_grasped:
                act[3] = -1.0
                self.grasp_counter += 1

                # 1. CLAMPING PAUSE (Frames 1-5): Hold XYZ still so fingers clamp tight
                if self.grasp_counter <= self.clamp_steps:
                    act[:3] = 0.0
                else:
                    # 2. SMOOTH LIFT-OFF RAMP (Frames 6-10): Ramp Z speed smoothly (no jerk)
                    ramp_step = min(5, self.grasp_counter - self.clamp_steps)
                    smooth_lift = (ramp_step / 5.0) * self.lift_boost  # Ramps up smoothly: 0.04 -> 0.08 -> 0.12 -> 0.16 -> 0.20
                    act[2] = max(act[2], smooth_lift)

        # Execute step in simulator
        obs, reward, terminated, truncated, info = self.env.step(act)
        
        # Extract distance EE -> Block
        raw_obs = obs[-25:] if obs.shape[0] > 50 else obs
        ee_pos = raw_obs[0:3].copy()
        
        if hasattr(self.env.unwrapped, 'get_block_position'):
            block_pos = np.array(self.env.unwrapped.get_block_position(), dtype=np.float32)
        else:
            block_pos = raw_obs[-6:-3].copy()

        grasp_pos = ee_pos.copy()
        grasp_pos[2] -= self.ee_z_offset
        dist_ee_block = float(np.linalg.norm(block_pos - grasp_pos))

        info["dist_ee_block"] = dist_ee_block

        # Proximity Latch: Triggers at your preferred 2.0cm threshold
        if dist_ee_block <= self.grasp_dist_thresh:
            if not self.is_grasped:
                print(f"🔒 [TEST GRASP TRIGGERED!] Dist: {dist_ee_block*100:.2f}cm <= {self.grasp_dist_thresh*100:.2f}cm")
            self.is_grasped = True

        return obs, reward, terminated, truncated, info


# =====================================================================
# 2. ACT TRANSFORMER ARCHITECTURE
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
    def __init__(self, obs_dim, act_dim, chunk_size, hidden_dim=256, latent_dim=16):
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
# 3. TEMPORAL ENSEMBLING
# =====================================================================

class TemporalEnsemble:
    """Ensembles XYZ predictions while using horizon lookahead for crisp gripper closure."""
    def __init__(self, act_dim=4, max_steps=1000, chunk_size=CHUNK_SIZE, exp_weight=0.01):
        self.act_dim = act_dim
        self.max_steps = max_steps
        self.chunk_size = chunk_size
        self.exp_weight = exp_weight
        
        buffer_len = max_steps + chunk_size + 100
        self.all_predictions = np.zeros((buffer_len, act_dim), dtype=np.float32)
        self.weights = np.zeros((buffer_len, 1), dtype=np.float32)
        self.horizon_gripper_min = 1.0

    def update(self, t, predicted_chunk):
        if self.act_dim == 4:
            self.horizon_gripper_min = float(np.min(predicted_chunk[:10, 3]))

        for i in range(self.chunk_size):
            step_idx = t + i
            w = np.exp(-self.exp_weight * i)
            self.all_predictions[step_idx] += w * predicted_chunk[i]
            self.weights[step_idx] += w

    def get_action(self, t):
        if self.weights[t] == 0:
            return np.zeros(self.act_dim, dtype=np.float32)
        
        spatial_act = self.all_predictions[t][:3] / self.weights[t]
        
        if self.act_dim == 4:
            gripper_act = -1.0 if self.horizon_gripper_min < 0.3 else 1.0
            return np.append(spatial_act, gripper_act)
        
        return spatial_act


# =====================================================================
# 4. MAIN TEST EVALUATION FUNCTION
# =====================================================================

def test_saved_model(
    model_path="best_act_model.pt", 
    env_id="FrankaPickAndPlaceSparse-v0", 
    num_episodes=5,
    max_steps=400,
    task_num=2
):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return

    print(f"Loading ACT model from '{model_path}'...")

    print("\n--- Launching 3D Simulation Evaluation ---")
    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=max_steps)
    flat_env = FlattenGoalEnv(raw_env) 
    grip_env = SmartBinaryGripperActionWrapper(flat_env, close_thresh=0.3, open_thresh=0.6, grasp_dist_thresh=0.025, ee_z_offset=EE_Z_OFFSET)
    rel_env = RelativeGoalWrapper(grip_env, ee_z_offset=EE_Z_OFFSET)            
    env = FixDoneWrapper(rel_env)                         

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    act_model = ACTPolicy(
        obs_dim=obs_dim, 
        act_dim=act_dim, 
        chunk_size=CHUNK_SIZE, 
        hidden_dim=HIDDEN_DIM, 
        latent_dim=LATENT_DIM
    )
    
    act_model.load_state_dict(th.load(model_path, map_location=device))
    act_model.to(device)
    act_model.eval()

    print("✅ ACT Model loaded successfully!")

    success_count = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        ep_success = False

        ensemble = TemporalEnsemble(act_dim=act_dim, max_steps=max_steps, chunk_size=CHUNK_SIZE)

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

        while not done:
            obs_tensor = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
            with th.no_grad():
                pred_chunk, _, _ = act_model(obs_tensor, None)
                pred_chunk = pred_chunk.squeeze(0).cpu().numpy()

            ensemble.update(step, pred_chunk)
            action = ensemble.get_action(step)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            step += 1

            dist_ee_block = info.get("dist_ee_block", 0.0)
            dist_block_goal = info.get("dist_block_goal", 0.0)

            if info.get("is_success", False) or terminated:
                ep_success = True

            if step % 50 == 0 or done:
                print(f"Step {step:03d} | Dist EE->Block: {dist_ee_block*100:.2f}cm | Dist Block->Goal: {dist_block_goal*100:.2f}cm")

            time.sleep(0.01)

        if ep_success or info.get("is_success", False):
            success_count += 1
            print(f"Result: SUCCESS in {step} steps!")
        else:
            print(f"Result: FAILED (reached max steps).")

    env.close()

    success_rate = (success_count / num_episodes) * 100.0
    print(f"\n==========================================")
    print(f"Final ACT Evaluation Summary:")
    print(f"   Successful Runs: {success_count}/{num_episodes}")
    print(f"   Success Rate:    {success_rate:.1f}%")
    print(f"==========================================")


if __name__ == "__main__":
    test_saved_model(
        model_path="best_act_model.pt", 
        env_id="FrankaPickAndPlaceSparse-v0",
        task_num=2,
        num_episodes=5,
        max_steps=400
    )