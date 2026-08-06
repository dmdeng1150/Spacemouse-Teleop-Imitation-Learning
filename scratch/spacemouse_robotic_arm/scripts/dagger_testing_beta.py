import os
import time
import pickle
import random
import threading
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import panda_mujoco_gym
import pyspacemouse
from scipy.signal import savgol_filter

from imitation.data import types
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.policies import ActorCriticPolicy

# Import wrappers from smooth_env.py matching operator setup
from smooth_env import (
    FlattenGoalEnv, 
    BinaryGripperActionWrapper,
    FixDoneWrapper
)

# ===== TUNABLE PARAMETERS & TASK CONFIG =====
CONTROL_HZ = 100
CONTROL_DT = 1.0 / CONTROL_HZ
EE_Z_OFFSET = 0.068  # Exact 6.8cm offset from hand frame to fingertip center
FRAME_STACK = 4      # Stack last 4 observations for velocity memory
SENSITIVITY = 8.0  
POLICY_ACTION_GAIN = 1.0  # Restored to 1.0x (Natural trained speed & deceleration)

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
# 1. RELATIVE GOAL WRAPPER
# =====================================================================

class RelativeGoalWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_clearance=0.05, transition_dist=0.08, ee_z_offset=0.068):
        super().__init__(env)
        self.max_clearance = max_clearance        
        self.transition_dist = transition_dist    
        self.ee_z_offset = ee_z_offset  
        
        if hasattr(env.observation_space, "spaces"):
            self.base_obs_dim = sum(space.shape[0] for space in env.observation_space.spaces.values())
        else:
            self.base_obs_dim = env.observation_space.shape[0]

        new_shape = self.base_obs_dim + 19
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )

        self.last_dist_ee_block = 0.0
        self.last_dist_block_goal = 0.0
        self.last_dist_xy = 0.0
        self.last_dist_z = 0.0
        self._debug_printed = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        info["dist_ee_block"] = float(self.last_dist_ee_block)
        info["dist_block_goal"] = float(self.last_dist_block_goal)
        info["dist_xy"] = float(self.last_dist_xy)
        info["dist_z"] = float(self.last_dist_z)
        return obs, reward, terminated, truncated, info

    def observation(self, obs, override_gripper=None):
        if isinstance(obs, dict):
            raw_ee_pos = obs["observation"][:3].copy()
            block_pos = obs["achieved_goal"][:3].copy()
            goal_pos = obs["desired_goal"][:3].copy()
            flat_base_obs = np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]]).astype(np.float32)
        else:
            raw_base = obs[:self.base_obs_dim] if obs.shape[0] >= self.base_obs_dim else obs
            flat_base_obs = raw_base.astype(np.float32)
            raw_ee_pos = flat_base_obs[0:3].copy()
            
            if hasattr(self.env.unwrapped, 'get_block_position'):
                block_pos = np.array(self.env.unwrapped.get_block_position(), dtype=np.float32)
                goal_pos = np.array(self.env.unwrapped.get_target_position(), dtype=np.float32)
            else:
                block_pos = flat_base_obs[-6:-3].copy()
                goal_pos = flat_base_obs[-3:].copy()

        grasp_pos = raw_ee_pos.copy()
        grasp_pos[2] -= self.ee_z_offset

        dy_rel = block_pos[1] - grasp_pos[1]
        dx_rel = block_pos[0] - grasp_pos[0]
        dz_rel = block_pos[2] - grasp_pos[2]

        rel_xy = np.array([dy_rel, dx_rel], dtype=np.float32)
        dist_xy_val = np.linalg.norm(rel_xy)
        dir_xy = rel_xy / (dist_xy_val + 1e-6)
        dist_xy = np.array([dist_xy_val], dtype=np.float32)

        clearance_ratio = min(1.0, dist_xy_val / self.transition_dist)
        funnel_target_z = block_pos[2] + self.max_clearance * clearance_ratio
        
        rel_ee_to_funnel = np.array([dy_rel, dx_rel, funnel_target_z - grasp_pos[2]], dtype=np.float32)
        rel_ee_to_block = np.array([dy_rel, dx_rel, dz_rel], dtype=np.float32)

        dy_goal = goal_pos[1] - block_pos[1]
        dx_goal = goal_pos[0] - block_pos[0]
        dz_goal = goal_pos[2] - block_pos[2]
        rel_block_to_goal = np.array([dy_goal, dx_goal, dz_goal], dtype=np.float32)

        smooth_alignment = np.array([np.exp(-dist_xy_val / 0.03)], dtype=np.float32)

        dist_block_goal_val = np.linalg.norm(rel_block_to_goal)
        dist_block_goal = np.array([dist_block_goal_val], dtype=np.float32)
        block_height = np.array([block_pos[2]], dtype=np.float32)

        if override_gripper is not None:
            g_val = float(override_gripper)
        else:
            g_val = 1.0
            curr = self.env
            while curr is not None:
                if hasattr(curr, 'is_grasped') and curr.is_grasped:
                    g_val = -1.0
                    break
                if hasattr(curr, 'state'):
                    g_val = float(curr.state)
                    break
                if hasattr(curr, 'env'):
                    curr = curr.env
                else:
                    break

        gripper_state = np.array([g_val], dtype=np.float32)

        # Dynamic Active Target Signal
        if g_val < 0.0:
            active_target = rel_block_to_goal
        else:
            active_target = rel_ee_to_funnel

        self.last_dist_xy = dist_xy_val
        self.last_dist_z = abs(dz_rel)
        self.last_dist_ee_block = np.linalg.norm(rel_ee_to_block)
        self.last_dist_block_goal = dist_block_goal_val

        if not self._debug_printed:
            print(f"🔍 [WRAPPER DEBUG] Offset={self.ee_z_offset:.3f}m | EE_Z={raw_ee_pos[2]:.3f} | Grasp_Z={grasp_pos[2]:.3f} | Block_Z={block_pos[2]:.3f} | Calculated Dist={self.last_dist_ee_block*100:.2f}cm")
            self._debug_printed = True

        return np.concatenate([
            flat_base_obs, 
            rel_ee_to_funnel,
            rel_ee_to_block, 
            rel_block_to_goal, 
            active_target,
            dir_xy,
            dist_xy,
            smooth_alignment,
            dist_block_goal, 
            block_height,
            gripper_state
        ]).astype(np.float32)


# =====================================================================
# 2. AUTO-GRASP WRAPPER (EXPANDED 5.0cm CAPTURE RADIUS)
# =====================================================================

class AutoGraspWrapper(gym.Wrapper):
    """Calculates EE-to-block distance directly from obs and latches gripper closed when <= 5.0cm."""
    def __init__(self, env, grasp_dist_thresh=0.050, lift_boost=0.8, ee_z_offset=0.068, clamp_steps=5): # Expanded to 5.0cm
        super().__init__(env)
        self.grasp_dist_thresh = grasp_dist_thresh
        self.lift_boost = lift_boost
        self.ee_z_offset = ee_z_offset
        self.clamp_steps = clamp_steps
        self.is_grasped = False
        self.grasp_counter = 0

    def reset(self, **kwargs):
        self.is_grasped = False
        self.grasp_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        act = np.array(action, copy=True)

        if len(act) >= 4:
            # Human operator manually pressed OPEN during SpaceMouse takeover
            if act[3] > 0.8:
                self.is_grasped = False
                self.grasp_counter = 0

            if self.is_grasped:
                act[3] = -1.0  # Force gripper closed
                self.grasp_counter += 1

                # 1. CLAMPING PAUSE: Hold XYZ still for 5 frames so fingers physically enclose block
                if self.grasp_counter <= self.clamp_steps:
                    act[:3] = 0.0
                else:
                    # 2. LIFT PHASE: Apply strong upward Z boost after clamping
                    act[2] = max(act[2], self.lift_boost)

        obs, reward, terminated, truncated, info = self.env.step(act)

        # Direct distance calculation from obs
        if isinstance(obs, dict):
            ee_pos = obs["observation"][:3].copy()
            block_pos = obs["achieved_goal"][:3].copy()
        else:
            raw_obs = obs[:25] if obs.shape[0] >= 25 else obs
            ee_pos = raw_obs[0:3].copy()
            
            if hasattr(self.env.unwrapped, 'get_block_position'):
                block_pos = np.array(self.env.unwrapped.get_block_position(), dtype=np.float32)
            else:
                block_pos = raw_obs[-6:-3].copy()

        grasp_pos = ee_pos.copy()
        grasp_pos[2] -= self.ee_z_offset
        
        dy_rel = block_pos[1] - grasp_pos[1]
        dx_rel = block_pos[0] - grasp_pos[0]
        dz_rel = block_pos[2] - grasp_pos[2]

        dist_xy = float(np.linalg.norm([dy_rel, dx_rel]))
        dist_z = float(abs(dz_rel))
        dist_ee_block = float(np.linalg.norm([dy_rel, dx_rel, dz_rel]))

        info["dist_xy"] = dist_xy
        info["dist_z"] = dist_z
        info["dist_ee_block"] = dist_ee_block

        # --- UNBLOCKED PROXIMITY TRIGGER ---
        # Trigger grasp as soon as EE gets within 5.0cm of the block!
        if dist_ee_block < self.grasp_dist_thresh:
            if not self.is_grasped:
                print(f"🔒 [AUTO-GRASP TRIGGERED!] Dist 3D: {dist_ee_block*100:.2f}cm <= {self.grasp_dist_thresh*100:.2f}cm")
            self.is_grasped = True

        return obs, reward, terminated, truncated, info


# =====================================================================
# 3. OBS HISTORY WRAPPER
# =====================================================================

class ObsHistoryWrapper(gym.Wrapper):
    """Stacks last `n_stack` observations into a single vector for velocity memory."""
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        orig_shape = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_shape * n_stack,), dtype=np.float32
        )
        self.history = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history = np.tile(obs, self.n_stack)
        return self.history.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        orig_dim = obs.shape[0]
        self.history = np.roll(self.history, -orig_dim)
        self.history[-orig_dim:] = obs
        return self.history.astype(np.float32), reward, terminated, truncated, info


# =====================================================================
# 4. CUSTOM POLICY WITH DROPOUT
# =====================================================================

class DropoutActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, dropout_rate=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        obs_dim = self.observation_space.shape[0]
        self.mlp_extractor.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )


# =====================================================================
# 5. SPACEMOUSE THREAD & TAKEOVER SOURCE
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
# 6. ENVIRONMENT FACTORY
# =====================================================================

def make_env(env_id="FrankaPickAndPlaceSparse-v0", max_steps=1000, render_mode=None, n_stack=FRAME_STACK):
    raw_env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_steps)
    flat_env = FlattenGoalEnv(raw_env)
    grip_env = BinaryGripperActionWrapper(flat_env, close_thresh=0.5, open_thresh=0.5)
    
    auto_env = AutoGraspWrapper(grip_env, grasp_dist_thresh=0.050, lift_boost=0.8, ee_z_offset=EE_Z_OFFSET, clamp_steps=5)
    
    rel_env = RelativeGoalWrapper(auto_env, ee_z_offset=EE_Z_OFFSET)
    fixed_env = FixDoneWrapper(rel_env)
    stacked_env = ObsHistoryWrapper(fixed_env, n_stack=n_stack)
    return stacked_env


# =====================================================================
# 7. DATASET FILTERING & STACKED FORMATTING
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


def format_dataset_for_bc(raw_trajectories, rel_transformer, act_dim=4, n_stack=FRAME_STACK, obs_noise_std=0.001):
    formatted = []

    for traj in raw_trajectories:
        raw_obs = traj["obs"]
        raw_acts = traj["acts"]

        if len(raw_acts) < 10:
            continue

        close_idx = None
        if act_dim == 4:
            for idx, act in enumerate(raw_acts):
                if len(act) >= 4 and act[3] < 0.2:
                    close_idx = idx
                    break

        single_obs_list = []
        acts_list = []

        for i in range(len(raw_acts)):
            base_obs = raw_obs[i][:25].copy()

            if obs_noise_std > 0:
                base_obs[:3] += np.random.normal(0, obs_noise_std, size=3)

            g_state = -1.0 if (close_idx is not None and i >= close_idx) else 1.0

            transformed_obs = rel_transformer.observation(base_obs, override_gripper=g_state)

            target_act = np.array(raw_acts[i], copy=True)

            if act_dim == 4:
                if close_idx is not None and i >= close_idx:
                    target_act[3] = -1.0
                    
                    base_dim = rel_transformer.base_obs_dim
                    rel_goal = transformed_obs[base_dim + 6 : base_dim + 9]
                    dist_to_goal = np.linalg.norm(rel_goal)

                    if np.linalg.norm(target_act[:3]) < 0.001 and dist_to_goal > 0.01:
                        goal_dir = rel_goal / (dist_to_goal + 1e-6)
                        target_act[:3] = goal_dir * 0.05
                else:
                    target_act[3] = 1.0

            single_obs_list.append(transformed_obs)
            acts_list.append(target_act)

        final_base_obs = raw_obs[-1][:25].copy()
        final_obs = rel_transformer.observation(final_base_obs, override_gripper=g_state)
        single_obs_list.append(final_obs)

        # Build Frame History Stack (44 x 4 = 176 features)
        stacked_obs_list = []
        for i in range(len(single_obs_list)):
            stack = []
            for k in range(n_stack - 1, -1, -1):
                past_idx = max(0, i - k)
                stack.append(single_obs_list[past_idx])
            stacked_obs_list.append(np.concatenate(stack, axis=0))

        dummy_rews = np.zeros(len(acts_list), dtype=np.float32)
        formatted.append(
            types.TrajectoryWithRew(
                obs=np.array(stacked_obs_list, dtype=np.float32),
                acts=np.array(acts_list, dtype=np.float32),
                infos=None,
                terminal=traj["terminal"],
                rews=dummy_rews
            )
        )
    return formatted


# =====================================================================
# 8. EARLY-STOPPING TRAIN LOOP
# =====================================================================

def train_bc_with_early_stopping(
    policy,
    formatted_dataset,
    max_epochs=25,
    patience=5,
    l2_weight=3e-2,
    lr=1e-4,
    val_ratio=0.2,
    batch_size=64,
    temp_save_path="temp_best_model.pt"
):
    shuffled_dataset = formatted_dataset.copy()
    random.shuffle(shuffled_dataset)
    
    val_size = max(1, int(len(shuffled_dataset) * val_ratio))
    val_demos = shuffled_dataset[:val_size]
    train_demos = shuffled_dataset[val_size:]

    def flatten_to_tensors(demos):
        obs_list, act_list = [], []
        for traj in demos:
            obs_list.append(traj.obs[:-1])
            act_list.append(traj.acts)
        obs_tensor = th.tensor(np.concatenate(obs_list, axis=0), dtype=th.float32)
        act_tensor = th.tensor(np.concatenate(act_list, axis=0), dtype=th.float32)
        return obs_tensor, act_tensor

    train_obs, train_acts = flatten_to_tensors(train_demos)
    val_obs, val_acts = flatten_to_tensors(val_demos)

    device = policy.device
    train_obs, train_acts = train_obs.to(device), train_acts.to(device)
    val_obs, val_acts = val_obs.to(device), val_acts.to(device)

    optimizer = th.optim.AdamW(policy.parameters(), lr=lr, weight_decay=l2_weight)

    num_samples = len(train_obs)
    best_val_mse = float("inf")
    patience_counter = 0

    print(f"\n--- BC Training: {num_samples} train steps | Obs Dim: {train_obs.shape[1]} ---")

    for epoch in range(1, max_epochs + 1):
        policy.train()
        indices = th.randperm(num_samples)
        train_losses = []

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            b_obs, b_acts = train_obs[batch_idx], train_acts[batch_idx]

            dist = policy.get_distribution(b_obs)
            loss = -dist.log_prob(b_acts).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_loss = np.mean(train_losses)

        policy.eval()
        with th.no_grad():
            pred_val_acts = policy.predict(val_obs.cpu().numpy(), deterministic=True)[0]
            val_mse = np.mean((pred_val_acts - val_acts.cpu().numpy()) ** 2)

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train LogLoss: {mean_train_loss:.3f} | Val Action MSE: {val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            policy.save(temp_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"✋ Early stopping triggered at Epoch {epoch}! Best Val MSE: {best_val_mse:.6f}")
                break

    policy = policy.load(temp_save_path, device=device)
    if os.path.exists(temp_save_path):
        os.remove(temp_save_path)
    return policy


# =====================================================================
# 9. SPACEMOUSE DAGGER ROLLOUTS
# =====================================================================

def collect_spacemouse_dagger_interventions(policy, sm_source, task_name="pick_and_place", num_episodes=3, max_steps=1000):
    cfg = TASK_CONFIG[task_name]
    env_id = cfg["env_id"]
    act_dim = cfg["act_dim"]

    print("\n" + "="*65)
    print(f" 🖱️  SPACEMOUSE DAGGER INTERACTIVE ROLLOUTS ({task_name.upper()})")
    print(" Controls:")
    print("   - Push SpaceMouse Joystick : Move Franka End-Effector (X, Y, Z)")
    if act_dim == 4:
        print("   - Left Button              : CLOSE Gripper")
        print("   - Right Button             : OPEN Gripper")
    print("   - Both Buttons             : Abort Episode")
    print("="*65)

    eval_env = make_env(env_id, max_steps=max_steps, render_mode="human")
    new_trajectories = []

    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        sm_source.reset_state()
        done = False
        step = 0
        
        ep_obs = [obs]
        ep_acts = []
        interventions_count = 0

        print(f"\n---> SpaceMouse DAgger Episode {ep + 1}/{num_episodes} Starting...")
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

            bot_action, _ = policy.predict(obs, deterministic=True)
            bot_action = np.array(bot_action, copy=True).flatten()

            human_action, human_active = sm_source.get_human_action(actual_dt, act_dim)

            if human_active:
                active_action = human_action
                interventions_count += 1
                status = "🔴 SPACEMOUSE TAKEOVER"
            else:
                active_action = bot_action.copy()
                active_action[:3] *= POLICY_ACTION_GAIN
                status = "🤖 AUTONOMOUS"

            active_action = np.clip(active_action, eval_env.action_space.low, eval_env.action_space.high)

            next_obs, reward, terminated, truncated, info = eval_env.step(active_action)
            done = bool(terminated) or bool(truncated)
            step += 1

            ep_obs.append(next_obs)
            ep_acts.append(active_action)

            obs = next_obs

            if step % 30 == 0 or done:
                dist_3d = info.get("dist_ee_block", 0.0)
                dist_xy = info.get("dist_xy", 0.0)
                dist_z = info.get("dist_z", 0.0)
                print(f"Step {step:03d} | Status: {status:22s} | 3D: {dist_3d*100:.2f}cm (XY: {dist_xy*100:.2f}cm | Z: {dist_z*100:.2f}cm) | Bot Act XYZ: {active_action[:3].round(3)}")

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
# 10. MAIN PIPELINE
# =====================================================================

def run_spacemouse_dagger_pipeline(
    task_name="pick_and_place",
    data_path=None,
    aggregated_save_path="operator_data_dagger_spacemouse_aggregated.pkl",
    model_save_path="bc_dagger_spacemouse_model.pt",
    dagger_rounds=3,
    intervene_episodes_per_round=3,
    max_epochs_per_round=25
):
    cfg = TASK_CONFIG[task_name]
    env_id = cfg["env_id"]
    act_dim = cfg["act_dim"]

    if data_path is None:
        data_path = cfg["default_data_path"]

    dummy_raw = gym.make(env_id)
    dummy_flat = FlattenGoalEnv(dummy_raw)
    dummy_grip = BinaryGripperActionWrapper(dummy_flat, close_thresh=0.5, open_thresh=0.5)
    rel_transformer = RelativeGoalWrapper(dummy_grip, ee_z_offset=EE_Z_OFFSET)

    if os.path.exists(aggregated_save_path):
        with open(aggregated_save_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded existing aggregated DAgger dataset ({len(all_raw_trajectories)} trajectories).")
    elif os.path.exists(data_path):
        with open(data_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded base SpaceMouse dataset '{data_path}' ({len(all_raw_trajectories)} trajectories).")
    else:
        fallback_path = "operator_data_pick_and_place_kb.pkl" if act_dim == 4 else "operator_data_push_kb.pkl"
        with open(fallback_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded fallback dataset '{fallback_path}' ({len(all_raw_trajectories)} trajectories).")

    dummy_raw.close()

    train_env = RolloutInfoWrapper(make_env(env_id, max_steps=1000, n_stack=FRAME_STACK))

    policy = DropoutActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=lambda _: 1e-4,
        net_arch=[128, 128],
        log_std_init=-0.5,
        dropout_rate=0.2
    )

    sm_source = SpaceMouseTakeoverSource()

    try:
        for dagger_round in range(1, dagger_rounds + 1):
            print(f"\n" + "🚀"*30)
            print(f"   STARTING SPACEMOUSE DAGGER ROUND {dagger_round}/{dagger_rounds}")
            print(f"   Current Dataset Size: {len(all_raw_trajectories)} Trajectories")
            print("🚀"*30)

            # Format dataset with 4-frame history stacking (176 features)
            formatted_dataset = format_dataset_for_bc(
                all_raw_trajectories, 
                rel_transformer, 
                act_dim=act_dim, 
                n_stack=FRAME_STACK, 
                obs_noise_std=0.003
            )

            policy = train_bc_with_early_stopping(
                policy=policy,
                formatted_dataset=formatted_dataset,
                max_epochs=max_epochs_per_round,
                patience=5,
                l2_weight=3e-2,
                lr=1e-4
            )
            
            policy.save(model_save_path)
            print(f"Saved updated policy to '{model_save_path}'.")

            # Collect SpaceMouse Interventions
            new_trajs = collect_spacemouse_dagger_interventions(
                policy=policy,
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
        train_env.close()

    print("\n" + "🎉"*30)
    print(f" SpaceMouse DAgger Complete! Saved to '{model_save_path}'.")
    print("🎉"*30)


if __name__ == "__main__":
    run_spacemouse_dagger_pipeline(
        task_name="pick_and_place",
        data_path="operator_data_pick_and_place_spacemouse.pkl",
        aggregated_save_path="operator_data_dagger_spacemouse_aggregated.pkl",
        model_save_path="bc_dagger_spacemouse_model.pt",
        dagger_rounds=3,
        intervene_episodes_per_round=3,
        max_epochs_per_round=25
    )