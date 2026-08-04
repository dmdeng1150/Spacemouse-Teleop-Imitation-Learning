import os
import time
import pickle
import threading
import numpy as np
import gymnasium as gym
import panda_mujoco_gym
import pyspacemouse

from imitation.data import types
from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.policies import ActorCriticPolicy

# Import all wrappers from smooth_env.py
from smooth_env_beta import (
    FlattenGoalEnv, 
    SmoothFrankaWrapper, 
    SmoothXYZActionWrapper,
    BinaryGripperActionWrapper,
    RelativeGoalWrapper, 
    FixDoneWrapper
)


# =====================================================================
# 1. SPACEMOUSE THREAD & TAKEOVER SOURCE (WITH LIVE STATE SYNC)
# =====================================================================

class SpaceMouseThread:
    """Non-blocking background HID reader for SpaceMouse."""
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
    """Polls SpaceMouse and detects human takeover interventions.
       Includes sync_gripper_state() so human takeover preserves the robot's current grasp!
    """
    GLITCH_FILTER_ALPHA = 0.55
    DEADZONE = 0.05

    def __init__(self):
        print("\nConnecting to SpaceMouse USB HID...")
        devices = pyspacemouse.get_all_hid_devices()
        if not devices:
            print("❌ ERROR: No SpaceMouse detected! Close official 3Dconnexion driver software in Task Manager.")
            
        self._cm = pyspacemouse.open()
        device = self._cm.__enter__()
        self.sm_thread = SpaceMouseThread(device)
        self.filtered_raw = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.gripper_val = 1.0  # 1.0 = OPEN, -1.0 = CLOSED
        self._abort = False
        print("✅ SpaceMouse Connected & Polling!\n")

    def sync_gripper_state(self, current_env_gripper):
        """Synchronizes human gripper memory with live environment state so 
           takeover steering never accidentally drops a held block!
        """
        state = self.sm_thread.get_state()
        if state is not None:
            left, right = state.buttons
            # Only auto-sync if operator is NOT actively pressing left/right buttons
            if not (left or right):
                self.gripper_val = current_env_gripper

    def get_human_action(self):
        state = self.sm_thread.get_state()
        if state is None:
            return np.array([0.0, 0.0, 0.0, self.gripper_val], dtype=np.float32), False

        left, right = state.buttons
        if left and right:
            self._abort = True
            print("\n[INFO] Episode aborted by operator (Both SpaceMouse buttons pressed).")
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

        dx = smooth_deadzone(self.filtered_raw["x"])
        dy = smooth_deadzone(self.filtered_raw["y"])
        dz = smooth_deadzone(self.filtered_raw["z"])

        binary_gripper = -1.0 if self.gripper_val < 0.0 else 1.0

        # Operator is active if translational deflection > 0 or a button was pressed
        human_active = (abs(dx) > 0.0 or abs(dy) > 0.0 or abs(dz) > 0.0 or left or right)

        action = np.array([dx, dy, dz, binary_gripper], dtype=np.float32)
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
# 2. ENVIRONMENT FACTORY
# =====================================================================

def make_env(env_id="FrankaPickAndPlaceSparse-v0", max_steps=250, render_mode=None):
    raw_env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_steps)
    flat_env = FlattenGoalEnv(raw_env)
    smooth_franka = SmoothFrankaWrapper(flat_env)
    smooth_xyz = SmoothXYZActionWrapper(smooth_franka, alpha=0.65)
    grip_env = BinaryGripperActionWrapper(smooth_xyz, close_thresh=0.5, open_thresh=0.8, min_release_z=0.035)
    rel_env = RelativeGoalWrapper(grip_env, ee_z_offset=0.058)
    fixed_env = FixDoneWrapper(rel_env)
    return fixed_env


# =====================================================================
# 3. DATASET UTILITIES
# =====================================================================

def format_dataset_for_bc(raw_trajectories, rel_transformer, oversample_factor=15):
    """Converts trajectory dictionaries into imitation TrajectoryWithRew objects,
       guaranteeing observation shape (43,) and preventing mid-air release targets.
    """
    formatted = []
    for traj in raw_trajectories:
        if len(traj["acts"]) < 5:
            continue

        close_idx = None
        for idx, act in enumerate(traj["acts"]):
            if len(act) >= 4 and act[3] < 0.2:
                close_idx = idx
                break

        obs_list = []
        acts_list = []

        for i, act in enumerate(traj["acts"]):
            curr_raw_obs = traj["obs"][i][:25]
            
            g_act = act[3] if len(act) >= 4 else 1.0
            g_state = -1.0 if g_act < 0.2 else 1.0

            transformed_o = rel_transformer.observation(curr_raw_obs, override_gripper=g_state)

            target_act = np.array(act, copy=True)

            # --- FORCE CLOSE TARGET IN GRASP WINDOW ---
            is_grasp_window = (close_idx is not None and abs(i - close_idx) <= 3)
            if is_grasp_window:
                target_act[3] = -1.0

            # --- MID-AIR TRANSPORT GUARD: Keep action[3] = -1.0 while block is lifted ---
            block_z = curr_raw_obs[14] if len(curr_raw_obs) >= 15 else 0.02
            if i > (close_idx or 0) and block_z > 0.035:
                target_act[3] = -1.0

            obs_list.append(transformed_o)
            acts_list.append(target_act)

            if is_grasp_window:
                for _ in range(oversample_factor):
                    obs_list.append(transformed_o)
                    acts_list.append(target_act)

        final_raw_obs = traj["obs"][-1][:25]
        final_transformed_obs = rel_transformer.observation(final_raw_obs, override_gripper=g_state)
        obs_list.append(final_transformed_obs)

        dummy_rews = np.zeros(len(acts_list), dtype=np.float32)
        formatted.append(
            types.TrajectoryWithRew(
                obs=np.array(obs_list, dtype=np.float32),
                acts=np.array(acts_list, dtype=np.float32),
                infos=None,
                terminal=traj["terminal"],
                rews=dummy_rews
            )
        )
    return formatted


def collect_spacemouse_dagger_interventions(policy, sm_source, env_id, num_episodes=3, max_steps=250):
    print("\n" + "="*65)
    print(" 🖱️  SPACEMOUSE DAGGER INTERACTIVE ROLLOUTS")
    print(" Controls:")
    print("   - Joystick Push / Deflect : Move Franka End-Effector (X, Y, Z)")
    print("   - Left Button            : CLOSE Gripper")
    print("   - Right Button           : OPEN Gripper")
    print("   - Both Buttons           : Abort Episode")
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

        while not done:
            if sm_source.should_abort():
                print("Aborting episode per user request...")
                break

            # FIX: Sync human input memory with environment's live gripper state (feature 40)
            current_g_state = float(obs[40]) if len(obs) >= 41 else 1.0
            sm_source.sync_gripper_state(current_g_state)

            bot_action, _ = policy.predict(obs, deterministic=True)
            bot_action = np.array(bot_action, copy=True).flatten()

            human_action, human_active = sm_source.get_human_action()

            if human_active:
                active_action = human_action
                interventions_count += 1
                status = "🔴 SPACEMOUSE TAKEOVER"
            else:
                active_action = bot_action
                status = "🤖 AUTONOMOUS"

            next_obs, reward, terminated, truncated, info = eval_env.step(active_action)
            done = bool(terminated) or bool(truncated)
            step += 1

            ep_obs.append(next_obs)
            ep_acts.append(active_action)

            obs = next_obs

            if step % 30 == 0 or done:
                dist_ee_block = info.get("dist_ee_block", 0.0)
                print(f"Step {step:03d} | Status: {status:22s} | Dist EE->Block: {dist_ee_block*100:.2f} cm")

            time.sleep(0.01)

        print(f"Episode {ep + 1} ended after {step} steps. Human Interventions: {interventions_count}/{step} frames.")

        if len(ep_acts) > 5 and not sm_source.should_abort():
            new_trajectories.append({
                "obs": np.array(ep_obs, dtype=np.float32),
                "acts": np.array(ep_acts, dtype=np.float32),
                "terminal": True
            })

    eval_env.close()
    return new_trajectories


# =====================================================================
# 4. MAIN SPACEMOUSE DAGGER PIPELINE
# =====================================================================

def run_spacemouse_dagger_pipeline(
    data_path="operator_data_pick_and_place_spacemouse.pkl",
    aggregated_save_path="operator_data_dagger_spacemouse_aggregated.pkl",
    model_save_path="bc_dagger_spacemouse_model.pt",
    env_id="FrankaPickAndPlaceSparse-v0",
    dagger_rounds=3,
    intervene_episodes_per_round=3,
    retrain_epochs=30
):
    dummy_raw = gym.make(env_id)
    dummy_flat = FlattenGoalEnv(dummy_raw)
    dummy_franka = SmoothFrankaWrapper(dummy_flat)
    dummy_xyz = SmoothXYZActionWrapper(dummy_franka, alpha=0.65)
    dummy_grip = BinaryGripperActionWrapper(dummy_xyz)
    rel_transformer = RelativeGoalWrapper(dummy_grip, ee_z_offset=0.058)

    if os.path.exists(aggregated_save_path):
        with open(aggregated_save_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded existing aggregated DAgger dataset ({len(all_raw_trajectories)} trajectories).")
    elif os.path.exists(data_path):
        with open(data_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded base SpaceMouse dataset '{data_path}' ({len(all_raw_trajectories)} trajectories).")
    else:
        fallback_path = "operator_data_pick_and_place_kb.pkl"
        with open(fallback_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded base keyboard dataset '{fallback_path}' ({len(all_raw_trajectories)} trajectories).")

    dummy_raw.close()

    train_env = RolloutInfoWrapper(make_env(env_id, max_steps=250))
    
    policy = ActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=lambda _: 1e-4,
        net_arch=[128, 128],
        log_std_init=-0.5
    )

    sm_source = SpaceMouseTakeoverSource()

    try:
        for dagger_round in range(1, dagger_rounds + 1):
            print(f"\n" + "🚀"*30)
            print(f"   STARTING SPACEMOUSE DAGGER ROUND {dagger_round}/{dagger_rounds}")
            print(f"   Current Total Dataset Size: {len(all_raw_trajectories)} Trajectories")
            print("🚀"*30)

            # A. Prepare formatted dataset
            formatted_dataset = format_dataset_for_bc(all_raw_trajectories, rel_transformer)

            # B. Retrain Policy on Aggregated Dataset
            print(f"\nRetraining BC Policy on {len(formatted_dataset)} aggregated trajectories ({retrain_epochs} epochs)...")
            bc_trainer = bc.BC(
                observation_space=train_env.observation_space,
                action_space=train_env.action_space,
                policy=policy,
                demonstrations=formatted_dataset,
                ent_weight=0.01,
                l2_weight=1e-2,
                rng=np.random.default_rng(seed=42 + dagger_round)
            )
            bc_trainer.train(n_epochs=retrain_epochs, progress_bar=True)
            
            bc_trainer.policy.save(model_save_path)
            print(f"Saved updated policy to '{model_save_path}'.")

            # C. Collect Human SpaceMouse Interventions
            new_trajs = collect_spacemouse_dagger_interventions(
                policy=bc_trainer.policy,
                sm_source=sm_source,
                env_id=env_id,
                num_episodes=intervene_episodes_per_round
            )

            # D. Aggregate Dataset
            all_raw_trajectories.extend(new_trajs)

            with open(aggregated_save_path, "wb") as f:
                pickle.dump(all_raw_trajectories, f)
            print(f"💾 Aggregated dataset saved to '{aggregated_save_path}' (Total: {len(all_raw_trajectories)} trajectories).")

    finally:
        sm_source.close()
        train_env.close()

    print("\n" + "🎉"*30)
    print(f" SpaceMouse DAgger Complete! Final policy saved to '{model_save_path}'.")
    print("🎉"*30)


if __name__ == "__main__":
    run_spacemouse_dagger_pipeline(
        data_path="operator_data_pick_and_place_spacemouse.pkl",
        aggregated_save_path="operator_data_dagger_spacemouse_aggregated.pkl",
        model_save_path="bc_dagger_spacemouse_model.pt",
        env_id="FrankaPickAndPlaceSparse-v0",
        dagger_rounds=3,                      # 3 DAgger iterations
        intervene_episodes_per_round=3,       # 3 SpaceMouse correction episodes per round
        retrain_epochs=30                    # Retrain BC for 30 epochs each round
    )