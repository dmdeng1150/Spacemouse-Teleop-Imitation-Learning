import os
import time
import pickle
import numpy as np
import gymnasium as gym
import panda_mujoco_gym
from pynput import keyboard

from imitation.data import types
from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.policies import ActorCriticPolicy

# Import all wrappers from smooth_env.py
from smooth_env import (
    FlattenGoalEnv, 
    SmoothFrankaWrapper, 
    SmoothXYZActionWrapper,
    BinaryGripperActionWrapper,
    RelativeGoalWrapper, 
    FixDoneWrapper
)


# =====================================================================
# 1. MUJOCO KEYBOARD CALLBACK DISABLE UTILITY
# =====================================================================

def disable_mujoco_key_callbacks(env):
    """Disables default MuJoCo viewer key bindings (like Spacebar pause) 
       so keys are reserved exclusively for keyboard teleoperation.
    """
    try:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "mujoco_renderer"):
            viewer = unwrapped.mujoco_renderer._get_viewer("human")
            if hasattr(viewer, "window") and viewer.window is not None:
                import glfw
                glfw.set_key_callback(viewer.window, lambda window, key, scancode, action, mods: None)
                print("[INFO] Native MuJoCo GUI key callbacks disabled for keyboard control.")
    except Exception as e:
        print(f"[WARNING] Could not disable MuJoCo key callbacks: {e}")


# =====================================================================
# 2. KEYBOARD TAKEOVER SOURCE (pynput)
# =====================================================================

class KeyboardTakeoverSource:
    """Listens for keyboard inputs asynchronously and detects human takeover interventions."""
    def __init__(self, move_speed=0.4, ramp_rate=0.15):
        self.move_speed = move_speed
        self.ramp_rate = ramp_rate

        self.vel = np.zeros(3, dtype=np.float32)
        self.is_closed = False  # False = OPEN (+1.0), True = CLOSED (-1.0)
        self.manual_abort = False

        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        k_str = None
        if hasattr(key, 'char') and key.char is not None:
            k_str = key.char.lower()
        elif key == keyboard.Key.space:
            k_str = 'space'

        if k_str:
            self.pressed_keys.add(k_str)

            if k_str == 'r':
                self.manual_abort = True
                print("\n[INFO] Episode aborted by operator ('R' pressed).")
            elif k_str == 'space':
                self.is_closed = not self.is_closed
                state_str = "CLOSED" if self.is_closed else "OPEN"
                print(f"   [KEYBOARD] Gripper toggled -> {state_str}")

    def _on_release(self, key):
        k_str = None
        if hasattr(key, 'char') and key.char is not None:
            k_str = key.char.lower()
        elif key == keyboard.Key.space:
            k_str = 'space'

        if k_str:
            self.pressed_keys.discard(k_str)

    def get_human_action(self):
        target_vel = np.zeros(3, dtype=np.float32)
        if 'a' in self.pressed_keys: target_vel[0] -= self.move_speed  # Move Left (-X)
        if 'd' in self.pressed_keys: target_vel[0] += self.move_speed  # Move Right (+X)
        if 'w' in self.pressed_keys: target_vel[1] += self.move_speed  # Move Forward (+Y)
        if 's' in self.pressed_keys: target_vel[1] -= self.move_speed  # Move Backward (-Y)
        if 'e' in self.pressed_keys: target_vel[2] += self.move_speed  # Move Up (+Z)
        if 'q' in self.pressed_keys: target_vel[2] -= self.move_speed  # Move Down (-Z)

        # Smooth velocity ramping
        self.vel = (1.0 - self.ramp_rate) * self.vel + self.ramp_rate * target_vel
        if np.all(target_vel == 0) and np.linalg.norm(self.vel) < 0.01:
            self.vel[:] = 0.0

        # Human is active if any directional key (W, A, S, D, Q, E) is currently pressed
        movement_keys = {'w', 's', 'a', 'd', 'q', 'e'}
        human_active = any(k in self.pressed_keys for k in movement_keys)

        # Gripper state: -1.0 for CLOSED, 1.0 for OPEN
        gripper_val = -1.0 if self.is_closed else 1.0
        action = np.append(self.vel, gripper_val).astype(np.float32)

        return action, human_active

    def should_abort(self):
        return self.manual_abort

    def reset_state(self):
        self.vel = np.zeros(3, dtype=np.float32)
        self.manual_abort = False

    def close(self):
        self.listener.stop()


# =====================================================================
# 3. ENVIRONMENT FACTORY
# =====================================================================

def make_env(env_id="FrankaPickAndPlaceSparse-v0", max_steps=250, render_mode=None):
    raw_env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_steps)
    flat_env = FlattenGoalEnv(raw_env)
    smooth_franka = SmoothFrankaWrapper(flat_env)
    smooth_xyz = SmoothXYZActionWrapper(smooth_franka, alpha=0.65)
    grip_env = BinaryGripperActionWrapper(smooth_xyz, close_thresh=0.2, open_thresh=0.6)
    rel_env = RelativeGoalWrapper(grip_env, ee_z_offset=0.058)
    fixed_env = FixDoneWrapper(rel_env)
    return fixed_env


# =====================================================================
# 4. DATASET UTILITIES
# =====================================================================

def format_dataset_for_bc(raw_trajectories, rel_transformer):
    """Converts raw trajectory dictionaries into imitation TrajectoryWithRew objects."""
    formatted = []
    for traj in raw_trajectories:
        if len(traj["acts"]) < 5:
            continue
        dummy_rews = np.zeros(len(traj["acts"]), dtype=np.float32)

        transformed_obs = []
        for i, o in enumerate(traj["obs"]):
            g_act = traj["acts"][min(i, len(traj["acts"]) - 1)][3] if len(traj["acts"][0]) >= 4 else 1.0
            g_state = -1.0 if g_act < 0.2 else 1.0
            transformed_obs.append(rel_transformer.observation(o, override_gripper=g_state))

        transformed_obs = np.array(transformed_obs, dtype=np.float32)

        formatted.append(
            types.TrajectoryWithRew(
                obs=transformed_obs,
                acts=np.array(traj["acts"], dtype=np.float32),
                infos=None,
                terminal=traj["terminal"],
                rews=dummy_rews
            )
        )
    return formatted


def collect_keyboard_dagger_interventions(policy, kb_source, env_id, num_episodes=3, max_steps=250):
    """Runs autonomous policy predictions. Whenever movement keys (W,A,S,D,Q,E) are pressed,
       human keyboard control overrides the policy and logs the corrected trajectory.
    """
    print("\n" + "="*65)
    print(" ⌨️  KEYBOARD DAGGER INTERACTIVE ROLLOUTS")
    print(" Controls:")
    print("   [W / S] : Move Forward / Backward (+Y / -Y)")
    print("   [A / D] : Move Left / Right (-X / +X)")
    print("   [E / Q] : Move Up / Down (+Z / -Z)")
    print("   [SPACE] : Toggle Gripper (OPEN / CLOSED)")
    print("   [R]     : Abort Episode")
    print("="*65)

    eval_env = make_env(env_id, max_steps=max_steps, render_mode="human")
    disable_mujoco_key_callbacks(eval_env)

    new_trajectories = []

    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        kb_source.reset_state()
        done = False
        step = 0
        
        ep_obs = [obs]
        ep_acts = []
        interventions_count = 0

        print(f"\n---> Keyboard DAgger Episode {ep + 1}/{num_episodes} Starting...")

        while not done:
            if kb_source.should_abort():
                print("Aborting episode per user request...")
                break

            # 1. Predict autonomous robot policy action
            bot_action, _ = policy.predict(obs, deterministic=True)
            bot_action = np.array(bot_action, copy=True).flatten()

            # 2. Get keyboard human action and active status
            human_action, human_active = kb_source.get_human_action()

            # 3. Determine active control (Keyboard Override vs Autonomous Policy)
            if human_active:
                active_action = human_action
                interventions_count += 1
                status = "🔴 KB TAKEOVER"
            else:
                active_action = bot_action
                status = "🤖 AUTONOMOUS"

            # Step environment forward
            next_obs, reward, terminated, truncated, info = eval_env.step(active_action)
            done = bool(terminated) or bool(truncated)
            step += 1

            ep_obs.append(next_obs)
            ep_acts.append(active_action)

            obs = next_obs

            if step % 30 == 0 or done:
                dist_ee_block = info.get("dist_ee_block", 0.0)
                print(f"Step {step:03d} | Status: {status:18s} | Dist EE->Block: {dist_ee_block*100:.2f} cm")

            time.sleep(0.01)

        print(f"Episode {ep + 1} ended after {step} steps. Human Interventions: {interventions_count}/{step} frames.")

        if len(ep_acts) > 5 and not kb_source.should_abort():
            new_trajectories.append({
                "obs": np.array(ep_obs, dtype=np.float32),
                "acts": np.array(ep_acts, dtype=np.float32),
                "terminal": True
            })

    eval_env.close()
    return new_trajectories


# =====================================================================
# 5. MAIN KEYBOARD DAGGER PIPELINE
# =====================================================================

def run_keyboard_dagger_pipeline(
    data_path="operator_data_pick_and_place_kb.pkl",
    aggregated_save_path="operator_data_dagger_kb_aggregated.pkl",
    model_save_path="bc_dagger_kb_model.pt",
    env_id="FrankaPickAndPlaceSparse-v0",
    dagger_rounds=3,
    intervene_episodes_per_round=3,
    retrain_epochs=30
):
    # 1. Observation Transformer Setup
    dummy_raw = gym.make(env_id)
    dummy_flat = FlattenGoalEnv(dummy_raw)
    dummy_franka = SmoothFrankaWrapper(dummy_flat)
    dummy_xyz = SmoothXYZActionWrapper(dummy_franka, alpha=0.65)
    dummy_grip = BinaryGripperActionWrapper(dummy_xyz)
    rel_transformer = RelativeGoalWrapper(dummy_grip, ee_z_offset=0.058)

    # 2. Load Base Dataset or Previous Aggregated Dataset
    if os.path.exists(aggregated_save_path):
        with open(aggregated_save_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded existing aggregated DAgger dataset ({len(all_raw_trajectories)} trajectories).")
    elif os.path.exists(data_path):
        with open(data_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded base keyboard dataset '{data_path}' ({len(all_raw_trajectories)} trajectories).")
    else:
        # Fallback to SpaceMouse dataset if keyboard dataset does not exist yet
        fallback_path = "operator_data_pick_and_place_spacemouse.pkl"
        with open(fallback_path, "rb") as f:
            all_raw_trajectories = pickle.load(f)
        print(f"📂 Loaded base SpaceMouse dataset '{fallback_path}' ({len(all_raw_trajectories)} trajectories).")

    dummy_raw.close()

    # 3. Training Environment & Policy Setup
    train_env = RolloutInfoWrapper(make_env(env_id, max_steps=250))
    
    policy = ActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=lambda _: 1e-4,
        net_arch=[128, 128],
        log_std_init=-0.5
    )

    kb_source = KeyboardTakeoverSource(move_speed=0.4, ramp_rate=0.15)

    try:
        for dagger_round in range(1, dagger_rounds + 1):
            print(f"\n" + "🚀"*30)
            print(f"   STARTING KEYBOARD DAGGER ROUND {dagger_round}/{dagger_rounds}")
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

            # C. Collect Human Keyboard Interventions
            new_trajs = collect_keyboard_dagger_interventions(
                policy=bc_trainer.policy,
                kb_source=kb_source,
                env_id=env_id,
                num_episodes=intervene_episodes_per_round
            )

            # D. Aggregate Dataset
            all_raw_trajectories.extend(new_trajs)

            with open(aggregated_save_path, "wb") as f:
                pickle.dump(all_raw_trajectories, f)
            print(f"💾 Aggregated dataset saved to '{aggregated_save_path}' (Total: {len(all_raw_trajectories)} trajectories).")

    finally:
        kb_source.close()
        train_env.close()

    print("\n" + "🎉"*30)
    print(f" Keyboard DAgger Complete! Final policy saved to '{model_save_path}'.")
    print("🎉"*30)


if __name__ == "__main__":
    run_keyboard_dagger_pipeline(
        data_path="operator_data_pick_and_place_kb.pkl",
        aggregated_save_path="operator_data_dagger_kb_aggregated.pkl",
        model_save_path="bc_dagger_kb_model.pt",
        env_id="FrankaPickAndPlaceSparse-v0",
        dagger_rounds=3,                      # 3 DAgger iterations
        intervene_episodes_per_round=3,       # 3 keyboard correction episodes per round
        retrain_epochs=30                    # Retrain BC for 30 epochs each round
    )