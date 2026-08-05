# shared file for all input methods used for training. 
import os
import pickle
import time
import numpy as np
from scipy.signal import savgol_filter
import gymnasium as gym
import panda_mujoco_gym

from smooth_env import FlattenGoalEnv, RelativeGoalWrapper, BinaryGripperActionWrapper, FixDoneWrapper

CONTROL_HZ = 100
CONTROL_DT = 1.0 / CONTROL_HZ
EE_Z_OFFSET = 0.058

# ===== TUNABLE PARAMETERS =====
SENSITIVITY = 0.5  # how fast the arm moves (0.5=slow, 1.0=medium, 2.0=fast)

TASK_CONFIG = {
    "push": {
        "env_id": "FrankaPushSparse-v0",
        "act_dim": 3,
    },
    "pick_and_place": {
        "env_id": "FrankaPickAndPlaceSparse-v0",
        "act_dim": 4,
    },
}

def prompt_for_task() -> str:
    raw = input("Select task -- Push (1) or Pick-and-Place (2): ").strip()
    return {"1": "push", "2": "pick_and_place"}.get(raw)

class InputSource:
    def get_action(self, act_dim: int) -> np.ndarray:
        raise NotImplementedError
    
    def should_abort(self) -> bool:
        raise NotImplementedError

    def reset_state(self) -> None:
        raise NotImplementedError

    def on_env_ready(self, env) -> None:
        pass

    def close(self) -> None:
        raise NotImplementedError

def _filter_and_save(existing_dataset, new_dataset, output_file):
    print("\n--- Applying dataset filters (smoothing & cleaning) ---")
    filtered_new_dataset = []

    for traj in new_dataset:
        raw_obs = traj["obs"]
        raw_acts = traj["acts"]

        if len(raw_acts) < 10:
            continue

        aligned_obs = raw_obs
        aligned_acts = raw_acts

        if len(aligned_acts) >= 5 and aligned_acts.shape[1] >= 3:
            window_len = min(11, len(aligned_acts) if len(aligned_acts) % 2 != 0 else len(aligned_acts) - 1)
            if window_len > 3:
                aligned_acts[:, :3] = savgol_filter(aligned_acts[:, :3], window_length=window_len, polyorder=3, axis=0)
 
        filtered_new_dataset.append({
            "obs": np.array(aligned_obs, dtype=np.float32),
            "acts": np.array(aligned_acts, dtype=np.float32),
            "terminal": traj["terminal"],
        })

    if len(filtered_new_dataset) > 3:
        filtered_new_dataset.sort(key=lambda x: len(x["acts"]))
        keep_count = int(len(filtered_new_dataset) * 0.75)
        removed = len(filtered_new_dataset) - keep_count
        filtered_new_dataset = filtered_new_dataset[:keep_count]
        print(f"Discarded {removed} meandering/slow episodes.")

    final_dataset = existing_dataset + filtered_new_dataset
    if len(new_dataset) == 0:
        print("\nNo new episodes completed. Saving existing data.")
        final_dataset = existing_dataset
 
    with open(output_file, "wb") as f:
        pickle.dump(final_dataset, f)
    print(f"Filtered Dataset saved successfully with {len(final_dataset)} optimized episodes.")


def run_operator_session(input_source: InputSource, task_name: str, total_episodes: int, output_file: str):
    cfg = TASK_CONFIG[task_name]
    env_id = cfg["env_id"]
    act_dim = cfg["act_dim"]
 
    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                existing_dataset = pickle.load(f)
            print(f"Loaded {len(existing_dataset)} existing episodes from {output_file}.")
        except Exception:
            existing_dataset = []
    else:
        existing_dataset = []
 
    initial_count = len(existing_dataset)
    new_dataset = []

    # ===== ENVIRONMENT (no SmoothFrankaWrapper) =====
    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=1000)
    flat_env = FlattenGoalEnv(raw_env)
    grip_env = BinaryGripperActionWrapper(flat_env, close_thresh=0.2, open_thresh=0.6)
    rel_env = RelativeGoalWrapper(grip_env, ee_z_offset=EE_Z_OFFSET)
    env = FixDoneWrapper(rel_env)
 
    env_act_dim = env.action_space.shape[0]
    if env_act_dim != act_dim:
        raise ValueError(f"Action dim mismatch: {env_act_dim} vs {act_dim}")
 
    env_ready_hook_fired = False
 
    try:
        ep = 0
        while ep < total_episodes:
            obs, info = env.reset()
 
            if not env_ready_hook_fired:
                input_source.on_env_ready(env)
                env_ready_hook_fired = True
 
            input_source.reset_state()

            gripper_state = 1.0
            done = False
            aborted = False
            ep_obs = [obs]
            ep_acts = []
 
            current_ep_num = initial_count + ep + 1
            print(f"\n--- Recording Episode {current_ep_num}/{initial_count + total_episodes} ---")
            print("Left = CLOSE | Right = OPEN | Both = ABORT")
            print(f"Sensitivity: {SENSITIVITY}")
            print("Push spacemouse = move arm | Release = HOLD POSITION")
 
            next_tick = time.perf_counter()
            step_count = 0
 
            while not done:
                if input_source.should_abort():
                    print("Aborted.")
                    aborted = True
                    break

                raw_action = input_source.get_action(act_dim)
                
                # Scale spacemouse input directly
                # Release = [0,0,0] = mocap target stays = arm holds position!
                dx = raw_action[0] * SENSITIVITY
                dy = raw_action[1] * SENSITIVITY
                dz = raw_action[2] * SENSITIVITY
                
                # Update gripper
                if act_dim == 4:
                    gripper_input = raw_action[3]
                    if gripper_input < 0.2:
                        gripper_state = -1.0
                    elif gripper_input > 0.6:
                        gripper_state = 1.0
                
                # Build action
                if act_dim == 4:
                    action = np.array([dy, dx, dz, gripper_state], dtype=np.float32)
                else:
                    action = np.array([dy, dx, dz], dtype=np.float32)
                
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_obs.append(next_obs)
                ep_acts.append(action)
                obs = next_obs
                step_count += 1

                if step_count % 200 == 0:
                    print(f"  Step {step_count}: raw={raw_action[:3].round(2)} -> action={action[:3].round(3)}")

                if info.get("is_success", False):
                    print(f"  ✓ Success at step {step_count}!")

                next_tick += CONTROL_DT
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
            
            if aborted:
                print(f"Trial discarded.")
            elif truncated:
                print(f"Trial discarded (max steps).")
            elif terminated:
                new_dataset.append({
                    "obs": np.array(ep_obs, dtype=np.float32),
                    "acts": np.array(ep_acts, dtype=np.float32),
                    "terminal": True,
                })
                print(f"✓ Saved with {len(ep_acts)} steps.")
                ep += 1
            else:
                print(f"Trial discarded.")
 
    except KeyboardInterrupt:
        print("\nInterrupted.")
 
    finally:
        input_source.close()
        env.close()
        _filter_and_save(existing_dataset, new_dataset, output_file)