import gymnasium as gym
import numpy as np
import pickle
import os
import time
import sys
import pyspacemouse  # Core SpaceMouse driver wrapper
import panda_mujoco_gym 
import threading
import time

from scipy.signal import savgol_filter 
from smooth_env import FlattenGoalEnv, SmoothFrankaWrapper

class SpaceMouseThread:
    """thread running in background for spacemouse input readings"""
    def __init__(self, device):
        self.device = device
        self.latest_state = None
        self.running = True
        # background (daemon) thread automatically terminates when main script exits
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

    def get_state(self):
        return self.latest_state

    def stop(self):
        self.running = False

def make_teleop_env():
    raw_env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human", max_episode_steps=1000)
    flat_env = FlattenGoalEnv(raw_env)
    return SmoothFrankaWrapper(flat_env)

def smooth_deadzone(x, deadzone=0.08):
    if abs(x) < deadzone:
        return 0.0
    return np.sign(x) * (abs(x) - deadzone) / (1.0 - deadzone)

def run_operator_session(env_id="FrankaPickAndPlaceSparse-v0", total_episodes=5, output_file="operator_data_pick_and_place.pkl"):
    CONTROL_HZ = 100
    CONTROL_DT = 1.0 / CONTROL_HZ

    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                existing_dataset = pickle.load(f)
            print(f"Loaded {len(existing_dataset)} existing episodes.")
        except Exception:
            existing_dataset = []
    else:
        existing_dataset = []

    initial_count = len(existing_dataset)
    new_dataset = []

    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=1000)
    flat_env = FlattenGoalEnv(raw_env)
    env = SmoothFrankaWrapper(flat_env, dt=CONTROL_DT)

    GLITCH_FILTER_ALPHA = 0.55

    try:
        with pyspacemouse.open() as device:
            sm_thread = SpaceMouseThread(device)
        
            for ep in range(total_episodes):
                obs, info = env.reset()
                done = False
                aborted = False
                terminated = False
                truncated = False

                ep_obs = [obs]
                ep_acts = []
                gripper_val = 1.0
            
                current_ep_num = initial_count + ep + 1
                print(f"\n--- Recording Episode {current_ep_num} ---")

                filtered_raw = {"x": 0.0, "y": 0.0, "z": 0.0}
                next_tick = time.perf_counter()

                while not done:
                    state = sm_thread.get_state()
                    if state is None:
                        time.sleep(0.002)
                        continue

                # Gripper buttons
                    left, right = state.buttons
                    if left and right:
                        print(f"Episode {current_ep_num} flagged complete by operator.")
                        aborted = True
                        break
                    elif left:
                        gripper_val = -1.0
                    elif right:
                        gripper_val = 1.0

                    # Low-pass filter raw SpaceMouse values to cancel snapback glitch
                    filtered_raw["x"] = (1 - GLITCH_FILTER_ALPHA) * filtered_raw["x"] + GLITCH_FILTER_ALPHA * state.x
                    filtered_raw["y"] = (1 - GLITCH_FILTER_ALPHA) * filtered_raw["y"] + GLITCH_FILTER_ALPHA * state.y
                    filtered_raw["z"] = (1 - GLITCH_FILTER_ALPHA) * filtered_raw["z"] + GLITCH_FILTER_ALPHA * state.z

                    # Raw normalized SpaceMouse action vector
                    dx = smooth_deadzone(filtered_raw["x"], deadzone=0.12)
                    dy = smooth_deadzone(filtered_raw["y"], deadzone=0.12)
                    dz = smooth_deadzone(filtered_raw["z"], deadzone=0.12)
                    raw_action = np.array([dx, dy, dz, gripper_val], dtype=np.float32)
                    raw_action = np.clip(raw_action, env.action_space.low, env.action_space.high)

                    # Environment handles internal smoothing, scaling, and dynamics step
                    next_obs, reward, terminated, truncated, info = env.step(raw_action)
                    done = terminated or truncated

                    # Save raw action (this is what AIRL / BC policy will learn to output)
                    ep_obs.append(next_obs)
                    ep_acts.append(raw_action)

                    obs = next_obs

                    next_tick += CONTROL_DT
                    sleep = next_tick - time.perf_counter()

                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        next_tick = time.perf_counter()
                
                if aborted:
                    print(f"Trial {current_ep_num} DISCARDED (Cancelled by operator).")
                elif truncated:
                    print(f"Trial {current_ep_num} DISCARDED (Reached max episode steps).")
                elif terminated:
                    new_dataset.append({
                        "obs": np.array(ep_obs, dtype=np.float32),
                        "acts": np.array(ep_acts, dtype=np.float32),
                        "terminal": True
                    })
                    print(f"Saved Episode {current_ep_num} with {len(ep_acts)} steps!")
                else:
                    print(f"Trial {current_ep_num} DISCARDED (Ended without success).")

    except KeyboardInterrupt:
            print("\nSession interrupted by user.")
    finally:
        print("Closing Gymnasium environment...")
        env.close()
        print("\n--- Applying Dataset Filters (Alignment, Smoothing, Downsampling) ---")
        filtered_new_dataset = []
                
        REACTION_SHIFT = 5  # Shift actions back 150ms to align with human reaction time
        SKIP_FRAMES = 1      # Downsample from 100Hz to 20Hz
                
        for traj in new_dataset:
            raw_obs = traj["obs"]   
            raw_acts = traj["acts"] 
                    
            if len(raw_acts) < REACTION_SHIFT + 10:
                continue # Skip corrupted/extremely short runs
                        
            # Filter 1: Human Reaction Time Alignment
            aligned_acts = raw_acts[REACTION_SHIFT:]
            aligned_obs = raw_obs[:len(aligned_acts) + 1] # Ensure obs is always exactly acts + 1
                    
                # Filter 2: Action Smoothing (Savitzky-Golay removes human hand jitter)
            window_len = min(11, len(aligned_acts) if len(aligned_acts) % 2 != 0 else len(aligned_acts) - 1)
            if window_len > 3:
                aligned_acts = savgol_filter(aligned_acts, window_length=window_len, polyorder=3, axis=0)
                        
            # Filter 3: Downsampling (Skip frames so state changes are obvious to NN)
            indices = np.arange(0, len(aligned_acts), SKIP_FRAMES)
            downsampled_acts = aligned_acts[indices]
                    
            # Get matching observations + terminal observation
            obs_indices = np.append(indices, indices[-1] + 1)
            downsampled_obs = aligned_obs[obs_indices]
                    
            filtered_new_dataset.append({
                "obs": np.array(downsampled_obs, dtype=np.float32),
                "acts": np.array(downsampled_acts, dtype=np.float32),
                "terminal": traj["terminal"]
            })
                    
        # Filter 4: Remove Meandering (Keep top 75% fastest episodes)
        if len(filtered_new_dataset) > 3:
            filtered_new_dataset.sort(key=lambda x: len(x["acts"]))
            keep_count = int(len(filtered_new_dataset) * 0.75)
            removed = len(filtered_new_dataset) - keep_count
            filtered_new_dataset = filtered_new_dataset[:keep_count]
            print(f"Discarded {removed} meandering/slow episodes.")
        
        final_dataset = existing_dataset + filtered_new_dataset
        
        if len(new_dataset) == 0:
            print("\nNo new episodes were completed. Saving existing data.")
            final_dataset = existing_dataset
        
                
        
                
                    
                # ====================================================================
        
        with open(output_file, "wb") as f:
            pickle.dump(final_dataset, f)
        print(f"Filtered Dataset saved successfully with {len(final_dataset)} optimized episodes.")
        
if __name__ == "__main__":
    run_operator_session(env_id="FrankaPickAndPlaceSparse-v0", total_episodes=5)