import gymnasium as gym
import numpy as np
import pickle
import os
import time
import sys
import pyspacemouse  # Core SpaceMouse driver wrapper
import panda_mujoco_gym 
import threading
from smooth_env import FlattenGoalEnv, SmoothFrankaWrapper3D

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
            time.sleep(0.001)

    def get_state(self):
        return self.latest_state

    def stop(self):
        self.running = False

def make_teleop_env():
    raw_env = gym.make("FrankaPushSparse-v0", render_mode="human", max_episode_steps=1000)
    flat_env = FlattenGoalEnv(raw_env)
    return SmoothFrankaWrapper3D(flat_env)

def smooth_deadzone(x, deadzone=0.08):
    if abs(x) < deadzone:
        return 0.0
    return np.sign(x) * (abs(x) - deadzone) / (1.0 - deadzone)

def run_operator_session(env_id="FrankaPushSparse-v0", total_episodes=5, output_file="operator_data_push.pkl"):
    CONTROL_HZ = 100
    CONTROL_DT = 1.0 / CONTROL_HZ

    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                dataset = pickle.load(f)
            print(f"Loaded {len(dataset)} existing episodes.")
        except Exception:
            dataset = []
    else:
        dataset = []

    initial_count = len(dataset)

    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=1000)
    flat_env = FlattenGoalEnv(raw_env)
    env = SmoothFrankaWrapper3D(flat_env, dt=CONTROL_DT)

    # --- DYNAMICALLY DETECT ACTION DIMENSION (3 for Push, 4 for Pick/Place) ---
    act_dim = env.action_space.shape[0]
    print(f"Detected Environment Action Dimension: {act_dim}D")

    GLITCH_FILTER_ALPHA = 0.55

    # Wrapped in try/finally to ensure env.close() runs if stopped early
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

                    # Buttons: Use both buttons to abort
                    left, right = state.buttons
                    if left and right:
                        print(f"Episode {current_ep_num} flagged complete by operator.")
                        aborted = True
                        break

                    # Low-pass filter raw SpaceMouse values
                    filtered_raw["x"] = (1 - GLITCH_FILTER_ALPHA) * filtered_raw["x"] + GLITCH_FILTER_ALPHA * state.x
                    filtered_raw["y"] = (1 - GLITCH_FILTER_ALPHA) * filtered_raw["y"] + GLITCH_FILTER_ALPHA * state.y
                    filtered_raw["z"] = (1 - GLITCH_FILTER_ALPHA) * filtered_raw["z"] + GLITCH_FILTER_ALPHA * state.z

                    # Calculate smoothed translations
                    dx = smooth_deadzone(filtered_raw["x"], deadzone=0.12)
                    dy = smooth_deadzone(filtered_raw["y"], deadzone=0.12)
                    dz = smooth_deadzone(filtered_raw["z"], deadzone=0.12)

                    # --- CREATE 3D OR 4D ACTION VECTOR BASED ON ENVIRONMENT ---
                    raw_action = np.array([dx, dy, dz], dtype=np.float32)

                    # Ensure action is within bounds
                    raw_action = np.clip(raw_action, env.action_space.low, env.action_space.high)

                    # Environment handles internal smoothing, scaling, and dynamics step
                    next_obs, reward, terminated, truncated, info = env.step(raw_action)
                    done = terminated or truncated

                    # Save raw action (this is what AIRL / BC policy will learn to output)
                    is_idle = (dx == 0.0 and dy == 0.0 and dz == 0.0)
                    if not is_idle:
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
                    dataset.append({
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
        
        with open(output_file, "wb") as f:
            pickle.dump(dataset, f)
        print("Dataset saved successfully.")

if __name__ == "__main__":
    run_operator_session(env_id="FrankaPushSparse-v0", total_episodes=5)