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

class SpaceMouseThread:
    """Dedicated background thread to drain the SpaceMouse USB HID queue in real time."""
    def __init__(self, device):
        self.device = device
        self.latest_state = None
        self.running = True
        # Daemon thread automatically terminates when the main script exits
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
            # Poll aggressively at ~1000 Hz to clear the C-buffer instantly
            time.sleep(0.001)

    def get_state(self):
        return self.latest_state

    def stop(self):
        self.running = False

class FlattenGoalEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = env.observation_space.spaces
        low = np.concatenate([spaces['observation'].low, spaces['achieved_goal'].low, spaces['desired_goal'].low])
        high = np.concatenate([spaces['observation'].high, spaces['achieved_goal'].high, spaces['desired_goal'].high])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])

def run_operator_session(env_id="FrankaPickAndPlaceSparse-v0", total_episodes=5, output_file="operator_data.pkl"):
    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                dataset = pickle.load(f)
            print(f"📂 Loaded existing dataset with {len(dataset)} episodes from '{output_file}'.")
        except Exception as e:
            print(f"⚠️ Could not load existing file ({e}). Starting a new dataset.")
            dataset = []
    else:
        dataset = []
    initial_count = len(dataset)
    # First, test if any SpaceMouse is visible to Python
    try:
        devices = pyspacemouse.get_all_hid_devices()
        if not devices:
            print("❌ Error: No SpaceMouse detected. Ensure it's plugged in and the official 3Dconnexion driver is completely closed!")
            return
        print(f"Found SpaceMouse device: {devices[0]}")
    except Exception as e:
        print(f"HID check crashed: {e}")

    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=350)
    env = FlattenGoalEnv(raw_env)
    dataset = []

    def applydeadzone(value, threshold=0.30):
        return value if abs(value) > threshold else 0.0
    def apply_cubic_scaling(value):
        return value ** 3
    

    print(f"\n=== SpaceMouse Teleoperation Mode ===")
    print("Instructions:\n- Move Joystick: Move Franka End-Effector (X, Y, Z)")
    print("- Left Button (1): CLOSE Gripper | Right Button (2): OPEN Gripper")
    print("- Press BOTH Buttons simultaneously (3): Save Episode and Move Next\n")

    # Use the context manager to natively handle connection lifecycles without silent failures
    with pyspacemouse.open() as device:
        sm_thread = SpaceMouseThread(device) 
        for ep in range(total_episodes):
            obs, info = env.reset()
            done = False
            
            ep_obs = [obs]
            ep_acts = []
            gripper_val = 1.0  # Initial open state
            GRIPPER_MIN = -1.0
            GRIPPER_MAX = 1.0
            smoothed_action = np.zeros(3)
            smoothed_rot = np.zeros(3, dtype=np.float32)
            alpha = 0.6
            
            current_ep_num = initial_count + ep + 1
            print(f"\n--- Recording Episode {current_ep_num} (Session {ep + 1}/{total_episodes}) ---")
            
            while not done:
                # Continuous polling loop
                state = sm_thread.get_state()
                if state is None:
                    time.sleep(0.01)
                    continue
                
                # Check button bitmask logic safely
                left, right = state.buttons
                if left and right:
                    print(f"Episode {ep + 1} flagged complete by operator.")
                    break
                elif left:
                    gripper_val = -1.0
                elif right:
                    gripper_val = 1.0
                gripper_val = float(np.clip(gripper_val, GRIPPER_MIN, GRIPPER_MAX))
                # Assign translation telemetry
                scale = 0.20
                rot_scale = 0.20
                deadzone = 0.20
                fx = applydeadzone(state.x, threshold=deadzone)
                fy = applydeadzone(state.y, threshold=deadzone)
                fz = applydeadzone(state.z, threshold=deadzone)
                if fx == 0.0 and fy == 0.0 and fz == 0.0:
                    smoothed_action = np.zeros(3, dtype=np.float32)
                    dx, dy, dz = 0.0, 0.0, 0.0
                else:
                    fx = apply_cubic_scaling(fx)
                    fy = apply_cubic_scaling(fy)
                    fz = apply_cubic_scaling(fz)
                    raw_delta = np.array([fx * scale, fy * scale, fz * scale], dtype=np.float32)
                    smoothed_action = alpha * raw_delta + (1.0 - alpha) * smoothed_action
                    dx, dy, dz = smoothed_action[0], smoothed_action[1], smoothed_action[2]
                r_roll = getattr(state, 'roll', getattr(state, 'rx', 0.0))
                r_pitch = getattr(state, 'pitch', getattr(state, 'ry', 0.0))
                r_yaw = getattr(state, 'yaw', getattr(state, 'rz', 0.0))

                f_roll = applydeadzone(r_roll, threshold=deadzone)
                f_pitch = applydeadzone(r_pitch, threshold=deadzone)
                f_yaw = applydeadzone(r_yaw, threshold=deadzone)

                if f_roll == 0.0 and f_pitch == 0.0 and f_yaw == 0.0:
                    smoothed_rot = np.zeros(3, dtype=np.float32)
                    droll, dpitch, dyaw = 0.0, 0.0, 0.0
                else:
                    raw_rot = np.array([
                        apply_cubic_scaling(f_roll) * rot_scale,
                        apply_cubic_scaling(f_pitch) * rot_scale,
                        apply_cubic_scaling(f_yaw) * rot_scale
                    ], dtype=np.float32)
                    smoothed_rot = alpha * raw_rot + (1.0 - alpha) * smoothed_rot
                    droll, dpitch, dyaw = smoothed_rot[0], smoothed_rot[1], smoothed_rot[2]
                if env.action_space.shape[0] == 7:
                    # 7D Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    action = np.array([dx, dy, dz, droll, dpitch, dyaw, gripper_val], dtype=np.float32)
                else:
                    # Fallback for 4D Action space: [dx, dy, dz, gripper]
                    action = np.array([dx, dy, dz, gripper_val], dtype=np.float32)

                action = np.clip(action, env.action_space.low, env.action_space.high)
               
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                ep_obs.append(next_obs)
                ep_acts.append(action)
                
                
            if len(ep_acts) > 1:
                dataset.append({
                    "obs": np.array(ep_obs, dtype=np.float32),
                    "acts": np.array(ep_acts, dtype=np.float32),
                    "terminal": True
                })
                print(f"Saved Episode {current_ep_num} with {len(ep_acts)} steps.")
                
    env.close()
    
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    run_operator_session(env_id="FrankaPickAndPlaceSparse-v0", total_episodes=5)