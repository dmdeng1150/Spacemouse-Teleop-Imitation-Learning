"""
Diagnostic: isolate whether the "arm drifts back to start pose" behavior is
coming from panda_mujoco_gym itself, completely independent of the
SpaceMouse/teleop script.

What it does:
  1. Resets the env.
  2. Drives the arm away from its start pose using a few steps of a FIXED,
     hardcoded action (no spacemouse involved at all).
  3. Switches to a hardcoded action of EXACTLY (0, 0, 0, gripper) -- a
     mathematically guaranteed zero, no filtering, no thread, no hardware.
  4. Holds that zero action for ~3 seconds and prints the end-effector /
     gripper position (best-effort key lookup) every step.

If the printed position visibly walks back toward its step-0 value while
step 3's action is verifiably (0,0,0,gripper), the drift is 100% coming from
panda_mujoco_gym's internal env/controller, not from teleop input -- and the
fix has to happen in that package's _set_action / step code, not here.

If the position HOLDS steady under zero action, the drift is coming from
somewhere else and we're back to the teleop script.
"""
import gymnasium as gym
import numpy as np
import time
import panda_mujoco_gym


def find_position_key(obs):
    """Best-effort: print all obs keys/shapes once so we can see what's
    available, since we don't have panda_mujoco_gym's exact obs layout."""
    if isinstance(obs, dict):
        for k, v in obs.items():
            print(f"  obs['{k}'] shape={np.shape(v)} first_vals={np.array(v).flatten()[:3]}")


def run_diagnostic(env_id="FrankaPickAndPlaceSparse-v0"):
    env = gym.make(env_id, render_mode="human", max_episode_steps=1000)
    obs, info = env.reset()

    print("\n=== Observation structure at reset (for reference) ===")
    find_position_key(obs)

    print("\n=== Phase 1: driving away from start with a fixed action for 60 steps ===")
    # Fixed, non-spacemouse action: push forward+up steadily.
    drive_action = np.array([0.8, 0.0, 0.3, 1.0], dtype=np.float32)
    for i in range(60):
        obs, reward, terminated, truncated, info = env.step(drive_action)
        time.sleep(1.0 / 100)

    print("Position after driving away from start:")
    find_position_key(obs)

    print("\n=== Phase 2: holding EXACTLY (0,0,0,gripper) for 300 steps (~3s) ===")
    zero_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for i in range(300):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        if i % 30 == 0:  # print roughly 10 times over the 3 seconds
            print(f"step {i}:")
            find_position_key(obs)
        time.sleep(1.0 / 100)

    print("\n=== Done. Compare the printed positions across Phase 2. ===")
    print("If they walk back toward the reset-time values above with action")
    print("verifiably (0,0,0,1.0) the whole time, this is a panda_mujoco_gym")
    print("internal behavior, not a teleop/input issue.")

    env.close()


if __name__ == "__main__":
    run_diagnostic()