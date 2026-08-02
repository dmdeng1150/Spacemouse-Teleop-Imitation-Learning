import gymnasium as gym
import numpy as np
import time
import os
import panda_mujoco_gym

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

# Import all 4 wrappers from your training script
from smooth_env import (
    FlattenGoalEnv, 
    SmoothFrankaWrapper, 
    RelativeGoalWrapper, 
    FixDoneWrapper
)

def test_saved_model(
    model_path="bc_spacemouse_model_pick_and_place.pt", 
    env_id="FrankaPickAndPlaceSparse-v0", 
    num_episodes=5,
    max_steps=400,
    task_num=2
):
    # 1. File existence check
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return

    print(f"Loading model from '{model_path}'...")

    # 2. Load policy based on model file extension (.pt vs .zip)
    if model_path.endswith(".pt"):
        # Load Behavioral Cloning (BC) Policy
        policy = ActorCriticPolicy.load(model_path)
    else:
        # Load PPO / AIRL Model (.zip)
        model = PPO.load(model_path)
        policy = model.policy

    print("✅ Model loaded successfully!")

    # 3. Create evaluation environment with full wrapper pipeline
    print("\n--- Launching 3D Simulation Evaluation ---")
    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=max_steps)
    flat_env = FlattenGoalEnv(raw_env)
    smooth_env = SmoothFrankaWrapper(flat_env, dt=0.01) # Matched with 100Hz training!
    rel_env = RelativeGoalWrapper(smooth_env)            # Appends 6 relative features
    env = FixDoneWrapper(rel_env)                         # Guarantees boolean signals

    success_count = 0

    # 4. Run evaluation episodes
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        ep_success = False

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

        while not done:
            # Predict action deterministically from trained policy
            action, _ = policy.predict(obs, deterministic=True)
            action = np.array(action, copy=True).flatten()

            # Snap continuous gripper prediction to binary state for Pick & Place
            if (task_num == 2 or "PickAndPlace" in env_id) and len(action) > 3:
                action[3] = 1.0 if action[3] > 0.3 else -1.0

            # Step environment forward
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            step += 1

            # Extract live distance diagnostics from observation
            # Relative features are appended at obs[-6:-3] and obs[-3:]
            dist_ee_block = np.linalg.norm(obs[-6:-3])
            dist_block_goal = np.linalg.norm(obs[-3:])

            if info.get("is_success", False) or terminated:
                ep_success = True

            if step % 50 == 0 or done:
                print(f"Step {step:03d} | Dist EE->Block: {dist_ee_block:.3f}m | Dist Block->Goal: {dist_block_goal:.3f}m")

            # Pace loop so 3D GUI renders smoothly (~100 FPS)
            time.sleep(0.01)

        if ep_success or info.get("is_success", False):
            success_count += 1
            print(f"Result: SUCCESS in {step} steps!")
        else:
            print(f"Result: FAILED (reached max steps).")

    env.close()

    # Print summary results
    success_rate = (success_count / num_episodes) * 100.0
    print(f"\n==========================================")
    print(f"Final Evaluation Summary:")
    print(f"   Successful Runs: {success_count}/{num_episodes}")
    print(f"   Success Rate:    {success_rate:.1f}%")
    print(f"==========================================")


if __name__ == "__main__":
    # Test Pick and Place BC Model (Task 2):
    test_saved_model(
        model_path="bc_spacemouse_model_pick_and_place.pt", 
        env_id="FrankaPickAndPlaceSparse-v0",
        task_num=2
    )

    # To test Pick and Place AIRL Model (.zip), uncomment below:
    # test_saved_model(
    #     model_path="airl_panda_spacemouse_model.zip", 
    #     env_id="FrankaPickAndPlaceSparse-v0",
    #     task_num=2
    # )

    # To test Push BC Model (Task 1), uncomment below:
    # test_saved_model(
    #     model_path="bc_spacemouse_model_push.pt", 
    #     env_id="FrankaPushSparse-v0",
    #     task_num=1
    # )