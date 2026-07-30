import gymnasium as gym
import pickle
import numpy as np
import panda_mujoco_gym
import time
import matplotlib.pyplot as plt

from imitation.data import types
from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper

# Import your wrappers (Make sure this matches your exact filename)
from training_code_push import FlattenGoalEnv, SmoothFrankaWrapper3D

# --- NEW IMPORT FOR BIGGER NEURAL NETWORK ---
from stable_baselines3.common.policies import ActorCriticPolicy

def evaluate_success_rate(policy, env_id="FrankaPushSparse-v0", num_episodes=10, max_steps=350):
    """Fast, headless evaluation to compute success rate (%)."""
    eval_raw = gym.make(env_id, max_episode_steps=250)
    flat_env = FlattenGoalEnv(eval_raw)
    eval_env = SmoothFrankaWrapper3D(flat_env)
    
    successes = 0
    for _ in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_success = False
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            if info.get("is_success", False) or terminated:
                ep_success = True
                
        if ep_success:
            successes += 1
            
    eval_env.close()
    return (successes / num_episodes) * 100.0

def evaluate_and_view_policy(policy, env_id="FrankaPushSparse-v0", num_episodes=3):
    """Opens the MuJoCo viewer and watches the trained BC agent perform the task live."""
    print("\n--- Launching 3D Simulation Evaluation ---")
    eval_raw = gym.make(env_id, render_mode="human", max_episode_steps=350)
    flat_env = FlattenGoalEnv(eval_raw)
    eval_env = SmoothFrankaWrapper3D(flat_env)
    
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        step = 0
        
        print(f"Running Evaluation Episode {ep + 1}/{num_episodes}...")
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            step += 1
            time.sleep(0.02)
            
        print(f"Episode {ep + 1} ended after {step} steps.")
        
    eval_env.close()

def plot_success_rate(epochs, success_rates, save_path="bc_push_success_rate.png"):
    """Plots Success Rate (%) vs Training Epochs using Matplotlib."""
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, success_rates, marker='o', linewidth=2, color='#ff7f0e', label='BC Policy (Push)')
    
    plt.title('Behavioral Cloning (Push): Success Rate vs. Training Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    
    plt.ylim(-5, 105)
    plt.xlim(0, max(epochs) if epochs else 20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully as '{save_path}'.")
    plt.show()

def train_on_operator_data(data_path="operator_data_push.pkl", total_epochs=500, eval_freq=100, eval_episodes=5):
    env_id = "FrankaPushSparse-v0"
    
    # 1. Instantiate wrapped environment matching data dimensionality
    raw_env = gym.make(env_id)
    flat_env = FlattenGoalEnv(raw_env)
    
    # CRITICAL: Include the 3D wrapper used during data collection!
    smooth_env = SmoothFrankaWrapper3D(flat_env)
    train_env = RolloutInfoWrapper(smooth_env)
    
    # 2. Load and parse the raw pickled operator trajectories
    with open(data_path, "rb") as f:
        raw_trajectories = pickle.load(f)
        
    formatted_dataset = []
    for traj in raw_trajectories:
        if len(traj["acts"]) < 5:
            continue
            
        dummy_rewards = np.zeros(len(traj["acts"]), dtype=np.float32)
        formatted_dataset.append(
            types.TrajectoryWithRew(
                obs=traj["obs"],
                acts=traj["acts"],
                infos=None,
                terminal=traj["terminal"],
                rews=dummy_rewards
            )
        )
        
    print(f"Loaded {len(formatted_dataset)} valid operator runs for training.")

    # 3. BUILD A BIGGER, FASTER-LEARNING NEURAL NETWORK
    custom_policy = ActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=lambda _: 1e-3,  # Fast, constant learning rate
        net_arch=[256, 256]          # 4x larger neural network capacity
    )

    # 4. Instantiate BC Trainer
    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=formatted_dataset,
        policy=custom_policy,        # <--- Inject our custom bigger brain
        rng=np.random.default_rng(seed=42)
    )
    
    # 5. Train Epoch-by-Epoch & Track Success Rate
    print("\nStarting Behavioral Cloning with Success Rate Tracking...")
    epochs_list = []
    success_rates_list = []

    # Baseline Evaluate at Epoch 0
    success_rate_initial = evaluate_success_rate(bc_trainer.policy, env_id=env_id, num_episodes=eval_episodes)
    epochs_list.append(0)
    success_rates_list.append(success_rate_initial)
    print(f"Epoch   0/{total_epochs:3d} | Success Rate: {success_rate_initial:5.1f}% (Untrained)")

    for epoch in range(1, total_epochs + 1):
        # Train 1 epoch at a time
        bc_trainer.train(n_epochs=1)
        
        # We ONLY evaluate every 100 epochs to speed up wall-clock time!
        if epoch % eval_freq == 0 or epoch == total_epochs:
            success_rate = evaluate_success_rate(bc_trainer.policy, env_id=env_id, num_episodes=eval_episodes)
            epochs_list.append(epoch)
            success_rates_list.append(success_rate)
            print(f"Epoch {epoch:3d}/{total_epochs:3d} | Success Rate: {success_rate:5.1f}%")
    
    # 6. Save the trained weights
    bc_trainer.policy.save("bc_push_spacemouse_model.pt")
    print("\nModel saved successfully as 'bc_push_spacemouse_model.pt'")

    # 7. Plot Success Rate vs. Epochs
    plot_success_rate(epochs_list, success_rates_list, save_path="bc_push_success_rate.png")

    # 8. Live 3D Visual Evaluation
    evaluate_and_view_policy(bc_trainer.policy, env_id=env_id, num_episodes=3)

if __name__ == "__main__":
    train_on_operator_data(data_path="operator_data_push.pkl", total_epochs=500, eval_freq=100, eval_episodes=5)