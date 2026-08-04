import gymnasium as gym
import pickle
import numpy as np
import panda_mujoco_gym
import time
import matplotlib.pyplot as plt

from imitation.data import types
from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.logger import KVWriter

# Import all wrappers from your smooth_env.py file
from smooth_env import (
    FlattenGoalEnv, 
    SmoothFrankaWrapper, 
    RelativeGoalWrapper, 
    FixDoneWrapper,
    BinaryGripperActionWrapper,
    SmoothXYZActionWrapper
)


# --- 1. ENVIRONMENT FACTORY FUNCTION (Put it here!) ---
def make_env(env_id, max_steps=400, render_mode=None):
    raw_env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_steps)
    flat_env = FlattenGoalEnv(raw_env)
    smooth_env = SmoothFrankaWrapper(flat_env, dt=0.01)
    xyz_env = SmoothXYZActionWrapper(smooth_env, alpha=0.3)
    grip_env = BinaryGripperActionWrapper(xyz_env)
    rel_env = RelativeGoalWrapper(grip_env)
    fixed_env = FixDoneWrapper(rel_env)
    return fixed_env


# --- 2. CUSTOM LOGGER TO CAPTURE METRICS ---
class CustomLogCollector(KVWriter):
    """Custom logger to programmatically extract SB3/imitation training metrics."""
    def __init__(self):
        self.logs = []
    def write(self, key_values, key_excluded, step=0):
        self.logs.append(key_values.copy())
    def close(self):
        pass


def evaluate_success_rate(policy, eval_env, num_episodes=10, max_steps=250):
    successes = 0
    final_distances = []
    for _ in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_success = False
        step_count = 0
        
        while not done and step_count < max_steps:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated) or bool(truncated)
            step_count += 1
            
            if info.get("is_success", False) or terminated:
                ep_success = True
                
        if ep_success:
            successes += 1
        final_distances.append(info.get("dist_block_goal", np.nan))
            
    success_rate = (successes / num_episodes) * 100.0
    mean_final_distance = float(np.nanmean(final_distances))
    return success_rate, mean_final_distance
   
def evaluate_and_view_policy(env_id, policy, num_episodes=3, max_steps=250):
    print("\n--- Launching 3D Simulation Evaluation ---")
    # Call make_env with render_mode="human" for visual evaluation
    eval_env = make_env(env_id, max_steps=max_steps, render_mode="human")
    
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        step = 0
        print(f"Running Evaluation Episode {ep + 1}/{num_episodes}...")
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated) or bool(truncated)
            step += 1
            time.sleep(0.01)
            
        print(f"Episode {ep + 1} ended after {step} steps. Success: {info.get('is_success', False)}")
        
    eval_env.close()


def plot_training_metrics(epochs, prob_true_act, loss, save_path="bc_training_metrics.png"):
    if len(epochs) == 0:
        return
    plt.style.use('default') 
    fig, ax1 = plt.subplots(figsize=(8, 5))
    matlab_blue = '#0072BD'
    matlab_orange = '#D95319'
    
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', color=matlab_blue, fontsize=12, fontweight='bold')
    line1 = ax1.plot(epochs, loss, '-', color=matlab_blue, linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=matlab_blue)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    if len(epochs) > 1:
        ax1.set_xlim(min(epochs), max(epochs))
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Prob True Act', color=matlab_orange, fontsize=12, fontweight='bold')
    line2 = ax2.plot(epochs, prob_true_act, '-', color=matlab_orange, linewidth=2, label='Prob True Act')
    ax2.tick_params(axis='y', labelcolor=matlab_orange)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', framealpha=0.9)
    
    plt.title('Behavioral Cloning: Loss & Prob True Action', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"MATLAB training metrics plot saved successfully as '{save_path}'.")
    plt.close()


def plot_success_rate(epochs, success_rates, save_path="bc_success_rate.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, success_rates, marker='o', linewidth=2, color='#1f77b4', label='BC Policy Success Rate')
    
    plt.title('Behavioral Cloning: Success Rate vs. Training Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    
    plt.ylim(-5, 105)
    if len(epochs) > 1:
        plt.xlim(0, max(epochs))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Success Rate plot saved successfully as '{save_path}'.")
    plt.close() 

def plot_distance_to_goal_comparison(results, x_label="Environment Timesteps", save_path="il_distance_to_goal_comparison.png"):
    # plot distance-to-goal
    colors = {
        "BC": '#1f77b4',
        "AIRL": '#d62728',
        "GAIL": '#2ca02c',
    }

    plt.figure(figsize=(8, 5))

    max_x = 0
    for name, (x_vals, dist_vals) in results.items():
        color = colors.get(name, None)
        plt.plot(x_vals, dist_vals, marker='o', linewidth=2, color=color, label=f'{name} Mean Final Distance')
        if len(x_vals) > 1:
            max_x = max(max_x, max(x_vals))

    plt.title('Imitation Learning: Block-to-Goal Distance vs. Environment Timesteps', fontsize=14, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Mean Final Distance (m)', fontsize=12)

    if max_x > 0:
        plt.xlim(0, max_x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Distance-to-goal plot saved successfully as '{save_path}'.")
    plt.close()

def cosine_lr_schedule(progress_remaining: float) -> float:
    """Smoothly decays learning rate from 1e-4 down to 1e-5 over total epochs.
       progress_remaining starts at 1.0 (epoch 0) and goes down to 0.0 (final epoch).
    """
    initial_lr = 1e-4
    min_lr = 1e-5
    return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))


def train_on_operator_data(task_num=1, total_epochs=60, eval_freq=10, eval_episodes=5):
    if task_num == 1:
        data_path = "operator_data_push.pkl"
        save_path = "bc_spacemouse_model_push.pt"
        env_id = "FrankaPushSparse-v0"
    elif task_num == 2:
        data_path = "operator_data_pick_and_place.pkl"
        save_path = "bc_spacemouse_model_pick_and_place.pt"
        env_id = "FrankaPickAndPlaceSparse-v0"
    
    # 1. Training Environment
    train_env = RolloutInfoWrapper(make_env(env_id, max_steps=250))

    # 2. Headless Evaluation Environment
    headless_eval_env = make_env(env_id, max_steps=250)
    
    # Load raw pickled trajectories
    with open(data_path, "rb") as f:
        raw_trajectories = pickle.load(f)
        
    # Instantiate transformer directly for dataset processing
    dummy_env = gym.make(env_id)
    dummy_flat = FlattenGoalEnv(dummy_env)
    dummy_smooth = SmoothFrankaWrapper(dummy_flat, dt=0.01)
    dummy_xyz = SmoothXYZActionWrapper(dummy_smooth, alpha=0.3)
    dummy_grip = BinaryGripperActionWrapper(dummy_xyz)
    rel_transformer = RelativeGoalWrapper(dummy_grip)

    formatted_dataset = []
    MIN_STEPS_THRESHOLD = 10
    
    for traj in raw_trajectories:
        if len(traj["acts"]) < MIN_STEPS_THRESHOLD:
            continue
        dummy_rewards = np.zeros(len(traj["acts"]), dtype=np.float32)

        # FIXED: Pass actual human recorded gripper state for every frame!
        transformed_obs = []
        for i, o in enumerate(traj["obs"]):
            # Fetch recorded gripper action for frame i
            g_act = traj["acts"][min(i, len(traj["acts"]) - 1)][3]
            g_state = -1.0 if g_act < 0.2 else 1.0
            
            # Pass g_state as override_gripper
            transformed_obs.append(rel_transformer.observation(o, override_gripper=g_state))

        transformed_obs = np.array(transformed_obs, dtype=np.float32)

        formatted_dataset.append(
            types.TrajectoryWithRew(
                obs=transformed_obs,
                acts=traj["acts"],
                infos=None,
                terminal=traj["terminal"],
                rews=dummy_rewards
            )
        )
    dummy_env.close()
        
    print(f"Loaded {len(formatted_dataset)} valid operator runs for training.")
    
    # --- POLICY WITH COSINE LR DECAY ---
    custom_policy = ActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=cosine_lr_schedule,  # <--- COSINE DECAY (Fine-tunes late epochs)
        net_arch=[128, 128],
        log_std_init=-0.5
    )

    # --- BC TRAINER WITH TUNED WEIGHT DECAY ---
    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        policy=custom_policy,
        demonstrations=formatted_dataset,
        ent_weight=0.05,
        l2_weight=1e-2,                  # <--- TUNED to 1e-2 (Relaxes the heavy brake!)
        rng=np.random.default_rng(seed=42)
    )

    log_collector = CustomLogCollector()
    sb3_logger = getattr(bc_trainer.logger, "default_logger", bc_trainer.logger)
    sb3_logger.output_formats.append(log_collector)
    
    print("\nStarting Behavioral Cloning with Success Rate Tracking...")
    epochs_list = []
    success_rates_list = []
    loss_list = []
    prob_true_act_list = []
    bc_distances_list = []

    best_success_rate = -1.0

    initial_rate, initial_dist = evaluate_success_rate(bc_trainer.policy, headless_eval_env, num_episodes=10, max_steps=400)
    epochs_list.append(0)
    success_rates_list.append(initial_rate)
    bc_distances_list.append(initial_dist)
    print(f"Epoch   0/{total_epochs} | Baseline Success Rate: {initial_rate:5.1f}% | Dist: {initial_dist:.4f}")

    for epoch in range(1, total_epochs + 1):
        bc_trainer.train(n_epochs=1, progress_bar=False)

        latest_logs = log_collector.logs[-1] if log_collector.logs else {}
        epoch_loss = latest_logs.get("bc/loss", latest_logs.get("training/loss", latest_logs.get("loss", 0.0)))
        epoch_prob = latest_logs.get("bc/prob_true_act", latest_logs.get("training/prob_true_act", latest_logs.get("prob_true_act", 0.0)))

        loss_list.append(epoch_loss)
        prob_true_act_list.append(epoch_prob)

        if epoch % eval_freq == 0 or epoch == total_epochs:
            success_rate, dist = evaluate_success_rate(bc_trainer.policy, headless_eval_env, num_episodes=eval_episodes, max_steps=400)
            epochs_list.append(epoch)
            success_rates_list.append(success_rate)
            bc_distances_list.append(dist)
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                bc_trainer.policy.save(save_path)
                print(f"Epoch {epoch:3d}/{total_epochs} | Loss: {epoch_loss:.4f} | Prob Act: {epoch_prob:.4f} | Success: {success_rate:5.1f}% | Dist: {dist:.4f} NEW BEST MODEL!")
            else:
                print(f"Epoch {epoch:3d}/{total_epochs} | Loss: {epoch_loss:.4f} | Prob Act: {epoch_prob:.4f} | Success: {success_rate:5.1f}% | Dist: {dist:.4f}")
    headless_eval_env.close()

    if best_success_rate == -1.0:
        bc_trainer.policy.save(save_path)

    # Plot both graphs
    plot_success_rate(epochs_list, success_rates_list)
    plot_training_metrics(range(1, total_epochs + 1), prob_true_act_list, loss_list)
    plot_distance_to_goal_comparison(
        {"BC": (epochs_list, bc_distances_list)},
        x_label="Training Epochs",
        save_path="bc_distance_to_goal.png",
    )

    # Visual evaluation of robot movement
    evaluate_and_view_policy(env_id, bc_trainer.policy, num_episodes=3, max_steps=300)


if __name__ == "__main__":
    task = int(input("Please enter number for task you are training for. Push (1) or Pick and Place (2): "))
    train_on_operator_data(task_num=task, total_epochs=60, eval_freq=10, eval_episodes=5)