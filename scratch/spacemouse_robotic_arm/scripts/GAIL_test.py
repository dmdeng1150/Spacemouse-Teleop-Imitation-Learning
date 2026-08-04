import os
import pickle
import time
import torch
import numpy as np
import gymnasium as gym
import panda_mujoco_gym
import matplotlib.pyplot as plt

from imitation.data import types
from imitation.algorithms import bc  
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.data.wrappers import RolloutInfoWrapper

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# Import wrappers from smooth_env.py
from smooth_env import (
    FlattenGoalEnv, 
    SmoothFrankaWrapper, 
    SmoothXYZActionWrapper,
    BinaryGripperActionWrapper,
    RelativeGoalWrapper, 
    FixDoneWrapper
)


# =====================================================================
# 1. DENSE MULTI-STAGE PROGRESS REWARD WRAPPER
# =====================================================================

class DenseProgressRewardWrapper(gym.Wrapper):
    """Provides continuous stage-by-stage guidance rewards:
       Stage 1: Reward for bringing end-effector closer to block.
       Stage 2: +2.0 Bonus when block is lifted off the table (Z > 3.5cm).
       Stage 3: Reward for bringing lifted block closer to target goal.
       Stage 4: +10.0 Bonus upon true task completion.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        dist_ee_block = info.get("dist_ee_block", 0.0)
        dist_block_goal = info.get("dist_block_goal", 0.0)
        
        # Block Z height is feature 36 in the 41D observation vector
        block_z = obs[36] if len(obs) == 41 else 0.02

        # 1. Approach Reward (Drives gripper to block)
        r_approach = -1.0 * dist_ee_block

        # 2. Lift Bonus (Triggers when block is lifted > 3.5cm)
        is_lifted = block_z > 0.035
        r_lift = 2.0 if is_lifted else 0.0

        # 3. Transport Reward (Drives lifted block to target goal)
        r_transport = -2.0 * dist_block_goal if is_lifted else 0.0

        # 4. Final Completion Bonus
        is_success = info.get("is_success", False)
        r_success = 10.0 if is_success else 0.0

        dense_reward = r_approach + r_lift + r_transport + r_success

        return obs, float(dense_reward), bool(terminated), bool(truncated), info


# =====================================================================
# 2. UNIFIED ENVIRONMENT FACTORY
# =====================================================================

def make_env(render_mode=None, max_steps=250, seed=42, rank=0):
    def _init():
        raw_env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode=render_mode, max_episode_steps=max_steps)
        if seed is not None:
            raw_env.action_space.seed(seed + rank)
        flat_env = FlattenGoalEnv(raw_env)
        smooth_franka = SmoothFrankaWrapper(flat_env, dt=0.01)
        smooth_xyz = SmoothXYZActionWrapper(smooth_franka, alpha=0.65)
        grip_env = BinaryGripperActionWrapper(smooth_xyz, close_thresh=0.2, open_thresh=0.6)
        rel_env = RelativeGoalWrapper(grip_env, ee_z_offset=0.058)
        fixed_env = FixDoneWrapper(rel_env)
        
        # Dense Progress Guidance Wrapper
        guided_env = DenseProgressRewardWrapper(fixed_env)
        return RolloutInfoWrapper(guided_env)
    return _init


# =====================================================================
# 3. SUCCESS RATE EVALUATION & PLOTTING
# =====================================================================

def evaluate_success_rate(policy, eval_env, num_episodes=10, max_steps=250):
    """Evaluates policy success rate directly across evaluation episodes."""
    successes = 0
    for _ in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        step = 0
        ep_success = False

        while not done and step < max_steps:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated) or bool(truncated)
            step += 1

            if info.get("is_success", False):
                ep_success = True

        if ep_success:
            successes += 1

    return (successes / num_episodes) * 100.0


def plot_success_rate(timesteps, success_rates, save_path="gail_success_rate.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, success_rates, marker='o', linewidth=2, color='#0072BD', label='GAIL Policy Success Rate')
    plt.title('GAIL Training: Success Rate vs. Environment Timesteps', fontsize=14, fontweight='bold')
    plt.xlabel('Environment Timesteps', fontsize=12, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    plt.ylim(-5, 105)
    if len(timesteps) > 1:
        plt.xlim(min(timesteps), max(timesteps))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📊 Success rate plot saved successfully as '{save_path}'.")
    plt.close()


def evaluate_and_view_policy(policy, num_episodes=5):
    print("\n--- Launching 3D Simulation Visual Evaluation ---")
    eval_env_fn = make_env(render_mode="human", max_steps=250)
    eval_env = eval_env_fn()
    
    successes = 0
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        step = 0
        print(f"\nRunning Visual Evaluation Episode {ep + 1}/{num_episodes}...")
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated) or bool(truncated)
            step += 1
            time.sleep(0.01)
            
            if info.get("is_success", False):
                successes += 1

        print(f"Episode {ep + 1} ended after {step} steps. Success: {info.get('is_success', False)}")
        
    print(f"\nFinal Visual Evaluation Success Rate: {(successes / num_episodes) * 100:.1f}%")
    eval_env.close()


# =====================================================================
# 4. MAIN GAIL TRAINING PIPELINE
# =====================================================================

def train_gail_on_operator_data(
    data_path="operator_data_pick_and_place_spacemouse.pkl",
    model_save_path="gail_panda_spacemouse_model",
    total_timesteps=100_000,
    eval_freq_steps=10_000,
    bc_pretrain_epochs=100
):
    num_cpu = os.cpu_count() or 8  
    
    # 1. Parallel Vectorized Environments
    print(f"Launching {num_cpu} parallel physics simulations for GAIL...")
    raw_venv = SubprocVecEnv([make_env(seed=42, rank=i) for i in range(num_cpu)])
    venv = VecNormalize(raw_venv, norm_obs=False, norm_reward=True, clip_reward=5.0)
    
    # 2. Load Dataset
    if not os.path.exists(data_path):
        data_path = "operator_data_pick_and_place_kb.pkl"

    with open(data_path, "rb") as f:
        raw_trajectories = pickle.load(f)
        
    sample_env = gym.make("FrankaPickAndPlaceSparse-v0")
    sample_flat = FlattenGoalEnv(sample_env)
    sample_franka = SmoothFrankaWrapper(sample_flat)
    sample_xyz = SmoothXYZActionWrapper(sample_franka)
    sample_grip = BinaryGripperActionWrapper(sample_xyz)
    rel_transformer = RelativeGoalWrapper(sample_grip, ee_z_offset=0.058)

    formatted_dataset = []
    for traj in raw_trajectories:
        dummy_rewards = np.zeros(len(traj["acts"]), dtype=np.float32)
        
        transformed_obs = []
        for i, o in enumerate(traj["obs"]):
            g_act = traj["acts"][min(i, len(traj["acts"]) - 1)][3] if len(traj["acts"][0]) >= 4 else 1.0
            g_state = -1.0 if g_act < 0.2 else 1.0
            transformed_obs.append(rel_transformer.observation(o, override_gripper=g_state))

        transformed_obs = np.array(transformed_obs, dtype=np.float32)

        formatted_dataset.append(
            types.TrajectoryWithRew(
                obs=transformed_obs,
                acts=np.array(traj["acts"], dtype=np.float32),
                infos=None,
                terminal=traj["terminal"],
                rews=dummy_rewards
            )
        )
    sample_env.close()
        
    print(f"Loaded {len(formatted_dataset)} operator runs from '{data_path}' for GAIL training.")

    total_expert_transitions = sum(len(traj.acts) for traj in formatted_dataset)
    safe_batch_size = min(1024, total_expert_transitions)

    # 3. Generator Policy (PPO) with Higher Learning Rate (1e-4)
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=128,               
        n_steps=2048 // num_cpu,      
        ent_coef=0.001,              
        learning_rate=1e-4,           # Increased to 1e-4 for faster policy gradients
        gamma=0.99,
        clip_range=0.2,               
        n_epochs=10,                  
        vf_coef=0.5,                  # Active Value Function Coefficient
        seed=42,
        device="cpu"                 
    )
    
    # 4. Regularized BC Pre-training
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=learner.policy,
        demonstrations=formatted_dataset,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-4},
        ent_weight=0.02,
        l2_weight=1e-2,
        rng=np.random.default_rng(42),
    )
    
    print("\nPre-training Generator via Regularized BC (Slow Warm-up over 100 Epochs)...")
    bc_trainer.train(n_epochs=bc_pretrain_epochs)
    
    # 5. GAIL Reward Network
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=None,  
    )

    # 6. GAIL Trainer
    gail_trainer = GAIL(
        demonstrations=formatted_dataset,
        demo_batch_size=128,                 
        gen_replay_buffer_capacity=32768,    
        gen_train_timesteps=10_000,
        n_disc_updates_per_round=1,         # 1 update per round prevents discriminator overpowering
        disc_opt_kwargs={"lr": 1e-4, "weight_decay": 1e-4},  
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )

    # 7. Online GAIL Training Loop
    print("\nStarting Online GAIL Training with Success Rate Evaluation...")
    headless_eval_fn = make_env(render_mode=None, max_steps=250)
    headless_eval_env = headless_eval_fn()

    timesteps_list = []
    success_rates_list = []

    init_sr = evaluate_success_rate(learner.policy, headless_eval_env, num_episodes=10)
    timesteps_list.append(0)
    success_rates_list.append(init_sr)
    print(f"Step      0/{total_timesteps} | Post-BC Baseline Success Rate: {init_sr:5.1f}%")

    current_timesteps = 0
    rounds = total_timesteps // eval_freq_steps

    for r in range(1, rounds + 1):
        gail_trainer.train(total_timesteps=eval_freq_steps)
        current_timesteps += eval_freq_steps

        sr = evaluate_success_rate(learner.policy, headless_eval_env, num_episodes=10)
        timesteps_list.append(current_timesteps)
        success_rates_list.append(sr)
        print(f"Step {current_timesteps:6d}/{total_timesteps} | Success Rate: {sr:5.1f}%")

    headless_eval_env.close()

    # 8. Plot & Save Success Rate Graph
    plot_success_rate(timesteps_list, success_rates_list, save_path="gail_success_rate.png")

    venv.close()
    
    # 9. Save Model & Visual Evaluation
    gail_trainer.gen_algo.save(model_save_path)
    print(f"\nGAIL Model saved successfully as '{model_save_path}.zip'")

    evaluate_and_view_policy(gail_trainer.gen_algo.policy, num_episodes=5)


if __name__ == "__main__":
    train_gail_on_operator_data(
        data_path="operator_data_pick_and_place_spacemouse.pkl",
        model_save_path="gail_panda_spacemouse_model",
        total_timesteps=100_000,
        eval_freq_steps=10_000,
        bc_pretrain_epochs=100
    )