import gymnasium as gym
import pickle
import numpy as np
import panda_mujoco_gym
import time
import os
import matplotlib.pyplot as plt

from imitation.data import types
from imitation.algorithms import bc  
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.data.wrappers import RolloutInfoWrapper

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import KVWriter

from smooth_env import FlattenGoalEnv, SmoothFrankaWrapper, RelativeGoalWrapper, BinaryGripperActionWrapper, FixDoneWrapper

def observation(self, obs):
        # Extract 3D positions from the raw observation
        # In panda_mujoco_gym, ee_pos is obs[0:3], block_pos is obs[6:9], goal is obs[-3:]
    ee_pos = obs[0:3]
    block_pos = obs[6:9]
    goal_pos = obs[-3:]

    rel_ee_to_block = block_pos - ee_pos
    rel_block_to_goal = goal_pos - block_pos

    return np.concatenate([obs, rel_ee_to_block, rel_block_to_goal]).astype(np.float32)




# --- CUSTOM LOGGER ---
class CustomLogCollector(KVWriter):
    def __init__(self):
        self.logs = []
    def write(self, key_values, key_excluded, step=0):
        self.logs.append(key_values.copy())
    def close(self):
        pass


def plot_training_metrics(epochs, prob_true_act, loss, save_path="bc_pretraining_metrics.png"):
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
    
    plt.title('BC Pre-Training: Loss & Prob True Action', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_and_view_policy(policy, num_episodes=5):
    print("\n--- Launching 3D Simulation Evaluation ---")
    
    eval_raw = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human", max_episode_steps=250)
    eval_env_flat = FlattenGoalEnv(eval_raw)
    eval_env_smooth = SmoothFrankaWrapper(eval_env_flat)
    eval_env_rel = RelativeGoalWrapper(eval_env_smooth)
    eval_env = FixDoneWrapper(eval_env_rel)
    
    successes = 0
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        step = 0
        
        print(f"Running Evaluation Episode {ep + 1}/{num_episodes}...")
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            action = np.array(action, copy=True).flatten()
            
            # Snap gripper for evaluation
            if len(action) > 3:
                action[3] = 1.0 if action[3] > 0.1 else -1.0

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            step += 1
            time.sleep(0.01)
            
            if info.get("is_success", False):
                successes += 1

        print(f"Episode {ep + 1} ended after {step} steps. Success: {info.get('is_success', False)}")
        
    print(f"\nFinal Visual Evaluation Success Rate: {(successes / num_episodes) * 100:.1f}%")
    eval_env.close()


def train_on_operator_data(data_path="operator_data_pick_and_place_spacemouse.pkl"):
    num_cpu = os.cpu_count() or 8  
    
    def make_env(rank, seed=0):
        def _init():
            raw_env = gym.make("FrankaPickAndPlaceSparse-v0", max_episode_steps=250)
            raw_env.action_space.seed(seed + rank)
            flat_env = FlattenGoalEnv(raw_env)
            smooth_env = SmoothFrankaWrapper(flat_env, dt=0.01)
            rel_env = RelativeGoalWrapper(smooth_env)            # <--- Adds relative features
            grip_env = BinaryGripperActionWrapper(rel_env)        # <--- Stops gripper chattering
            fixed_env = FixDoneWrapper(grip_env) 
            return RolloutInfoWrapper(fixed_env)
        return _init

    print(f"Launching {num_cpu} parallel physics simulations...")
    venv = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    with open(data_path, "rb") as f:
        raw_trajectories = pickle.load(f)
        
    # Apply RelativeGoalWrapper transformation to demonstration observations as well!
    # Create a temporary single env to process trajectory observations
    sample_env = gym.make("FrankaPickAndPlaceSparse-v0")
    sample_flat = FlattenGoalEnv(sample_env)
    sample_smooth = SmoothFrankaWrapper(sample_flat)
    rel_transformer = RelativeGoalWrapper(sample_smooth)

    formatted_dataset = []
    for traj in raw_trajectories:
        dummy_rewards = np.zeros(len(traj["acts"]), dtype=np.float32)
        
        # Transform raw observations to include relative features
        transformed_obs = np.array([rel_transformer.observation(o) for o in traj["obs"]], dtype=np.float32)

        formatted_dataset.append(
            types.TrajectoryWithRew(
                obs=transformed_obs,
                acts=traj["acts"],
                infos=None,
                terminal=traj["terminal"],
                rews=dummy_rewards
            )
        )
    sample_env.close()
        
    print(f"Loaded {len(formatted_dataset)} operator runs for training.")

    total_expert_transitions = sum(len(traj.acts) for traj in formatted_dataset)
    safe_batch_size = min(1024, total_expert_transitions)

    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=128,               
        n_steps=2048 // num_cpu,      
        ent_coef=0.02,                
        learning_rate=5e-5,           
        gamma=0.99,
        clip_range=0.05,               
        n_epochs=10,                  
        seed=42,
        device="cpu"                 
    )
    
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=learner.policy,
        demonstrations=formatted_dataset,
        rng=np.random.default_rng(42),
    )
    
    log_collector = CustomLogCollector()
    sb3_logger = getattr(bc_trainer.logger, "default_logger", bc_trainer.logger)
    sb3_logger.output_formats.append(log_collector)
    
    print("\nPre-training Generator via Behavioral Cloning (Offline)...")
    bc_epochs = 100
    bc_trainer.train(n_epochs=bc_epochs)
    
    loss_list = []
    prob_true_act_list = []
    for log in log_collector.logs:
        loss = log.get("bc/loss", log.get("training/loss", log.get("loss", None)))
        prob = log.get("bc/prob_true_act", log.get("training/prob_true_act", log.get("prob_true_act", None)))
        if loss is not None and prob is not None:
            loss_list.append(loss)
            prob_true_act_list.append(prob)
            
    tracked_epochs = list(range(1, len(loss_list) + 1))
    plot_training_metrics(tracked_epochs, prob_true_act_list, loss_list, save_path="bc_pretraining_metrics.png")
    
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=None,
    )

    airl_trainer = AIRL(
        demonstrations=formatted_dataset,
        demo_batch_size=safe_batch_size,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=2,   
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )
    
    print("\nStarting AIRL training (Online)...")
    airl_trainer.train(total_timesteps=100_000)
    
    venv.close()
    
    airl_trainer.gen_algo.save("airl_panda_spacemouse_model")
    print("\nModel saved successfully as 'airl_panda_spacemouse_model.zip'")

    evaluate_and_view_policy(airl_trainer.gen_algo.policy, num_episodes=5)


if __name__ == "__main__":
    train_on_operator_data()