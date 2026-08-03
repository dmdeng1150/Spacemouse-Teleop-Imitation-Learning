import gymnasium as gym
import numpy as np


class FlattenGoalEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = env.observation_space.spaces
        low = np.concatenate([spaces['observation'].low, spaces['achieved_goal'].low, spaces['desired_goal'].low])
        high = np.concatenate([spaces['observation'].high, spaces['achieved_goal'].high, spaces['desired_goal'].high])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])


class SmoothFrankaWrapper(gym.ActionWrapper):
    def __init__(self, env, sensitivity=0.15, max_accel=1.5, dt=1.0/100.0):
        super().__init__(env)
        self.sensitivity = sensitivity
        self.max_accel = max_accel
        self.dt = dt
        self.current_action_xyz = np.zeros(3, dtype=np.float32)

    def reset(self, **kwargs):
        self.current_action_xyz = np.zeros(3, dtype=np.float32)
        return self.env.reset(**kwargs)

    def action(self, raw_action):
        desired_xyz = raw_action[:3] * self.sensitivity
        action_change = desired_xyz - self.current_action_xyz
        max_change = self.max_accel * self.dt
        action_change = np.clip(action_change, -max_change, max_change)
        
        self.current_action_xyz += action_change
        xyz_action = np.clip(self.current_action_xyz, -1.0, 1.0)

        if len(raw_action) == 4:
            gripper = raw_action[3]
            processed_action = np.array([
                xyz_action[1],
                xyz_action[0],
                xyz_action[2],
                gripper
            ], dtype=np.float32)
        else:
            processed_action = np.array([
                xyz_action[1],
                xyz_action[0],
                xyz_action[2]
            ], dtype=np.float32)

        return processed_action

    def step(self, action):
        return self.env.step(self.action(action))


class SmoothXYZActionWrapper(gym.ActionWrapper):
    def __init__(self, env, alpha=0.65):
        super().__init__(env)
        self.alpha = alpha 
        self.last_xyz = np.zeros(3, dtype=np.float32)

    def reset(self, **kwargs):
        self.last_xyz = np.zeros(3, dtype=np.float32)
        return super().reset(**kwargs)

    def action(self, action):
        action = np.array(action, copy=True)
        if len(action) >= 3:
            smoothed_xyz = (1.0 - self.alpha) * action[:3] + self.alpha * self.last_xyz
            self.last_xyz = smoothed_xyz
            action[:3] = smoothed_xyz
        return action


class BinaryGripperActionWrapper(gym.ActionWrapper):
    def __init__(self, env, close_thresh=0.2, open_thresh=0.6):
        super().__init__(env)
        self.close_thresh = close_thresh
        self.open_thresh = open_thresh
        self.state = 1.0

    def reset(self, **kwargs):
        self.state = 1.0
        return super().reset(**kwargs)

    def action(self, action):
        action = np.array(action, copy=True)
        if len(action) > 3:
            raw_g = action[3]
            if raw_g < self.close_thresh:
                self.state = -1.0
            elif raw_g > self.open_thresh:
                self.state = 1.0
            action[3] = self.state
        return action


class RelativeGoalWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_clearance=0.05, transition_dist=0.08, ee_z_offset=0.058):
        super().__init__(env)
        self.max_clearance = max_clearance        
        self.transition_dist = transition_dist    
        self.ee_z_offset = ee_z_offset  
        
        self.base_obs_dim = env.observation_space.shape[0]
        old_shape = self.base_obs_dim
        new_shape = old_shape + 16
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )

        self.last_dist_ee_block = 0.0
        self.last_dist_block_goal = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        info["dist_ee_block"] = float(self.last_dist_ee_block)
        info["dist_block_goal"] = float(self.last_dist_block_goal)
        return obs, reward, terminated, truncated, info

    def observation(self, obs, override_gripper=None):
        raw_ee_pos = obs[0:3]          
        
        block_pos = obs[self.base_obs_dim - 6 : self.base_obs_dim - 3]         
        goal_pos = obs[self.base_obs_dim - 3 : self.base_obs_dim]          

        grasp_pos = raw_ee_pos.copy()
        grasp_pos[2] -= self.ee_z_offset

        dy_rel = block_pos[1] - grasp_pos[1]
        dx_rel = block_pos[0] - grasp_pos[0]
        dz_rel = block_pos[2] - grasp_pos[2]

        rel_xy = np.array([dy_rel, dx_rel], dtype=np.float32)
        dist_xy_val = np.linalg.norm(rel_xy)
        dir_xy = rel_xy / (dist_xy_val + 1e-6)
        dist_xy = np.array([dist_xy_val], dtype=np.float32)

        clearance_ratio = min(1.0, dist_xy_val / self.transition_dist)
        funnel_target_z = block_pos[2] + self.max_clearance * clearance_ratio
        
        rel_ee_to_funnel = np.array([dy_rel, dx_rel, funnel_target_z - grasp_pos[2]], dtype=np.float32)
        rel_ee_to_block = np.array([dy_rel, dx_rel, dz_rel], dtype=np.float32)

        dy_goal = goal_pos[1] - block_pos[1]
        dx_goal = goal_pos[0] - block_pos[0]
        dz_goal = goal_pos[2] - block_pos[2]
        rel_block_to_goal = np.array([dy_goal, dx_goal, dz_goal], dtype=np.float32)

        smooth_alignment = np.array([np.exp(-dist_xy_val / 0.03)], dtype=np.float32)

        dist_block_goal_val = np.linalg.norm(rel_block_to_goal)
        dist_block_goal = np.array([dist_block_goal_val], dtype=np.float32)
        block_height = np.array([block_pos[2]], dtype=np.float32)

        # INLINE GRIPPER STATE EXTRACTION (Guarantees zero AttributeError)
        if override_gripper is not None:
            gripper_state = np.array([override_gripper], dtype=np.float32)
        else:
            g_val = 1.0
            curr = self.env
            while curr is not None:
                if isinstance(curr, BinaryGripperActionWrapper):
                    g_val = float(curr.state)
                    break
                if hasattr(curr, 'env'):
                    curr = curr.env
                else:
                    break
            gripper_state = np.array([g_val], dtype=np.float32)

        self.last_dist_ee_block = np.linalg.norm(rel_ee_to_block)
        self.last_dist_block_goal = dist_block_goal_val

        return np.concatenate([
            obs, 
            rel_ee_to_funnel,
            rel_ee_to_block, 
            rel_block_to_goal, 
            dir_xy,
            dist_xy,
            smooth_alignment,
            dist_block_goal, 
            block_height,
            gripper_state
        ]).astype(np.float32)


class FixDoneWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward), bool(terminated), bool(truncated), info