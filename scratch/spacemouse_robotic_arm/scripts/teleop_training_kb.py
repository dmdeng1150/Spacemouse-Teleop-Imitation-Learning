# main teleop training file for kb input
import numpy as np
from pynput import keyboard # library to read kb input

from teleop_training_common import InputSource, prompt_for_task, run_operator_session

def disable_mujoco_key_callbacks(env):
    # method to disable default key bindings in mujoco so we can use them for kb control
    try:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "mujoco_renderer"):
            viewer = unwrapped.mujoco_renderer._get_viewer("human")
            if hasattr(viewer, "window") and viewer.window is not None:
                import glfw
                glfw.set_key_callback(viewer.window, lambda window, key, scancode, action, mods: None)
                print("[INFO] Native MuJoCo key bindings disabled.")
    except Exception as e:
        print(f"[WARNING] Could not disable MuJoCo key callbacks: {e}")

class KeyboardSource(InputSource):
    def __init__(self, move_speed=0.4, ramp_rate=0.15):
        self.move_speed = move_speed
        self.ramp_rate = ramp_rate

        self.vel = np.zeros(3, dtype=np.float32)
        self.is_closed = False
        self.manual_abort = False

        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    # method to detect key and perform corresponding motion
    def _on_press(self, key):
        k_str = None
        if hasattr(key, 'char') and key.char is not None:
            k_str = key.char.lower()
        elif key == keyboard.Key.space:
            k_str = 'space'

        if k_str:
            self.pressed_keys.add(k_str)

            if k_str == 'r':
                self.manual_abort = True
                print("Episode aborted by operator.")
            elif k_str == 'space':
                self.is_closed = not self.is_closed

    # method to detect key release
    def _on_release(self, key):
        k_str = None
        if hasattr(key, 'char') and key.char is not None:
            k_str = key.char.lower()
        elif key == keyboard.Key.space:
            k_str = 'space'

        if k_str:
            self.pressed_keys.discard(k_str)

    # method to calculate smoothed action vector in 3d or 4d (depending on push or pick/place task)
    def get_action(self, act_dim):
        target_vel = np.zeros(3, dtype=np.float32)
        if 'a' in self.pressed_keys: target_vel[0] -= self.move_speed # move left (-x)
        if 'd' in self.pressed_keys: target_vel[0] += self.move_speed # move right (+x)
        if 'w' in self.pressed_keys: target_vel[1] += self.move_speed # move forward (+y)
        if 's' in self.pressed_keys: target_vel[1] -= self.move_speed # move backward (-y)
        if 'e' in self.pressed_keys: target_vel[2] += self.move_speed # move up (+z)
        if 'q' in self.pressed_keys: target_vel[2] -= self.move_speed # move down (-z) 

        # smooth velocity ramping
        self.vel = (1.0 - self.ramp_rate) * self.vel + self.ramp_rate * target_vel
        # deadzone filter to stop drift
        if np.all(target_vel == 0) and np.linalg.norm(self.vel) < 0.01:
            self.vel[:] = 0.0

        # only emit gripper channel when task has one (ie pick-and-place, not push)
        if act_dim == 4:
            if self.is_closed:
                gripper_state = 1.0
            else:
                gripper_state = -1.0
            return np.append(self.vel, gripper_state).astype(np.float32) # if the task requires the gripper, then append it to the velocity vector
        return self.vel.copy()

    def should_abort(self):
        return self.manual_abort

    def reset_state(self):
        self.vel = np.zeros(3, dtype=np.float32)
        self.is_closed = False
        self.manual_abort = False

    def on_env_ready(self, env):
        disable_mujoco_key_callbacks(env)

    def close(self):
        self.listener.stop()

if __name__ == "__main__":
    task = prompt_for_task() # prompt user to select which task they want to teleop for
    if task is None:
        raise SystemExit("Invalid selection -- enter 1 (push) or 2 (pick_and_place).")

    # define pickle files where we will save training data
    output_files = {
        "push": "operator_data_push_kb.pkl",
        "pick_and_place": "operator_data_pick_and_place_kb.pkl",
    }

    # set kb as input method and run operator session for selected task
    source = KeyboardSource(move_speed=0.4, ramp_rate=0.15)
    run_operator_session(
        input_source=source,
        task_name=task,
        total_episodes=5,
        output_file=output_files[task],
    )