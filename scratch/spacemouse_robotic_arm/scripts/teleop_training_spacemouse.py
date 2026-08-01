import time
import threading
import numpy as np
import pyspacemouse # library to read raw values from spacemouse

from teleop_training_common import InputSource, prompt_for_task, run_operator_session

# method to implement deadzones for movement, so if there is drift then it does not impede ease of teleop
def smooth_deadzone(x, deadzone=0.08):
    if abs(x) < deadzone:
        return 0.0
    return np.sign(x) * (abs(x) - deadzone) / (1.0 - deadzone)

class SpaceMouseThread:
    # thread running in background for polling spacemouse input readings
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

    def get_state(self):
        return self.latest_state

    def stop(self):
        self.running = False

class SpaceMouseSource(InputSource):
    GLITCH_FILTER_ALPHA = 0.55
    DEADZONE = 0.12

    def __init__(self):
        self._cm = pyspacemouse.open() # get context manager object for spacemouse to then access it
        device = self._cm.__enter__() # get device handle using context manager
        self.sm_thread = SpaceMouseThread(device)
        self.filtered_raw = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.gripper_val = 1.0 # gripper value set to open by default (since it
        # should be for pick-and-place. for push, it is not used at all)
        self._abort = False

    def get_action(self, act_dim):
        state = self.sm_thread.get_state()
        while state is None:
            time.sleep(0.002)
            state = self.sm_thread.get_state()

        left, right = state.buttons # buttons on spacemouse toggle gripper
        if left and right: # left and right buttons pressed simultaneously abort episode
            self._abort = True
        elif left:
            self.gripper_val = -1.0 # open gripper
        elif right:
            self.gripper_val = 1.0 # close gripper

        # low-pass filter raw SpaceMouse values to cancel snapback glitch
        a = self.GLITCH_FILTER_ALPHA
        self.filtered_raw["x"] = (1 - a) * self.filtered_raw["x"] + a * state.x
        self.filtered_raw["y"] = (1 - a) * self.filtered_raw["y"] + a * state.y
        self.filtered_raw["z"] = (1 - a) * self.filtered_raw["z"] + a * state.z

        # raw normalized SpaceMouse action vector
        dx = smooth_deadzone(self.filtered_raw["x"], self.DEADZONE)
        dy = smooth_deadzone(self.filtered_raw["y"], self.DEADZONE)
        dz = smooth_deadzone(self.filtered_raw["z"], self.DEADZONE)

        # only append gripper state to command vector when task has one (ie pick-and-place, not for push)
        if act_dim == 4:
            return np.array([dx, dy, dz, self.gripper_val], dtype=np.float32)
        return np.array([dx, dy, dz], dtype=np.float32)
    
    def should_abort(self):
        return self._abort
 
    def reset_state(self):
        self.filtered_raw = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.gripper_val = 1.0
        self._abort = False
 
    def close(self):
        self.sm_thread.stop()
        self._cm.__exit__(None, None, None)
 
 
if __name__ == "__main__":
    task = prompt_for_task() # read which task user wants to teleop for
    if task is None:
        raise SystemExit("Invalid selection -- enter 1 (push) or 2 (pick_and_place).")

    # set operator data to be saved as pickle files w/ corresponding names
    output_files = {
        "push": "operator_data_push_spacemouse.pkl",
        "pick_and_place": "operator_data_pick_and_place_spacemouse.pkl",
    }

    # define spacemouse as input method and run operator session for selected task
    source = SpaceMouseSource()
    run_operator_session(
        input_source=source,
        task_name=task,
        total_episodes=5,
        output_file=output_files[task],
    )