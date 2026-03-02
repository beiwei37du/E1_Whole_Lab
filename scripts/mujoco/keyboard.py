import numpy as np
from pynput import keyboard


class KeyboardCommand:
    def __init__(
        self,
        vx=0.0,
        vy=0.0,
        dyaw=0.0,
        vx_increment=0.1,
        vy_increment=0.1,
        dyaw_increment=0.1,
        min_vx=-0.8,
        max_vx=2.5,
        min_vy=-0.8,
        max_vy=0.8,
        min_dyaw=-1.0,
        max_dyaw=1.0,
    ):
        self.vx = vx
        self.vy = vy
        self.dyaw = dyaw
        self.vx_increment = vx_increment
        self.vy_increment = vy_increment
        self.dyaw_increment = dyaw_increment
        self.min_vx = min_vx
        self.max_vx = max_vx
        self.min_vy = min_vy
        self.max_vy = max_vy
        self.min_dyaw = min_dyaw
        self.max_dyaw = max_dyaw
        self.reset_requested = False
        self._listener = None

    def _print_status(self):
        print(f"vx: {self.vx:.2f}, vy: {self.vy:.2f}, dyaw: {self.dyaw:.2f}")

    def update_vx(self, delta):
        self.vx = float(np.clip(self.vx + delta, self.min_vx, self.max_vx))
        self._print_status()

    def update_vy(self, delta):
        self.vy = float(np.clip(self.vy + delta, self.min_vy, self.max_vy))
        self._print_status()

    def update_dyaw(self, delta):
        self.dyaw = float(np.clip(self.dyaw + delta, self.min_dyaw, self.max_dyaw))
        self._print_status()

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0
        self.dyaw = 0.0
        print(f"Velocities reset: vx: {self.vx:.2f}, vy: {self.vy:.2f}, dyaw: {self.dyaw:.2f}")

    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char is not None:
                c = key.char.lower()
                if c == '8':
                    self.update_vx(self.vx_increment)
                elif c == '2':
                    self.update_vx(-self.vx_increment)
                elif c == '4':
                    self.update_vy(self.vy_increment)
                elif c == '6':
                    self.update_vy(-self.vy_increment)
                elif c == '7':
                    self.update_dyaw(self.dyaw_increment)
                elif c == '9':
                    self.update_dyaw(-self.dyaw_increment)
                elif c == '0':
                    self.reset_requested = True
                    print('Reset requested (0 key pressed)')
        except AttributeError:
            pass

    def on_release(self, key):
        pass

    def start(self):
        self._listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self._listener.start()
        return self._listener

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


def print_keyboard_instructions():
    print("=" * 60)
    print("Keyboard control instructions (numpad style):")
    print("  8: forward (vx +)")
    print("  2: backward (vx -)")
    print("  4: left (vy +)")
    print("  6: right (vy -)")
    print("  7: turn left (dyaw +)")
    print("  9: turn right (dyaw -)")
    print("  0: reset command + robot state")
    print("=" * 60)
