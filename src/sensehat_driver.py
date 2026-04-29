from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

try:
    from sense_hat import SenseHat
except ImportError:
    SenseHat = None


class SenseHatDriver:
    """Threaded driver for Raspberry Pi SenseHat orientation and gravity tracking."""

    def __init__(self, update_rate_hz: float = 30.0) -> None:
        if SenseHat is None:
            raise ImportError(
                "The 'sense-hat' library is not installed. "
                "Please run 'pip install sense-hat' on a Raspberry Pi."
            )

        self._sense = SenseHat()
        self._update_rate_hz = float(update_rate_hz)
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._latest_gravity: Optional[np.ndarray] = None
        self._latest_orientation: Optional[dict[str, float]] = None
        self._running = False

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()

        self._thread = threading.Thread(target=self._worker_loop, name="SenseHatDriver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_gravity_unit(self) -> Optional[np.ndarray]:
        """Returns the gravity vector in the camera frame (+X right, +Y down, +Z forward)."""
        with self._lock:
            return self._latest_gravity

    def get_orientation(self) -> Optional[dict[str, float]]:
        """Returns the orientation (pitch, roll, yaw) in degrees."""
        with self._lock:
            return self._latest_orientation

    # ── LED matrix ────────────────────────────────────────────────────────────

    def show_risk_grid(self, risk_grid: np.ndarray) -> None:
        """Map a (rows, cols) float risk grid onto the 8×8 LED matrix.

        Risk 0.0 → green, Risk 1.0 → red. Nearest-neighbour mapping.
        Horizontally mirrored for user perspective.
        Call from the main thread only.
        """
        grid_rows, grid_cols = risk_grid.shape
        pixels = []
        for led_r in range(8):
            gr = led_r * grid_rows // 8
            for led_c in range(8):
                # Map column mirrored (8 - 1 - led_c)
                mirrored_c = 7 - led_c
                gc = mirrored_c * grid_cols // 8
                risk = float(np.clip(risk_grid[gr, gc], 0.0, 1.0))
                pixels.append([int(risk * 100), int((1.0 - risk) * 100), 0])
        self._sense.set_pixels(pixels)

    def show_settings_hud(self, volume_level: int, nav_enabled: bool) -> None:
        """Display a settings HUD on the 8×8 LED matrix.

        Left 4 columns: navigation status (solid green = on, solid red = off).
        Right 4 columns: volume bar (fills bottom-up, green→red, 0–10 scale).
        (Note: mirrored from original: original had volume left, nav right).
        Call from the main thread only.
        """
        filled_rows = round(int(np.clip(volume_level, 0, 10)) * 8 / 10)
        nav_color = [0, 100, 0] if nav_enabled else [90, 0, 0]
        pixels = []
        for led_r in range(8):
            idx_from_bottom = 7 - led_r  # 0 at bottom row, 7 at top row
            row_lit = idx_from_bottom < filled_rows
            if row_lit:
                t = idx_from_bottom / 7.0  # 0=bottom(green) → 1=top(red)
                vol_color = [int(t * 100), int((1.0 - t) * 100), 0]
            else:
                vol_color = [7, 7, 7]
            for led_c in range(8):
                # led_c 0-3: Left (now Nav)
                # led_c 4-7: Right (now Vol)
                pixels.append(list(nav_color) if led_c < 4 else list(vol_color))
        self._sense.set_pixels(pixels)

    def clear_leds(self) -> None:
        """Clear the LED matrix. Call from the main thread only."""
        self._sense.clear()

    def _worker_loop(self) -> None:
        # The SenseHat is rigidly attached to the camera.
        # Orientation mapping based on calibration:
        # Camera: +X Right, +Y Down, +Z Forward.
        # Calibration results show:
        # Camera +X (Right)   -> SenseHat -X
        # Camera +Y (Down)    -> SenseHat +Y
        # Camera +Z (Forward) -> SenseHat -Z
        
        period = 1.0 / self._update_rate_hz
        
        while not self._stop_event.is_set():
            start_time = time.monotonic()
            
            try:
                # Raw accelerometer values (G-force acting on the sensor)
                accel = self._sense.get_accelerometer_raw()
                
                sx, sy, sz = accel['x'], accel['y'], accel['z']
                
                # Map to camera frame based on calibration:
                cx = -sx
                cy = sy
                cz = -sz
                
                # Gravity vector (opposite of acceleration felt by sensor)
                gv = -np.array([cx, cy, cz], dtype=np.float32)
                norm = np.linalg.norm(gv)
                if norm > 0.01:
                    gravity_unit = gv / norm
                else:
                    gravity_unit = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                # Orientation in degrees
                orient = self._sense.get_orientation()
                
                with self._lock:
                    self._latest_gravity = gravity_unit
                    self._latest_orientation = orient
                    
            except Exception as e:
                print(f"SenseHatDriver error: {e}")

            elapsed = time.monotonic() - start_time
            wait = max(0, period - elapsed)
            if wait > 0:
                time.sleep(wait)

if __name__ == "__main__":
    driver = SenseHatDriver()
    driver.start()
    try:
        while True:
            print(f"Gravity: {driver.get_gravity_unit()}, Orientation: {driver.get_orientation()}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        driver.stop()
