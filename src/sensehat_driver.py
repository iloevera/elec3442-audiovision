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

    def _worker_loop(self) -> None:
        # We assume the SenseHat is rigidly attached to the camera.
        # Orientation mapping details:
        # Camera: +X Right, +Y Down, +Z Forward.
        # User confirmed top layer faces +Z (same as camera).
        # Standard SenseHat (Leds up): X points right, Y forward, Z up.
        # If TOP is facing +Z:
        # SenseHat Z -> Camera +Z
        # SenseHat X -> Camera +X
        # SenseHat Y -> Camera -Y (Up)
        
        period = 1.0 / self._update_rate_hz
        
        while not self._stop_event.is_set():
            start_time = time.monotonic()
            
            try:
                # Raw accelerometer values (G-force acting on the sensor)
                accel = self._sense.get_accelerometer_raw()
                # SenseHat returns acceleration. When still and level (LEDs up), z correlates to ~1.0
                # Mapping:
                # Camera X = SenseHat X
                # Camera Y = -SenseHat Y
                # Camera Z = SenseHat Z
                
                # We want a GRAVITY vector (pointing down).
                # If the sensor measures acceleration 'a', then gravity 'g' is approximately '-a'
                # when the sensor is stationary.
                sx, sy, sz = accel['x'], accel['y'], accel['z']
                
                # Map to camera frame:
                cx = sx
                cy = -sy
                cz = sz
                
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
