from __future__ import annotations

import sys
import os
import time

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensehat_driver import SenseHatDriver

def run_imu_demo():
    print("Starting SenseHat IMU Demo...")
    print("Press Ctrl+C to exit.")
    
    try:
        driver = SenseHatDriver()
        driver.start()
        
        while True:
            gravity = driver.get_gravity_unit()
            orientation = driver.get_orientation()
            
            if gravity is not None and orientation is not None:
                g_str = f"[{gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f}]"
                o_str = f"P: {orientation['pitch']:.1f}, R: {orientation['roll']:.1f}, Y: {orientation['yaw']:.1f}"
                print(f"\rGravity: {g_str} | Orientation: {o_str}", end="", flush=True)
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping IMU Demo...")
    finally:
        driver.stop()

if __name__ == "__main__":
    run_imu_demo()
