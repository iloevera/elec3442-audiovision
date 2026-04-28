from __future__ import annotations

import sys
import os
import time

try:
    from sense_hat import SenseHat
except ImportError:
    print("SenseHat library not found. Please run this on a Raspberry Pi with SenseHat installed.")
    sys.exit(1)

def wait_for_enter(prompt):
    input(f"\n{prompt}\nPress ENTER to capture...")

def capture_acceleration(sense):
    # Average some readings for stability
    samples = 10
    ax, ay, az = 0.0, 0.0, 0.0
    for _ in range(samples):
        accel = sense.get_accelerometer_raw()
        ax += accel['x']
        ay += accel['y']
        az += accel['z']
        time.sleep(0.01)
    return ax/samples, ay/samples, az/samples

def main():
    sense = SenseHat()
    
    print("=== IMU Orientation Calibration Utility ===")
    print("We will determine how the SenseHat axes map to the Camera axes.")
    print("Camera Axes convention: +X Right, +Y Down, +Z Forward.")
    
    # 1. Forward (+Z)
    wait_for_enter("1. Tilt the CAMERA to face straight DOWN (Camera +Z pointing to floor).")
    f_ax, f_ay, f_az = capture_acceleration(sense)
    print(f"Captured values: x={f_ax:.3f}, y={f_ay:.3f}, z={f_az:.3f}")
    
    # 2. Right (+X)
    wait_for_enter("2. Tilt the CAMERA so the RIGHT side is facing the floor (Camera +X pointing to floor).")
    r_ax, r_ay, r_az = capture_acceleration(sense)
    print(f"Captured values: x={r_ax:.3f}, y={r_ay:.3f}, z={r_az:.3f}")
    
    # 3. Down (+Y)
    wait_for_enter("3. Keep the CAMERA LEVEL (Normal upright position, Camera +Y pointing to floor).")
    d_ax, d_ay, d_az = capture_acceleration(sense)
    print(f"Captured values: x={d_ax:.3f}, y={d_ay:.3f}, z={d_az:.3f}")

    print("\n=== Analysis ===")
    
    def get_max_axis(vals):
        axes = ['x', 'y', 'z']
        max_idx = 0
        max_val = abs(vals[0])
        for i in range(1, 3):
            if abs(vals[i]) > max_val:
                max_val = abs(vals[i])
                max_idx = i
        
        sign = "+" if vals[max_idx] > 0 else "-"
        return f"{sign}{axes[max_idx]}"

    cam_z_imu = get_max_axis((f_ax, f_ay, f_az))
    cam_x_imu = get_max_axis((r_ax, r_ay, r_az))
    cam_y_imu = get_max_axis((d_ax, d_ay, d_az))

    print(f"Camera +X (Right)   maps to SenseHat {cam_x_imu}")
    print(f"Camera +Y (Down)    maps to SenseHat {cam_y_imu}")
    print(f"Camera +Z (Forward) maps to SenseHat {cam_z_imu}")
    
    print("\nExpected code mapping in sensehat_driver.py:")
    print(f"sx, sy, sz = accel['x'], accel['y'], accel['z']")
    
    def axis_to_expr(axis_map):
        sign = axis_map[0]
        comp = axis_map[1]
        return f"{'' if sign == '+' else '-'}s{comp}"

    print(f"cx = {axis_to_expr(cam_x_imu)}")
    print(f"cy = {axis_to_expr(cam_y_imu)}")
    print(f"cz = {axis_to_expr(cam_z_imu)}")

if __name__ == "__main__":
    main()
