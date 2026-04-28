import argparse
import sys
import os

# Add project root to path so demos can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="AudioVision Demo Runner")
    parser.add_argument("--audio", action="store_true", help="Run the audio rotating tones demo")
    parser.add_argument("--navigation", action="store_true", help="Run the navigation assist demo")
    parser.add_argument("--realsense", action="store_true", help="Run the RealSense preview demo")
    parser.add_argument("--imu", action="store_true", help="Run the SenseHat IMU demo")

    args = parser.parse_args()

    # Check if at least one flag is provided
    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.audio:
        print("Launching Audio Rotating Tones Demo...")
        from demo.demo_audio_rotating_tones import run_demo
        run_demo()

    elif args.navigation:
        print("Launching Navigation Assist Demo...")
        from demo.demo_navigation_assist import run_demo
        run_demo()

    elif args.realsense:
        print("Launching RealSense Preview Demo...")
        from demo.demo_realsense_preview import run_preview
        run_preview()

    elif args.imu:
        print("Launching SenseHat IMU Demo...")
        from demo.demo_imu import run_imu_demo
        run_imu_demo()

if __name__ == "__main__":
    main()
