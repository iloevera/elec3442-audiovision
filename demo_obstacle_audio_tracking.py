"""End-to-end demo: RealSense depth obstacle tracking driving spatial audio."""

from __future__ import annotations

import argparse
import signal
import threading
import time

from obstacle_audio_adapter import ObstacleAudioAdapter
from obstacle_models import ObstacleTrackerConfig, ObstacleUpdate
from obstacle_tracker import ObstacleTrackerService
from realsense_driver import D435iDriver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth obstacle tracking audio demo.")
    parser.add_argument(
        "--disable-imu",
        action="store_true",
        help="Force depth-only mode (useful for D435 without IMU).",
    )
    parser.add_argument(
        "--require-imu",
        action="store_true",
        help="Exit if IMU is unavailable instead of falling back to depth-only mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stop_event = threading.Event()

    def _handle_sigint(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    adapter = ObstacleAudioAdapter(warning_ttc_s=2.5)

    last_print_s = 0.0

    def on_update(update: ObstacleUpdate) -> None:
        nonlocal last_print_s
        adapter.on_update(update)

        now_s = time.monotonic()
        if now_s - last_print_s < 0.5:
            return
        last_print_s = now_s

        risk = sorted(update.obstacles, key=lambda obs: obs.collision_score, reverse=True)
        if not risk:
            print("state=degraded obstacles=0")
            return

        top = risk[0]
        ttc_text = "inf" if top.ttc_s == float("inf") else f"{top.ttc_s:.2f}s"
        print(
            "state={state} count={count} top_id={obstacle_id} "
            "xyz=({x:.2f},{y:.2f},{z:.2f})m ttc={ttc} score={score:.2f} conf={conf:.2f}".format(
                state=update.state,
                count=len(update.obstacles),
                obstacle_id=top.obstacle_id,
                x=top.xyz_m[0],
                y=top.xyz_m[1],
                z=top.xyz_m[2],
                ttc=ttc_text,
                score=top.collision_score,
                conf=top.confidence,
            )
        )

    request_imu = not args.disable_imu

    with D435iDriver(enable_imu=request_imu) as driver:
        if args.require_imu and not driver.imu_enabled_runtime:
            raise RuntimeError("IMU is required but unavailable. Remove --require-imu or connect a D435i.")

        imu_comp_enabled = request_imu and driver.imu_enabled_runtime
        config = ObstacleTrackerConfig(
            enable_imu_compensation=imu_comp_enabled,
            warning_ttc_s=2.5,
            urgent_ttc_s=1.0,
        )

        if imu_comp_enabled:
            print("IMU: enabled (gyro compensation active)")
        else:
            print("IMU: disabled/unavailable (running in depth-only mode)")

        service = ObstacleTrackerService(driver=driver, callback=on_update, config=config)
        service.start()

        print("Running obstacle audio tracking demo. Press Ctrl+C to stop.")
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        finally:
            service.stop()
            adapter.close()


if __name__ == "__main__":
    main()
