"""Assistive navigation demo — pi_normal mode, no preview.

Controls (joystick outputs keyboard events):
  Up arrow    — volume up   (plays 880 Hz feedback tone)
  Down arrow  — volume down (plays 220 Hz feedback tone)
  Enter       — toggle navigation audio on/off
                  on  → 660 Hz tone
                  off → 330 Hz tone
  Q / Escape  — quit

LED matrix:
  Default : risk grid (green = safe, red = obstacle)
  3 s after any joystick event : settings HUD
    Left 4 cols  — volume bar (bottom-up, green → red)
    Right 4 cols — nav status (green = on, red = off)
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty
from contextlib import suppress
from typing import TYPE_CHECKING

from src.navigation_processing import NavigationProcessor, NavigationProcessorConfig
from src.realsense_driver import D435iDriver
from src.sensehat_driver import SenseHatDriver
from src.audio_spatial_tone import SpatialTone

if TYPE_CHECKING:
    from src.navigation_audio import NavigationAudioController


# ── Pi-normal hard-coded config ───────────────────────────────────────────────

DEPTH_SIZE = (424, 240)
COLOR_SIZE = (424, 240)
FPS = 30
ALIGN_DEPTH = False

REALSENSE_PROFILES = (
    {
        "name": "pi_normal",
        "depth_size": DEPTH_SIZE,
        "color_size": COLOR_SIZE,
        "depth_fps": FPS,
        "color_fps": FPS,
        "align_depth_to_color": ALIGN_DEPTH,
    },
    {
        # Matches the settings used by the working RealSense preview demo.
        "name": "default_fallback",
        "depth_size": (640, 480),
        "color_size": (640, 480),
        "depth_fps": 30,
        "color_fps": 30,
        "align_depth_to_color": True,
    },
)

PROCESSOR_CONFIG = NavigationProcessorConfig(
    downsample_step=3,
    ransac_iterations=30,
    min_plane_inliers=180,
    ground_plane_refit_interval_frames=3,
)

# ── UI constants ──────────────────────────────────────────────────────────────

SETTINGS_HUD_DURATION_S = 3.0
LED_UPDATE_INTERVAL_S = 0.2   # ≈ 5 Hz LED refresh


# ── Non-blocking keyboard reader ──────────────────────────────────────────────

class KeyboardInput:
    """Non-blocking ANSI escape-sequence keyboard reader using cbreak stdin.

    Parses arrow keys (output by the SenseHat joystick) and Enter / Q.
    """

    def __init__(self) -> None:
        try:
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._available = True
        except (termios.error, ValueError):
            self._available = False

    def close(self) -> None:
        if self._available:
            with suppress(Exception):
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> str | None:
        """Return 'up', 'down', 'left', 'right', 'enter', 'quit', or None."""
        if not self._available:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return None
        raw = os.read(self._fd, 8)
        if raw == b"\x1b[A":
            return "up"
        if raw == b"\x1b[B":
            return "down"
        if raw == b"\x1b[C":
            return "right"
        if raw == b"\x1b[D":
            return "left"
        if raw in (b"\r", b"\n"):
            return "enter"
        if raw in (b"q", b"Q", b"\x1b"):
            return "quit"
        return None


# ── UI feedback tones ─────────────────────────────────────────────────────────

class UITonePlayer:
    """Short feedback tones via a dedicated always-running SpatialTone voice."""

    VOLUME_UP_HZ:   float = 880.0
    VOLUME_DOWN_HZ: float = 220.0
    NAV_ON_HZ:      float = 660.0
    NAV_OFF_HZ:     float = 330.0

    def __init__(self) -> None:
        self._tone = SpatialTone(initial_pitch_hz=440.0, initial_volume=0.0)
        self._tone.start()
        self._timer: threading.Timer | None = None

    def play(self, pitch_hz: float, duration_s: float = 0.35) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._tone.set_pitch(pitch_hz)
        self._tone.set_volume(0.6)
        self._timer = threading.Timer(duration_s, lambda: self._tone.set_volume(0.0))
        self._timer.daemon = True
        self._timer.start()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        with suppress(Exception):
            self._tone.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    processor = NavigationProcessor(config=PROCESSOR_CONFIG)
    imu = SenseHatDriver()
    imu.start()

    kb = KeyboardInput()
    ui_tone = UITonePlayer()

    volume_level: int = 5        # 0 – 10
    nav_enabled: bool = True
    last_joystick_time: float = float("-inf")
    last_led_update: float = float("-inf")
    audio: NavigationAudioController | None = None

    print("Navigation running.")
    print("  Up/Down arrow : volume  |  Enter : toggle nav  |  Q : quit")

    try:
        driver: D435iDriver | None = None
        start_error: Exception | None = None
        active_profile_name = ""

        for profile in REALSENSE_PROFILES:
            candidate = D435iDriver(
                depth_size=profile["depth_size"],
                color_size=profile["color_size"],
                depth_fps=profile["depth_fps"],
                color_fps=profile["color_fps"],
                align_depth_to_color=profile["align_depth_to_color"],
            )
            try:
                candidate.start()
            except Exception as exc:
                start_error = exc
                print(
                    "RealSense start failed for profile"
                    f" '{profile['name']}'"
                    f" ({profile['depth_size'][0]}x{profile['depth_size'][1]} depth,"
                    f" {profile['color_size'][0]}x{profile['color_size'][1]} color): {exc}"
                )
                continue

            driver = candidate
            active_profile_name = str(profile["name"])
            print(f"RealSense started using profile: {active_profile_name}")
            break

        if driver is None:
            if start_error is None:
                raise RuntimeError("Failed to start RealSense for unknown reasons")
            raise RuntimeError(
                "Unable to start RealSense with any configured profile. "
                "Check USB connection/power and ensure no other app is using the camera."
            ) from start_error

        try:
            while True:
                # ── Keyboard / joystick ────────────────────────────────────
                key = kb.read_key()
                if key == "quit":
                    break
                elif key == "up":
                    volume_level = min(10, volume_level + 1)
                    last_joystick_time = time.monotonic()
                    ui_tone.play(UITonePlayer.VOLUME_UP_HZ, 0.35)
                    print(f"  Volume: {volume_level}/10")
                elif key == "down":
                    volume_level = max(0, volume_level - 1)
                    last_joystick_time = time.monotonic()
                    ui_tone.play(UITonePlayer.VOLUME_DOWN_HZ, 0.35)
                    print(f"  Volume: {volume_level}/10")
                elif key == "enter":
                    nav_enabled = not nav_enabled
                    last_joystick_time = time.monotonic()
                    hz = UITonePlayer.NAV_ON_HZ if nav_enabled else UITonePlayer.NAV_OFF_HZ
                    ui_tone.play(hz, 0.5)
                    print(f"  Navigation: {'ON' if nav_enabled else 'OFF'}")

                # ── Capture ────────────────────────────────────────────────
                bundle = driver.wait_for_bundle(timeout_s=1.0)
                if bundle is None:
                    if driver.last_error is not None:
                        raise RuntimeError("Capture thread stopped after error") from driver.last_error
                    continue

                gravity_unit = imu.get_gravity_unit()

                # ── Lazy audio init after first frame ──────────────────────
                if audio is None:
                    from src.navigation_audio import NavigationAudioController
                    audio = NavigationAudioController(column_count=processor.config.cols)
                    audio.start()

                # ── Process ────────────────────────────────────────────────
                analysis = processor.process_bundle(bundle, gravity_unit=gravity_unit)

                # ── Audio ──────────────────────────────────────────────────
                if nav_enabled:
                    audio.apply(analysis.column_states, now_s=time.monotonic())
                    scale = volume_level / 10.0
                    for voice in audio._voices:
                        voice.set_volume(voice.volume * scale)
                else:
                    for voice in audio._voices:
                        voice.set_volume(0.0)

                # ── LED (throttled to ~5 Hz) ───────────────────────────────
                now = time.monotonic()
                if now - last_led_update >= LED_UPDATE_INTERVAL_S:
                    last_led_update = now
                    if now - last_joystick_time < SETTINGS_HUD_DURATION_S:
                        imu.show_settings_hud(volume_level, nav_enabled)
                    else:
                        imu.show_risk_grid(analysis.risk_grid)
        finally:
            driver.stop()

    finally:
        kb.close()
        ui_tone.stop()
        imu.clear_leds()
        imu.stop()
        if audio is not None:
            audio.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
