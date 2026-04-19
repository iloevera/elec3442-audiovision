# RealSense Test

Small Python project for:
- Intel RealSense D435i depth/color capture with IMU support
- simple spatial audio tone generation and demo playback
- early assistive-navigation pipeline with IMU-guided ground subtraction, percentile-depth grid statistics, and TTC-driven audio cues

## Files

- `realsense_driver.py`: reusable `D435iDriver` class for depth/color/IMU capture
- `demo_realsense_preview.py`: OpenCV preview demo for color, depth, and IMU telemetry
- `audio_spatial_tone.py`: continuously playing tone class with live pitch, volume, and azimuth control
- `audio_mixer.py`: shared audio output mixer for multiple simultaneous tones
- `demo_audio_rotating_tones.py`: rotating stereo tone demo
- `navigation_processing.py`: depth+IMU obstacle analysis, configurable percentile-depth grid statistics, and TTC estimation
- `navigation_audio.py`: column-based pulsed audio controller for navigation cues
- `demo_navigation_assist.py`: end-to-end depth/IMU/audio debug demo with OpenCV overlay

## Requirements

- Windows with Python 3.10 or 3.11 recommended
- Intel RealSense D435i connected over USB 3 if you want camera access
- The default driver configuration expects a D435i with working IMU streams
- Speakers or headphones for the audio demo

Python packages are listed in `requirements.txt`.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Notes About `pyrealsense2`

`pyrealsense2` depends on Intel RealSense support on the target machine. If `pip install -r requirements.txt` fails for `pyrealsense2`, install the Intel RealSense SDK / runtime for your system first, then retry the pip install.

Typical things that matter for successful camera startup:
- use a direct USB 3 port rather than a low-bandwidth hub
- use a D435i if you expect IMU data
- if IMU is enabled and cannot be started, driver startup fails immediately

## Run

### RealSense preview

```powershell
python .\demo_realsense_preview.py
```

This opens a preview window with color and depth. Press `q` or `Esc` to exit.

### Audio demo

```powershell
python .\demo_audio_rotating_tones.py
```

This starts two rotating tones. Press `Ctrl+C` to stop.

### Assistive navigation demo

```powershell
python .\demo_navigation_assist.py
```

This starts the first end-to-end navigation prototype. It uses depth + IMU data to estimate a ground plane, segments remaining obstacles into a 3x5 grid, computes configurable percentile depth (currently 5th percentile) and TTC per sector, and drives a five-column spatial audio soundscape. Press `q` or `Esc` to exit the preview window.

Use `--no-preview` if you want audio without the OpenCV debug window.

## Import Examples

### Use the RealSense driver in your own code

```python
from realsense_driver import D435iDriver

with D435iDriver() as driver:
    bundle = driver.wait_for_bundle(timeout_s=2.0)
    if bundle is not None:
        print(bundle.color.image.shape)
        print(bundle.depth.image.shape)
        print(driver.imu_enabled_runtime)
```

Pass `enable_imu=False` if you explicitly want depth/color only on a non-IMU RealSense device.

### Use the spatial audio tone class

```python
from audio_spatial_tone import SpatialTone

tone = SpatialTone(initial_pitch_hz=440.0, initial_volume=0.2, initial_azimuth_deg=0.0)
tone.start()
try:
    tone.set_pitch(660.0)
    tone.set_azimuth(-45.0)
finally:
    tone.stop()
```

## Sharing This Branch

For another user to run this branch on their own machine, they need:
- this repository contents
- Python installed
- `pip install -r requirements.txt`
- a RealSense D435i and drivers/runtime if they want camera features
- an audio output device if they want the tone demo
