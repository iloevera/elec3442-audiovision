# RealSense Test

Small Python project for:
- Intel RealSense D435i depth/color capture with an optional IMU fallback
- simple spatial audio tone generation and demo playback

## Files

- `realsense_driver.py`: reusable `D435iDriver` class with preview mode
- `audio.py`: convenience export for `SpatialTone`
- `audio_spatial_tone.py`: continuously playing tone class with live pitch, volume, and azimuth control
- `audio_mixer.py`: shared audio output mixer for multiple simultaneous tones
- `demo_audio_rotating_tones.py`: rotating stereo tone demo

## Requirements

- Windows with Python 3.10 or 3.11 recommended
- Intel RealSense D435i connected over USB 3 if you want camera access
- RealSense depth/color work without IMU if IMU stream negotiation fails
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
- if IMU cannot be started, this branch automatically falls back to depth/color only

## Run

### RealSense preview

```powershell
python .\realsense_driver.py
```

This opens a preview window with color and depth. Press `q` or `Esc` to exit.

### Audio demo

```powershell
python .\demo_audio_rotating_tones.py
```

This starts two rotating tones. Press `Ctrl+C` to stop.

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

### Use the spatial audio tone class

```python
from audio import SpatialTone

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
