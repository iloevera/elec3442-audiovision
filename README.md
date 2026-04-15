# RealSense Test

Small Python project for:
- Intel RealSense D435i depth/color capture with an optional IMU fallback
- simple spatial audio tone generation and demo playback
- real-time 3D obstacle tracking and collision-course sonification from depth + optional IMU

## Files

- `realsense_driver.py`: reusable `D435iDriver` class with preview mode
- `audio_spatial_tone.py`: continuously playing tone class with live pitch, volume, and azimuth control
- `audio_mixer.py`: shared audio output mixer for multiple simultaneous tones
- `demo_audio_rotating_tones.py`: rotating stereo tone demo
- `obstacle_models.py`: data contracts for obstacle detections, tracks, and callback payloads
- `imu_compensation.py`: short-horizon gyro-based camera motion compensation helpers
- `obstacle_tracker.py`: depth segmentation, temporal tracking, collision scoring, and callback service
- `obstacle_audio_adapter.py`: maps collision-course obstacles to `SpatialTone` controls
- `demo_obstacle_audio_tracking.py`: end-to-end D435i obstacle tracker + audio demo

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

### Obstacle tracking + audio demo

```powershell
python .\demo_obstacle_audio_tracking.py
```

This starts a background depth tracker that:
- segments obstacles from depth
- tracks obstacles over time with stable IDs
- estimates collision risk using approach velocity and projected time-to-collision (TTC)
- drives spatial tones for collision-course obstacles only

Audio mapping in this demo:
- higher volume for lower projected TTC (more imminent collision)
- higher pitch for closer physical distance
- stereo azimuth from obstacle horizontal angle

IMU mode options:
- default: attempts IMU when available, otherwise automatically runs depth-only (works on D435)
- force depth-only mode: `python .\demo_obstacle_audio_tracking.py --disable-imu`
- require IMU (fail if unavailable): `python .\demo_obstacle_audio_tracking.py --require-imu`

Press `Ctrl+C` to stop.

## Obstacle Tracker Output Contract

The callback payload is `ObstacleUpdate` from `obstacle_models.py`.
Each entry in `ObstacleUpdate.obstacles` is a `TrackedObstacle` containing:
- `obstacle_id`: stable integer track ID
- `xyz_m`: obstacle position in camera frame meters `[x, y, z]`
- `confidence`: tracking confidence score `[0, 1]`
- `distance_m`: Euclidean distance from camera
- `approach_rate_mps`: positive radial closing speed
- `ttc_s`: projected time-to-collision in seconds (`inf` when not closing)
- `collision_score`: normalized threat score `[0, 1]`
- `is_collision_course`: true when obstacle meets collision-course criteria

This output is intended to be consumed by the audio synthesis layer or any other real-time consumer.

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
