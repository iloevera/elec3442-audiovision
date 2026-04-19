# AudioVision Prototype

Wearable synthetic audio-vision prototype for assistive navigation.

This repository contains a real-time pipeline that senses the environment with depth + IMU data and converts spatial risk into directional audio cues.

## Project Snapshot

This project tackles the navigation information gap left by cane-range sensing and limited auditory bandwidth in crowded real-world environments. The vision is a low-profile wearable "audio vision" system that delivers an intuitive, trainable spatial cue stream. The prototype goal is to provide real-time, direction-aware obstacle awareness and early hazard signaling (distance + TTC + ground context) using depth/IMU sensing and spatial audio.

## Current Scope in This Repo

Implemented now:
1. Intel RealSense D435i depth/color/IMU capture.
2. Ground-plane estimation with IMU guidance.
3. Grid-based obstacle analysis with percentile depth and TTC metrics.
4. Spatial multi-voice audio cue generation.
5. Interactive demos for preview, audio behavior, and end-to-end navigation.

Future-oriented ideas (not fully implemented here):
1. Multi-camera fusion.
2. Richer environmental affordance layers (crossings, signage, tripping hazards).

## Project Layout

- src/realsense_driver.py: RealSense capture driver and frame bundle API.
- src/navigation_processing.py: obstacle/ground analysis and risk scoring.
- src/navigation_audio.py: column-level audio controller.
- src/audio_spatial_tone.py: controllable spatial tone voice.
- src/audio_mixer.py: shared real-time mixer for multiple voices.
- demo_realsense_preview.py: camera and IMU OpenCV preview.
- demo_audio_rotating_tones.py: rotating stereo tone demonstration.
- demo_navigation_assist.py: end-to-end navigation demo with optional preview.

## Requirements

1. Windows with Python 3.10 or 3.11.
2. Intel RealSense D435i for camera workflows.
3. Speakers or headphones for audio demos.
4. Python dependencies from requirements.txt.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installing pyrealsense2 fails, install Intel RealSense runtime/SDK first, then retry.

## Run Demos

RealSense preview:

```powershell
python .\demo_realsense_preview.py
```

Audio rotating tones:

```powershell
python .\demo_audio_rotating_tones.py
```

Navigation assist:

```powershell
python .\demo_navigation_assist.py
```

Navigation runtime modes:

```powershell
python .\demo_navigation_assist.py --mode desktop_debug
python .\demo_navigation_assist.py --mode pi_normal
python .\demo_navigation_assist.py --mode pi_debug --preview-fps 8
```

Useful flags:
1. --preview
2. --no-preview
3. --no-audio

## Audio Encoding (Current)

At the prototype stage, cues are intentionally simple and consistent:
1. Azimuth maps to stereo position (left/right direction).
2. Risk and proximity drive pitch and loudness.
3. TTC urgency increases pulse rate.
4. Column-based scene partitioning keeps cue density manageable.

## Minimal Import Examples

```python
from src.realsense_driver import D435iDriver
from src.audio_spatial_tone import SpatialTone
```

## Safety and Research Note

This is a research prototype, not a certified medical device.
Use controlled testing practices before any real-world mobility deployment.
