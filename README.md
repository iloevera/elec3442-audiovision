# AudioVision: Advanced Spatial Perception for the Visually Impaired

The core concept is to transcend the limitations of natural human echolocation and existing assistive technologies by equipping users with high-bandwidth sensors to gather rich spatial data, which is then processed and translated into a custom, non-verbal audio language delivered via headphones. Unlike natural echolocation, this system offers high-resolution, long-range environmental detail, and an expressive audio language that maps diverse spatial variables to distinct audio cues.

## Functions & Goals

The primary goal is to empower blind individuals with a "sixth sense" that allows intuitive environmental perception. The system aims to:
- **Provide Real-Time Spatial Awareness**: Accurately convey object location, movement, and environmental features.
- **Enhance Safety**: Proactively alert users to collision risks, including predictive non-collision logic.
- **Facilitate Independent Navigation**: Guide users along paths, around obstacles, and through complex intersections.
- **Maintain Low Profile**: Be inconspicuous and socially acceptable.
- **Be Intuitive & Trainable**: Employ a carefully designed audio language that minimizes cognitive load.

## Technical Details

The system processes 3D depth data from an Intel RealSense camera and translates it into spatialized audio feedback.

### Key Features
- **Data Richness**: High-resolution, long-range detail, sensitive to small objects, and immune to ambient noise.
- **Expressive Audio Language**: Maps depth, azimuth, velocity, and collision risk to pitch, panning, timbre, and urgency.
- **High Update Rate**: Audio feedback refreshed at 30-60 Hz, providing significantly more spatial samples per second.
- **Multi-Sensor Fusion**: Integrates data from various sensors (Depth, IMU) for robust perception.

## Project Structure

- `main.py`: Primary application entry point for the assistive navigation system.
- `src/realsense_driver.py`: Interface for Intel RealSense D400-series depth cameras.
- `src/navigation_processing.py`: Core logic for obstacle detection, ground plane estimation, and risk assessment.
- `src/navigation_audio.py`: Translates spatial data into audio cues.
- `src/sensehat_driver.py`: Driver for the Raspberry Pi Sense HAT (LED matrix and Joystick).
- `src/audio_spatial_tone.py`: Low-level audio synthesis and spatialization.

## Setup and Installation

### Prerequisites
- Python 3.10+
- Intel RealSense D435/D435i camera
- Raspberry Pi (optional, for Sense HAT integration)
- Linux environment (tested on Raspberry Pi OS)

### Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install `pyrealsense2` by compiling from source (required for some platforms like Raspberry Pi):
   ```bash
   sudo apt-get install cmake libusb-1.0-0-dev pkg-config libgtk-3-dev libglfw3-dev libglu1-mesa-dev libgl1-mesa-dev
   git clone https://github.com/IntelRealSense/librealsense.git
   cd librealsense
   mkdir build && cd build
   cmake .. -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=$(which python3)
   make -j$(nproc)
   sudo make install
   
   # After compilation, copy the binaries to your virtual environment if applicable:
   # Replace the path and filenames with the ones generated on your system
   export VENV_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
   cp Release/pyrealsense2.cpython-*.so $VENV_PACKAGES/
   cp Release/librealsense2.so* $VENV_PACKAGES/
   
   # Verify installation
   python -c "import pyrealsense2 as rs; print('Success')"
   ```

## Usage

### Running the Main Application
```bash
python main.py
```

### Automatic Startup (Raspberry Pi)
To configure the application to run automatically on power-on using systemd:
1. Run the setup script:
   ```bash
   ./setup_autostart.sh
   ```
2. To restart the service manually:
   ```bash
   ./restart_service.sh
   ```
3. To view logs:
   ```bash
   sudo journalctl -u audiovision.service -f
   ```

### Controls (Sense HAT Joystick / Keyboard)
- **Up / Down Arrow**: Adjust volume.
- **Enter**: Toggle navigation audio on/off.
- **Q / Escape**: Quit application.

### Demos
Several standalone scripts are available to test individual components:
- `demo.py`: General entry point for demos.
- `demo/demo_realsense_preview.py`: Visual preview of the depth and color streams.
- `demo/demo_imu.py`: Displays IMU and orientation data from the Sense HAT.
- `demo/demo_audio_rotating_tones.py`: Tests spatial audio positioning.
- `demo/demo_navigation_assist.py`: Core navigation logic demo.

## License
Refer to the [LICENSE](LICENSE) file for details.


## Safety and Research Note

This is a research prototype, not a certified medical device.
Use controlled testing practices before any real-world mobility deployment.
