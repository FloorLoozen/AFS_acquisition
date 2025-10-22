# AFS Acquisition

## Overview

Real-time control of cameras, positioning stages, function generators, and oscilloscopes with scientific data recording in HDF5 format. Features comprehensive experimental metadata, automated measurement sequences, and resonance analysis.

**Key Features:**
- 🎥 High-performance video recording with metadata
- 🎛️ Multi-instrument control (cameras, stages, function generators, oscilloscopes)  
- 📊 Scientific data management with HDF5 format
- 🔍 Interactive frequency sweep analysis
- ⚡ Force path design and automated measurements
- 🖥️ Modern PyQt5 interface with real-time monitoring

## Quick Start

```bash
# Clone and setup
git clone https://github.com/FloorLoozen/AFS_tracking.git
cd AFS_tracking
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run application
python src/main.py
```

**Requirements:** Python 3.8+, Windows 10/11 (primary), 4+ GB RAM

## Hardware Support

| Type | Manufacturer | Models | Interface |
|------|--------------|--------|-----------|
| Camera | IDS Imaging | uEye Series | USB 3.0 |
| XY Stage | Mad City Labs | MicroDrive | USB |
| Function Generator | Siglent | SDG Series | USB/VISA |
| Oscilloscope | Tektronix | Various | VISA |

*Includes demo modes for testing without hardware*

## Data Format

HDF5 scientific format with hierarchical structure:
```
experiment.hdf5
├── video/                    # 4D video dataset
├── function_generator_timeline/  # Output sequences  
├── execution_data/           # Measurement results
│   ├── force_path_design_*/
│   ├── resonance_finder_data_*/
└── metadata/                 # Complete experimental context
    ├── recording/, camera_settings/
    ├── stage_settings/, system_info/
```

## Key Components

- **Main Interface**: Live camera view, instrument controls, real-time monitoring
- **Force Path Designer**: Multi-point measurement sequences with spatial/temporal control
- **Resonance Finder**: Interactive frequency sweeps with automatic peak detection
- **Configuration Manager**: Hardware profiles and performance optimization

## Keyboard Shortcuts

- `Ctrl + Arrow Keys`: Stage movement (fine)
- `Ctrl + Shift + Arrow`: Stage movement (coarse)  
- `Space`: Start/stop recording
- `F11`: Toggle fullscreen

## Architecture

**Robust Design:**
- Multi-threaded video capture and processing
- Comprehensive error handling and recovery
- Hardware abstraction with graceful fallbacks
- Scientific-grade data validation
- Performance optimization for real-time operation

**Core Structure:**
- `src/main.py`: Application entry point
- `src/ui/`: PyQt5 user interface
- `src/controllers/`: Hardware communication
- `src/utils/`: Data management and utilities

## API Example

```python
from src.controllers.camera_controller import CameraController
from src.utils.hdf5_video_recorder import HDF5VideoRecorder

# Initialize components
camera = CameraController(camera_id=0)
recorder = HDF5VideoRecorder("experiment.hdf5", frame_shape=(480, 640, 3))

# Start recording with metadata
camera.start_capture()
recorder.start_recording(metadata={"experiment": "resonance_analysis"})
```

---

**Version 3.0.0** | Python 3.8+ | Scientific HDF5 Data Standard


