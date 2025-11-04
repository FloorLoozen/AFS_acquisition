# AFS Acquisition

## Overview

Real-time control of cameras, positioning stages, function generators, and oscilloscopes with scientific data recording in HDF5 format.

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

**Requirements:** Python 3.8+, Windows 10/11, 4+ GB RAM

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
├── raw_data/
│   ├── main_video                     # 4D video dataset (frames, height, width, channels)
│   ├── function_generator_timeline    # FG parameter changes over time
│   └── LUT/                           # Lookup tables (placeholder)
└── meta_data/
    ├── hardware_settings/
    │   ├── camera_settings            # Camera parameters and configuration
    │   └── stage_settings             # XY stage position and settings
    ├── recording_info                 # Recording session metadata
    └── force_path_execution           # Force path table (optional, if used)
```

### Camera Frame Format

**Expected format:** MONO8 (grayscale, 8-bit unsigned)
- Shape: `(height, width, 1)` - single channel for grayscale
- Data type: `numpy.uint8`
- HDF5 storage: 4D dataset `(n_frames, height, width, 1)`

**Example shapes:**
- Full resolution: `(1200, 1920, 1)` → stored as `(n_frames, 1200, 1920, 1)`
- Quarter resolution (downscale=4): `(300, 480, 1)` → stored as `(n_frames, 300, 480, 1)`

**Performance optimization:**
- Real-time compression: LZF (fast) or GZIP (best)
- Downscale options: 1x (full), 2x (half), 4x (quarter)
- Post-recording compression: configurable via `ConfigManager`

## Key Components

- **Main Interface**: Live camera view, instrument controls, real-time monitoring
- **Force Path Designer**: Multi-point measurement sequences with spatial/temporal control
- **Resonance Finder**: Interactive frequency sweeps with automatic peak detection
- **Configuration Manager**: Hardware profiles and performance optimization

## Keyboard Shortcuts

- `Ctrl + Arrow Keys`: Stage movement (fine)
- `Ctrl + Shift + Arrow`: Stage movement (coarse)  
- `Ctrl + Space`: Start/stop recording
- `F11`: Toggle fullscreen

---

**Version 3.0.0** | Python 3.8+ | Scientific HDF5 Data Standard


