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

---

**Version 3.0.0** | Python 3.8+ | Scientific HDF5 Data Standard


