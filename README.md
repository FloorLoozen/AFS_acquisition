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

## Live View / Recording Decoupling

The application separates the live display from recording to ensure a responsive UI even when disk I/O or compression is slow.

- Live view runs at 12 FPS (optimized to prevent lag)
- Recording runs at 30 FPS with strict rate limiting
- Real-time LZF compression during recording (no post-processing wait)
- Frames are enqueued and processed asynchronously to avoid blocking the GUI
- Live view continues during recording with dual FPS display

This design prevents lag in the live view while maintaining high-quality recordings.

## Recent Optimizations (2025-11-27)

### Recording Performance
- **30 FPS Recording**: Fixed double rate limiting issue (was 17 FPS, now 30 FPS)
- **Real-time Compression**: Enabled LZF compression during recording (eliminates 77+ second post-processing wait)
- **Instant Save**: Files ready immediately after stopping recording
- **File Size**: LZF compression more effective than post-processing GZIP (was increasing files 59%)

### Post-LUT Bug Fixes
- **Black Recording Fix**: Automatic camera settings restoration after LUT acquisition
- **Buffer Flushing**: Stale frames automatically flushed before recording
- **Camera Restart**: Clean camera state after LUT completion
- **Settings Recovery**: Exposure/gain/FPS properly restored (5ms, gain 2, 30 FPS)

### Code Quality
- **Eliminated Duplication**: Extracted `_flush_camera_buffer()` helper method
- **Unified Recording Path**: Both LUT and non-LUT recordings use same code path
- **Removed Redundant Imports**: Moved imports to module level
- **Better Error Handling**: Graceful fallbacks for camera state issues
- **Improved Logging**: Clear status messages for debugging

### Performance Metrics
- Camera: 57.2 FPS max (IDS uEye MONO8)
- Live Display: 12 FPS (lag-free UI)
- Recording: 30 FPS (strict enforcement)
- File Format: HDF5 with LZF compression
- Typical File Size: ~25 MB/second of recording

## Keyboard Shortcuts

- `Ctrl + Arrow Keys`: Stage movement (fine)
- `Ctrl + Shift + Arrow`: Stage movement (coarse)  
- `Ctrl + Space`: Start/stop recording
- `F11`: Toggle fullscreen

---

**Version 3.0.0** | Python 3.8+ | Scientific HDF5 Data Standard


