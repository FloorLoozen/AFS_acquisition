# AFS Acquisition

Scientific instrument control and data acquisition system for high-speed camera recording with synchronized positioning stages, function generators, and oscilloscopes.

## System Requirements

**Platform:**
- Windows 11 Enterprise (x64)
- Intel Core i7-14700 (20 cores, 28 threads)
- 64 GB RAM
- AMD Radeon Pro WX 3100 (OpenCL support)

**Software:**
- Python 3.11+
- PyQt5 3.0.0

## Hardware Configuration

**Camera:**
- IDS uEye UI-3080CP (MONO8)
- Resolution: 1296 × 1024 pixels
- Max Frame Rate: 57.2 FPS
- Interface: USB 3.0

**Positioning Stages:**
- Mad City Labs Nano-Drive XY Stage
- Mad City Labs Nano-Drive Z Stage
- Interface: USB

**Signal Generation & Measurement:**
- Siglent SDG1032X Function Generator (USB/VISA)
- Siglent SDS804X HD Oscilloscope (USB/VISA)

## Installation

```bash
git clone https://github.com/FloorLoozen/AFS_acquisition.git
cd AFS_acquisition
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

## Features

**Real-Time Acquisition:**
- 30 FPS high-speed recording (MONO8, 1296×1024)
- 12 FPS live preview during recording
- Real-time LZF compression
- 2× downsampling option

**Instrument Control:**
- Synchronized camera and stage positioning
- Function generator waveform control
- Oscilloscope measurement integration
- Lookup table (LUT) acquisition for calibration

**Data Management:**
- HDF5 hierarchical storage format
- Automatic metadata capture
- Force path execution with multi-point measurements
- Resonance frequency sweep analysis

**User Interface:**
- Live camera view with histogram and statistics
- Hardware status monitoring
- Configurable measurement settings (10×, 20×, 40× magnification)
- Keyboard shortcuts for stage control and recording

## Data Format

HDF5 files with the following structure:

```
recording.hdf5
├── raw_data/
│   ├── main_video                  # Shape: (n_frames, height, width, 1), dtype: uint8
│   ├── function_generator_timeline # FG parameters over time
│   └── LUT/                        # Lookup table calibration data
└── meta_data/
    ├── hardware_settings/
    │   ├── camera_settings         # Exposure, gain, FPS, pixel format
    │   └── stage_settings          # XY/Z position coordinates
    ├── recording_info              # Timestamp, duration, frame count
    └── force_path_execution        # Measurement sequence (if used)
```

**Video Encoding:**
- Format: MONO8 (8-bit grayscale)
- Compression: LZF (real-time) or GZIP (configurable)
- Downscaling: 1× (full), 2× (half), 4× (quarter)
- Storage: 4D array `(frames, height, width, 1)`

## Performance

**Frame Rates:**
- Camera acquisition: 57.2 FPS (hardware maximum)
- Recording: 30 FPS (software-limited for stable performance)
- Live preview: 12 FPS (UI optimization)

**Compression:**
- Real-time LZF compression during acquisition
- Optional post-processing GZIP compression
- Typical file size: ~25 MB per second of recording

**Processing:**
- Asynchronous frame handling (ThreadPoolExecutor)
- 16 worker threads (optimized for i7-14700)
- OpenCL GPU support for display processing (optional)
- Automatic camera buffer flushing for clean recordings

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + Arrow Keys` | Fine stage movement |
| `Ctrl + Shift + Arrow` | Coarse stage movement |
| `Ctrl + Space` | Start/stop recording |
| `F11` | Toggle fullscreen |

---

**Version 3.0.0** | Scientific Data Acquisition System


