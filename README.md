# AFS Acquisition

Scientific instrument control and data acquisition system for high-speed camera recording with synchronized hardware control, real-time visualization, and comprehensive data management.

## System Requirements

**Platform:**
- Windows 11 (x64)
- Intel Core i7-14700 (20 cores, 28 threads)
- 64 GB RAM
- AMD Radeon Pro WX 3100 (OpenCL support)

**Software:**
- Python 3.11+
- PyQt5 for GUI
- OpenCL for GPU acceleration (optional)

## Hardware Configuration

**Camera:**
- **Model:** IDS uEye UI-3080CP Rev.2
- **Sensor:** CMOS MONO8 (8-bit grayscale)
- **Resolution:** 1296 × 1024 pixels (1.33 MP)
- **Pixel Size:** 5.3 × 5.3 µm
- **Frame Rate:** 57.2 FPS maximum
- **Exposure Range:** 0.1 - 1000 ms
- **Gain Range:** 0 - 100
- **Interface:** USB 3.0 (5 Gbps)
- **Bit Depth:** 8-bit

**Positioning Stages:**
- **XY Stage:** Mad City Labs Nano-Drive
  - Travel Range: ±12.5 mm (25 mm total)
  - Interface: USB
- **Z Stage:** Mad City Labs Nano-Drive
  - Travel Range: 0 - 100 µm
  - Interface: USB
- **Control:** Madlib.dll (Windows DLL API)

**Test Equipment:**
- **Function Generator:** Siglent SDG1032X
  - Channels: 2
  - Frequency Range: 1 µHz - 30 MHz
  - Sample Rate: 150 MSa/s
  - Amplitude: 1 mVpp - 20 Vpp (into 50Ω)
  - Interface: USB-TMC (VISA/SCPI)
- **Oscilloscope:** Siglent SDS804X HD
  - Channels: 4
  - Bandwidth: 100 MHz
  - Sample Rate: 1 GSa/s
  - Resolution: 12-bit
  - Interface: USB-TMC (VISA/SCPI)

## Quick Start

### Option 1: Run Executable (Recommended)
```bash
# Download or build AFS_acquisition.exe
# Double-click to run - no Python installation needed
```

### Option 2: Run from Source
```bash
# Clone repository
git clone https://github.com/FloorLoozen/AFS_acquisition.git
cd AFS_acquisition

# Install dependencies
pip install -r requirements.txt

# Launch application
python src/main.py
```

### Building Executable
```bash
# Install dependencies
pip install -r requirements.txt
pip install pyinstaller pillow

# Build exe
python build_exe.py

# Executable will be created at:
# C:\Users\AFS\Documents\Software\AFS_acquisition.exe
```

## How to Use

### Starting the Application

1. **Launch:** Run `python src/main.py`
2. **Hardware Detection:** The application automatically detects and connects to available hardware on startup
3. **Status Check:** Verify hardware connections in the status bar (green = connected, red = disconnected)

### Recording Video Data

**Basic Recording:**
1. Adjust camera settings (exposure, gain) for optimal image quality
2. Set recording frame rate (default 30 FPS)
3. Click "Start Recording" or press `Ctrl+Space`
4. Click "Stop Recording" when finished
5. Data automatically saves to HDF5 file in timestamped format

**Monitor Status:** Watch the status indicator for feedback:
- Gray = Ready to record
- Blue = Recording in progress
- Blue = Saving data
- Green = Successfully saved
- Red = Error occurred

### Controlling Positioning Stages

**Manual Control:**
1. Press `Ctrl+S` to open stage control dialog
2. Use arrow buttons or keyboard shortcuts:
   - `Ctrl+Left/Right` = Move X axis
   - `Ctrl+Up/Down` = Move Y axis
   - `Ctrl+U` = Move Z up
   - `Ctrl+D` = Move Z down
3. Adjust step size for precision (smaller) or speed (larger)
4. Monitor current position in real-time display

**Absolute Positioning:**
1. Enter desired X, Y, Z coordinates directly
2. Click "Move to Position"
3. Stage moves automatically to target location

**Status Feedback:**
- Gray = Ready to move
- Blue = Moving in progress
- Red = Out of range (requested position exceeds limits)

### Finding Resonance Frequencies

**Automated Sweep:**
1. Open "Resonance Finder" tab
2. Set sweep parameters:
   - **Start Frequency:** Beginning of sweep range (MHz)
   - **Stop Frequency:** End of sweep range (MHz)
   - **Amplitude:** Signal voltage (Vpp)
   - **Sweep Time:** Duration of sweep (seconds)
3. Click "Start Sweep"
4. Watch real-time plot of oscilloscope voltage vs frequency
5. Click on plot to select resonance peaks
6. Results automatically saved to HDF5 execution log

**Status Flow:**
- Gray = Ready to sweep
- Blue = Sweeping in progress
- Blue = Retrieving oscilloscope data
- Red = Error (check hardware connections)

### Generating Calibration Lookup Tables (LUT)

**Z-Calibration for Particle Tracking:**
1. Open "LUT Generator" tab
2. Configure Z-scan parameters:
   - **Start Z:** Initial focal plane position (µm)
   - **End Z:** Final focal plane position (µm)
   - **Step Size:** Distance between captures (µm)
   - **Settle Time:** Wait time before capture (ms)
3. Click "Start Acquisition"
4. System automatically:
   - Moves Z stage through range
   - Captures diffraction pattern at each position
   - Pauses live view during acquisition
   - Saves LUT data to HDF5
5. Use resulting LUT for 3D particle tracking algorithms

**Status Flow:**
- Gray = Ready to capture
- Blue = Capturing frames
- Blue = Saving LUT data
- Green = LUT saved successfully

### Designing Force Path Measurements

**Multi-Point Acquisition:**
1. Open "Force Path Designer" tab
2. Click grid cells to define measurement path
3. Set measurement parameters (frequency, duration)
4. Click "Execute Path"
5. System automatically:
   - Moves stage to each point in sequence
   - Applies function generator signal
   - Records camera data
   - Saves results with metadata

### Configuring Measurement Settings

**Function Generator Control:**
1. Open "Measurement Controls" tab
2. Set drive frequency (Hz)
3. Select magnification objective (10×, 20×, or 40×)
4. Enable Channel 1 or Channel 2 output
5. Monitor generator status (green = on, gray = off)

### Understanding Status Indicators

All controls show color-coded status for instant feedback:
- **Red:** Error, disconnected, out of range
- **Blue:** Busy (recording, moving, sweeping, capturing)
- **Green:** Connected, operation completed successfully
- **Gray:** Ready, idle, waiting for user input
- **Orange:** Transitional (initializing, connecting)
- **Yellow:** Paused
- **Purple:** Test mode

## Data Storage Format

All data is saved in HDF5 hierarchical format with comprehensive metadata.

### Repository Structure

```
AFS_acquisition/
├── src/
│   ├── __init__.py
│   ├── main.py                        # Application entry point
│   ├── controllers/
│   │   ├── camera_controller.py       # IDS uEye camera control
│   │   ├── device_manager.py          # Hardware initialization
│   │   ├── function_generator_controller.py
│   │   ├── oscilloscope_controller.py
│   │   ├── stage_manager.py           # XY/Z stage coordination
│   │   ├── xy_stage_controller.py
│   │   └── z_stage_controller.py
│   ├── ui/
│   │   ├── acquisition_controls_widget.py
│   │   ├── camera_settings_widget.py
│   │   ├── camera_widget.py           # Live video display
│   │   ├── force_path_designer_widget.py
│   │   ├── frequency_settings_widget.py
│   │   ├── hardware_status_dialog.py
│   │   ├── lookup_table_widget.py
│   │   ├── main_window.py             # Main application window
│   │   ├── measurement_controls_widget.py
│   │   ├── resonance_finder_widget.py
│   │   └── stages_widget.py
│   └── utils/
│       ├── config_manager.py          # Configuration management
│       ├── constants.py               # System constants
│       ├── data_integrity.py          # Checksum validation
│       ├── exceptions.py              # Custom exceptions
│       ├── hdf5_video_recorder.py     # HDF5 recording engine
│       ├── keyboard_shortcuts.py      # Hotkey management
│       ├── logger.py                  # Error-only logging
│       ├── performance_monitor.py     # Performance tracking
│       ├── state_recovery.py          # State persistence
│       ├── status_display.py          # Status indicators
│       ├── validation.py              # Input validation
│       └── visa_helper.py             # VISA/SCPI utilities
├── build_exe.py                       # PyInstaller build script
├── build_exe.bat                      # Build wrapper
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── CHANGELOG.md                       # Version history
```

### HDF5 Data File Structure

```
measurement_YYYYMMDD_HHMMSS.hdf5
├── raw_data/
│   ├── main_video                     # Video frames (n_frames, height, width, 1)
│   ├── function_generator_timeline    # FG parameters over time
│   └── LUT/                           # Lookup table calibration data
│       ├── frames                     # Diffraction patterns
│       └── z_positions                # Corresponding Z positions
├── meta_data/
│   ├── hardware_settings/
│   │   ├── camera_settings           # Exposure, gain, FPS, format
│   │   ├── stage_settings            # XY/Z positions
│   │   └── function_generator_settings
│   ├── recording_info                # Timestamp, duration, frame count
│   ├── performance_metrics           # FPS, compression stats
│   └── audit_trail                   # Data integrity log
└── execution_log/
    ├── force_path_execution_XXX      # Measurement sequences
    └── resonance_sweep_XXX           # Frequency sweep results
        ├── frequencies_mhz
        ├── voltages_v
        ├── selected_frequencies_mhz
        └── plot_image_png            # Saved plot
```

### Video Encoding
- **Format:** MONO8 (8-bit grayscale)
- **Compression:** LZF (real-time) or GZIP (configurable)
- **Downscaling:** 1× (full resolution), 2× (half), 4× (quarter)
- **Shape:** 4D array `(frames, height, width, 1)`

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + Arrow Keys` | Move XY stage (fine control) |
| `Ctrl + U` | Move Z stage up |
| `Ctrl + D` | Move Z stage down |
| `Ctrl + Space` | Toggle recording |
| `Ctrl + S` | Open stage control dialog |

**Note:** Camera orientation is rotated, so:
- UP/DOWN arrows control Y axis
- LEFT/RIGHT arrows control X axis

## Performance Specifications

**Camera Performance:**
- **Hardware Maximum:** 57.2 FPS at 1296×1024 MONO8
- **Recording Rate:** 30 FPS (stable real-time acquisition)
- **Live Preview:** 12 FPS (optimized for UI responsiveness)
- **Data Rate:** ~40 MB/s uncompressed (~25 MB/s with LZF compression)
- **Latency:** <50 ms camera-to-display

**Data Compression:**
- **Real-time (during recording):** LZF compression level 1 (~35% reduction, no frame drops)
- **Post-processing (after recording):** GZIP-9 + shuffle filter (60-80% total reduction)
- **Compression Ratio:** 1.5:1 real-time (LZF), 3.5-5:1 post-processing (GZIP-9)
- **File Size (30 FPS, half res 648×512):** ~23 MB/sec during recording, ~5-9 MB/sec after compression
- **Post-processing Speed:** ~150-200 MB/sec on i7-14700 (20 parallel workers)

**Compression Tool:**
```bash
# Compress single recording (lossless, 60-70% reduction)
python compress_recordings.py recording.hdf5

# Compress with quality reduction (70-80% reduction, slightly lossy)
python compress_recordings.py recording.hdf5 --quality-reduction

# Compress all recordings in folder
python compress_recordings.py recordings/

# Example: 700 MB → 140-210 MB in ~15-20 seconds
```

**Processing Architecture:**
- **Threading:** ThreadPoolExecutor with 16 workers (optimized for i7-14700)
- **GPU Acceleration:** OpenCL support for display rendering (optional)
- **Buffer Management:** Automatic flushing prevents frame carryover
- **Memory:** ~2 GB typical usage during recording

**Stage Performance:**
- **XY Movement Speed:** Configurable step size (1 µm - 1000 µm)
- **Z Movement Speed:** Configurable step size (0.1 µm - 50 µm)
- **Positioning Accuracy:** Nanometer-scale repeatability
- **Response Time:** <100 ms for typical movements

**Data Acquisition:**
- **Storage Format:** HDF5 with hierarchical structure
- **Metadata:** Comprehensive hardware settings and timestamps
- **Integrity:** Automatic audit trail and checksum validation
- **Write Speed:** >100 MB/s sustained (SSD required)

**Logging System:**
- **Error-Only Mode:** Log files created only when errors occur
- **Log Location:** `logs/afs_error_YYYYMMDD_HHMMSS.log` (in executable directory)
- **Normal Operation:** No log files created
- **Performance Impact:** <1% CPU overhead

**Default Save Location:**
- **HDF5 Files:** `C:\Users\AFS\Documents\Data`
- **Configurable:** Path can be changed in Measurement Settings

## Hardware Auto-Detection

The application automatically detects and connects to available hardware:
- Camera initialization on startup
- Function generator auto-connect
- Oscilloscope auto-connect
- Stage controllers on-demand connection
- Graceful degradation if hardware unavailable

## Error Handling

**Robust operation with:**
- Automatic reconnection for transient failures
- Demo mode when hardware unavailable
- Out-of-range protection for stages
- Buffer overflow prevention
- Disk space monitoring
- Clean shutdown on errors

## Development

**Architecture:**
- **UI Layer:** PyQt5 widgets (`src/ui/`)
- **Controllers:** Hardware abstraction (`src/controllers/`)
- **Utils:** Logging, config, HDF5 management (`src/utils/`)

**Key Design Patterns:**
- Singleton for hardware managers
- Signal/slot for thread-safe UI updates
- Background threading for long operations
- Centralized status display system

---

**Version 1.0.0** | December 2025 | Scientific Data Acquisition System


