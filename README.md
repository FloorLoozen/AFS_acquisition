# AFS Tracking System

Scientific data acquisition system for Atomic Force Spectroscopy with **HDF5 video recording**, camera control, and XY stage positioning.

## 🚀 Key Features

### Video Recording & Data Management
- **HDF5 Video Format**: Scientific-grade recording with frame-level access and compression
- **Comprehensive Metadata**: Sample information, camera settings, and stage positions automatically saved
- **Clean Data Structure**: Only relevant metadata - no placeholder values or technical noise

### Hardware Integration  
- **IDS uEye Camera Support**: Live feed with automatic reconnection and hardware optimization
- **MCL MicroDrive XY Stage**: Precise positioning with keyboard shortcuts (Ctrl+Arrow keys)
- **Graceful Fallbacks**: Test pattern mode when hardware unavailable

### User Interface
- **Modern PyQt5 GUI**: Intuitive measurement setup and acquisition controls
- **Real-time Monitoring**: Live camera feed with performance statistics
- **Keyboard Shortcuts**: Quick stage movement and measurement controls
- **Color-coded Logging**: Console and file logging with detailed system information

## 🏃‍♂️ Quick Start

### 1. Setup Environment
```powershell
git clone https://github.com/FloorLoozen/AFS_tracking
cd AFS_tracking
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
# Optionally install IDS camera support from their website
```

### 3. Run Application
```powershell
python -m src.main
```

The application will automatically optimize itself for your system on first run!

## 📊 HDF5 Video Recording

### File Structure
```
📁 recording.hdf5
├── 📹 video/                           # 4D dataset: (frames, height, width, channels)
├── 📝 metadata/                        # Sample info, notes, operator
├── 🔧 hardware_settings/               # Camera & stage parameters
│   ├── camera/                         # Real camera settings (23+ parameters)
│   └── xy_stage/                       # Stage position and configuration
└── 📊 function_generator_timeline/     # Function generator event timeline
    └── timeline/                       # Timestamped frequency/voltage changes
```



### Function Generator Timeline Features
**Complete State Recreation**: The timeline captures EVERY function generator operation:
- ✅ **Initial State**: Starting frequency/voltage/on-off status when recording begins
- ✅ **Frequency Changes**: Every MHz adjustment logged with precise timing (13-15 MHz range)
- ✅ **Voltage Changes**: Every Vpp adjustment logged independently
- ✅ **Output Toggle**: ON/OFF events with current settings preserved
- ✅ **Combined Changes**: Simultaneous frequency + voltage modifications
- ✅ **Rapid Changes**: Debounced but complete capture of user adjustments

**Default Settings**:
- 🎯 **Default Frequency**: 14.0 MHz (optimal frequency)
- 📏 **Valid Range**: 13.0 - 15.0 MHz (enforced in UI and validation)
- ⚡ **Default Voltage**: 4.0 Vpp

**Event Types Logged**:
- `initial_state`: Function generator state when recording starts
- `output_on`: Output enabled with current frequency/voltage
- `output_off`: Output disabled with settings preserved  
- `parameter_change`: Frequency and/or voltage modified while running

**Simplified Timeline Structure**: Only essential data stored (no absolute timestamps or channel info for efficiency)

## 🛠️ System Requirements

### Software Dependencies
- **Python 3.9+** (tested with 3.13)
- **PyQt5**: Modern UI framework
- **h5py**: HDF5 scientific data format  
- **NumPy**: Numerical computing foundation
- **OpenCV**: Image processing and display
- **psutil**: System performance monitoring
- **colorama**: Enhanced console output

### Performance Optimizations
- **Multi-threaded**: Camera capture, HDF5 writing, and UI run independently
- **Memory pooling**: Reduces garbage collection overhead
- **Async HDF5 writes**: Non-blocking data recording with batching
- **Auto-configuration**: Automatically optimizes for your system
- **Frame pooling**: Efficient memory reuse for high frame rates

### Optional Hardware  
- **IDS uEye Camera**: High-performance scientific imaging
- **MCL MicroDrive XY Stage**: Precision positioning system

*System works in test mode without hardware for development and testing.*

## 📋 Usage Workflow

1. **Launch Application**: `python -m src.main`
2. **Configure Measurement**: Set sample name, notes, and save path
3. **Position Sample**: Use Ctrl+Arrow keys for stage movement  
4. **Start Recording**: Click record button or use keyboard shortcut

## ⚡ Performance Optimization

### Automatic Optimization
The system automatically optimizes itself based on your hardware:
- **CPU cores**: Adjusts thread pool size
- **RAM amount**: Configures frame buffers and queue sizes  
- **Disk space**: Selects optimal compression settings

### Manual Performance Presets
Access through configuration or programmatically:

```python
from src.utils.config_manager import apply_performance_preset

# Maximum performance (high CPU/RAM usage)
apply_performance_preset("max_performance")

# Balanced performance (recommended)
apply_performance_preset("balanced") 

# Memory efficient (low RAM systems)
apply_performance_preset("memory_efficient")
```

For HDF5 data handling, use external tools and scripts.

## 🔧 Configuration

- **Default Save Path**: `C:/Users/fAFS/Documents/Floor/tmp`
- **Log Files**: `logs/afs_tracking_YYYYMMDD_HHMMSS.log`
- **Camera Settings**: Auto-detected and saved with each recording
- **Stage Settings**: Position and configuration preserved in metadata

---

**Version**: 2.0 | **Updated**: September 2025  
**Format**: HDF5 Scientific Data | **License**: Research Use


