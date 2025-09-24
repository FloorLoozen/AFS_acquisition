# AFS Tracking System

Scientific data acquisition system for Atomic Force Spectroscopy with **HDF5 video recording**, camera control, and XY stage positioning.

## 🚀 Key Features

### Video Recording & Data Management
- **HDF5 Video Format**: Scientific-grade recording with frame-level access and compression
- **Comprehensive Metadata**: Sample information, camera settings, and stage positions automatically saved
- **Clean Data Structure**: Only relevant metadata - no placeholder values or technical noise
- **Analysis Ready**: Direct NumPy compatibility for immediate data processing

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

## 📊 HDF5 Video Recording

### File Structure
```
📁 recording.hdf5
├── 📹 video/              # 4D dataset: (frames, height, width, channels)
├── 📝 metadata/           # Sample info, notes, operator
└── 🔧 hardware_settings/  # Camera & stage parameters
    ├── camera/            # Real camera settings (23+ parameters)
    └── xy_stage/          # Stage position and configuration
```

### Python Analysis Example
```python
import h5py
import numpy as np

# Open recorded data
with h5py.File("measurement.hdf5", 'r') as f:
    # Video data access
    video = f['video']  # Shape: (n_frames, height, width, 3)
    
    # Direct frame access (no video decoding!)
    first_frame = video[0]              # Single frame
    frame_subset = video[10:20]         # Frame range  
    every_10th = video[::10]            # Subsampling
    
    # Metadata access
    sample = f['metadata'].attrs['sample_name'].decode('utf-8')
    notes = f['metadata'].attrs['measurement_notes'].decode('utf-8')
    fps = video.attrs['fps']
    
    # Camera settings
    camera = f['hardware_settings/camera']
    exposure = camera.attrs['exposure_time_us']
    gain = camera.attrs['hardware_gain']
    
    print(f"Sample: {sample}")
    print(f"Frames: {video.shape[0]} at {fps:.1f} FPS")
    print(f"Resolution: {video.shape[1]}x{video.shape[2]}")
```

### Metadata Utility
```powershell
# Inspect any HDF5 file
python src/utils/read_hdf5_metadata.py recording.hdf5
```

## 🛠️ System Requirements

### Software Dependencies
- **Python 3.9+** (tested with 3.13)
- **PyQt5**: Modern UI framework
- **h5py**: HDF5 scientific data format
- **NumPy**: Numerical computing foundation  
- **OpenCV**: Image processing and display
- **colorama**: Enhanced console output

### Optional Hardware
- **IDS uEye Camera**: High-performance scientific imaging
- **MCL MicroDrive XY Stage**: Precision positioning system

*System works in test mode without hardware for development and testing.*

## 📋 Usage Workflow

1. **Launch Application**: `python -m src.main`
2. **Configure Measurement**: Set sample name, notes, and save path
3. **Position Sample**: Use Ctrl+Arrow keys for stage movement  
4. **Start Recording**: Click record button or use keyboard shortcut
5. **Analyze Data**: Load HDF5 files directly in Python scripts

## 🔧 Configuration

- **Default Save Path**: `C:/Users/fAFS/Documents/Floor/tmp`
- **Log Files**: `logs/afs_tracking_YYYYMMDD_HHMMSS.log`
- **Camera Settings**: Auto-detected and saved with each recording
- **Stage Settings**: Position and configuration preserved in metadata

---

**Version**: 2.0 | **Updated**: September 2025  
**Format**: HDF5 Scientific Data | **License**: Research Use


