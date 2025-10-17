# AFS Tracking System

Scientific data acquisition system for Atomic Force Spectroscopy with HDF5 video recording, camera control, and XY stage positioning.

## 🚀 Key Features

- **HDF5 Video Recording**: Scientific-grade recording with comprehensive metadata
- **IDS uEye Camera**: Live feed with automatic reconnection
- **MCL MicroDrive XY Stage**: Precise positioning with keyboard shortcuts (Ctrl+Arrow keys)
- **Function Generator Control**: Real-time frequency/amplitude adjustment with timeline logging
- **Modern PyQt5 GUI**: Intuitive controls with real-time monitoring
- **Graceful Fallbacks**: Test pattern mode when hardware unavailable

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
├── � data/
│   ├── �📹 video/                       # 4D dataset: (frames, height, width, channels)
│   ├── � function_generator_timeline/ # Function generator event timeline  
│   └── 📋 look_up_table/               # (To be implemented later)
└── 📝 meta_data/
    ├── 🔧 hardware_settings/           # Camera & stage parameters
    │   ├── camera/                     # Real camera settings (23+ parameters)
    │   └── xy_stage/                   # Stage position and configuration  
    └── � resonance_finder/            # (To be implemented later)
        ├── figure/                     # Figure data
        └── list/                       # List data
```



### Function Generator Timeline
The system logs all function generator operations during recording:
- **Initial State**: Starting settings when recording begins
- **Parameter Changes**: Frequency (13-15 MHz) and voltage (Vpp) adjustments  
- **Output Events**: ON/OFF toggle events with timestamps

**Defaults**: 14.0 MHz, 4.0 Vpp

## 🛠️ System Requirements

### Software Dependencies
- **Python 3.9+** (tested with 3.13)
- **PyQt5**: UI framework
- **h5py, NumPy, OpenCV**: Core scientific libraries
- **psutil, colorama**: System utilities

### Hardware (Optional)
- **IDS uEye Camera**: High-performance scientific imaging
- **MCL MicroDrive XY Stage**: Precision positioning system
- **Function Generator**: Siglent or compatible VISA instrument

*System works in test mode without hardware for development and testing.*

## 📋 Usage Workflow

1. **Launch Application**: `python -m src.main`
2. **Configure Measurement**: Set sample name, notes, and save path
3. **Position Sample**: Use Ctrl+Arrow keys for stage movement  
4. **Start Recording**: Click record button or use keyboard shortcut



## 🔧 Configuration

- **Default Save Path**: `C:/Users/fAFS/Documents/Floor/tmp`
- **Log Files**: `logs/afs_tracking_YYYYMMDD_HHMMSS.log`
- **Camera Settings**: Auto-detected and saved with each recording
- **Stage Settings**: Position and configuration preserved in metadata

---

**Version**: 2.1 | **Updated**: October 2025  
**Format**: HDF5 Scientific Data | **License**: Research Use


