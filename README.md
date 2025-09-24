# AFS Tracking

Automated tracking system for AFS using IDS cameras and MCL MicroDrive XY stage hardware with **HDF5 video recording**.

## Key Features

- **HDF5 Video Recording**: Frame-level access with compression and metadata
- **Camera Control**: Live feed from IDS uEye cameras with auto-reconnect  
- **XY Stage Control**: Precise positioning using MCL MicroDrive hardware
- **Integrated Data Storage**: Camera settings, stage positions, and function generator data in one file
- **User Interface**: PyQt5-based GUI with keyboard shortcuts
- **Logging**: Color-coded console and file logging

## Quick Start

1. **Clone and setup**
   ```powershell
   git clone https://github.com/FloorLoozen/AFS_tracking
   cd AFS_tracking
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install pyueye  # If IDS camera available
   ```

3. **Run the application**
   ```powershell
   python -m src.main
   ```

## HDF5 Video Recording

### What's New
- **No more AVI files** - All recordings now use HDF5 format
- **Direct frame access** - Load individual frames without decoding entire videos
- **Integrated metadata** - Sample info, camera settings, and measurement parameters stored with video
- **Scientific workflow** - Optimized for analysis with NumPy and Python

### Basic Usage in Python
```python
import h5py

# Open HDF5 video file
with h5py.File("recording.hdf5", 'r') as f:
    video = f['video']  # 4D dataset: (n_frames, height, width, channels)
    
    # Load specific frame
    frame_10 = video[10]
    
    # Load frame range
    frames_100_to_200 = video[100:200]
    
    # Load every 10th frame
    subsampled = video[::10]
    
    # Get metadata
    fps = video.attrs['fps']
    sample_name = f['metadata'].attrs['sample_name'].decode('utf-8')
```

## Dependencies

- **PyQt5**: UI framework
- **h5py**: HDF5 file format support  
- **NumPy**: Numerical operations
- **OpenCV**: Image processing
- **pyueye**: IDS camera interface (optional)
- **colorama**: Colored console output
- **matplotlib**: For visualization (optional)

## Hardware Requirements

- **IDS uEye camera** (optional - test pattern mode available)
- **MCL MicroDrive XY stage** (optional - software control available)

_Updated: 2025-09-24 - Now with HDF5 video recording!_


