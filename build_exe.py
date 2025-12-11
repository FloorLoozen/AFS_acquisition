"""
Build script for creating AFS Acquisition executable.
This script creates a PyInstaller spec file and builds the exe.
"""

import PyInstaller.__main__
import os
import shutil
from pathlib import Path
from PIL import Image

# Get the project root directory
project_root = Path(__file__).parent

# Convert PNG icon to ICO format
icon_exe_png = Path(r"C:\Users\AFS\Documents\Software\Icons\acquistion_afs.png")  # For exe file and taskbar
icon_window_png = Path(r"C:\Users\AFS\Documents\Software\Icons\acquistion.png")  # For window icon (upper left)
icon_loading_png = Path(r"C:\Users\AFS\Documents\Software\Icons\acquistion_loading.png")  # For splash screen
icon_ico = project_root / "AFS_icon.ico"
icon_window_local = project_root / "AFS_icon.png"
icon_loading_local = project_root / "AFS_loading.png"

if icon_exe_png.exists():
    print(f"Converting exe icon from {icon_exe_png} to {icon_ico}...")
    img = Image.open(icon_exe_png)
    # Create multiple sizes for the icon
    img.save(icon_ico, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print("Exe icon conversion complete!")
else:
    print(f"Warning: Exe icon file not found at {icon_exe_png}")
    icon_ico = None

if icon_window_png.exists():
    # Copy the window PNG for runtime use
    if not icon_window_local.exists():
        shutil.copy2(icon_window_png, icon_window_local)
        print(f"Copied window icon to {icon_window_local}")
else:
    print(f"Warning: Window icon file not found at {icon_window_png}")

if icon_loading_png.exists():
    # Copy the loading PNG for splash screen
    if not icon_loading_local.exists():
        shutil.copy2(icon_loading_png, icon_loading_local)
        print(f"Copied loading icon to {icon_loading_local}")
else:
    print(f"Warning: Loading icon file not found at {icon_loading_png}")

# Build the executable
build_args = [
    str(project_root / 'src' / 'main.py'),
    '--name=AFS_acquisition',
    '--windowed',  # GUI application (no console window)
    '--onefile',   # Single executable file
    
    # Add all source directories and icons
    f'--add-data={project_root / "src"};src',
    f'--add-data={icon_window_png};.',  # Add window PNG icon to root of exe
    f'--add-data={icon_loading_png};.',  # Add loading PNG for splash screen
    f'--add-data={icon_ico};.',  # Add ICO version for exe file icon
    
    # Hidden imports that PyInstaller might miss
    '--hidden-import=PyQt5',
    '--hidden-import=PyQt5.QtCore',
    '--hidden-import=PyQt5.QtGui',
    '--hidden-import=PyQt5.QtWidgets',
    '--hidden-import=h5py',
    '--hidden-import=h5py.defs',
    '--hidden-import=h5py.utils',
    '--hidden-import=h5py.h5ac',
    '--hidden-import=h5py._proxy',
    '--hidden-import=numpy',
    '--hidden-import=cv2',
    '--hidden-import=matplotlib',
    '--hidden-import=pyvisa',
    '--hidden-import=pyueye',
    '--hidden-import=colorama',
    '--hidden-import=psutil',
    
    # Exclude unnecessary packages to reduce size
    '--exclude-module=tkinter',
    '--exclude-module=unittest',
    '--exclude-module=test',
    
    # Output directory - save to C:\Users\AFS\Documents\Software
    r'--distpath=C:\Users\AFS\Documents\Software',
    f'--workpath={project_root / "build"}',
    f'--specpath={project_root}',
    
    # Clean build
    '--clean',
    '--noconfirm',
]

# Add icon if it was converted successfully
if icon_ico and icon_ico.exists():
    build_args.append(f'--icon={icon_ico}')

PyInstaller.__main__.run(build_args)

print("\n" + "="*60)
print("Build complete!")
print(r"Executable location: C:\Users\AFS\Documents\Software\AFS_acquisition.exe")
print("="*60)
