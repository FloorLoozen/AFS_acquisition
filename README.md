# AFS Tracking

Automated tracking system for AFS using IDS cameras and MCL MicroDrive XY stage hardware.

## Installation & Quick Start

1. **Clone the repository**
	```powershell
	git clone https://github.com/FloorLoozen/AFS_tracking
	cd AFS_tracking
	```

2. **Create and activate a virtual environment**
	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```

3. **Install dependencies**
	```powershell
	pip install --upgrade pip
	pip install --force-reinstall -r requirements.txt
	pip install pyueye
	```

4. **Run the application**
	```powershell
	python -m src.main
	```

**Note:**
- Use the `--ignore-venv` flag to run without virtual environment checks (e.g., `python -m src.main --ignore-venv`)
- Always run commands from the project root directory (`AFS_tracking`).
- Ensure your Python version is compatible with all dependencies (Python 3.10–3.12 recommended).
- Hardware features require IDS uEye camera and MCL MicroDrive XY stage with appropriate drivers.

## Features

- **Camera Control**: Live feed from IDS uEye cameras with auto-reconnect
- **XY Stage Control**: Precise positioning using MCL MicroDrive hardware
- **Tools**: Camera settings, Stage controller, Resonance finder, Force path designer
- **User Interface**: PyQt5-based GUI with keyboard shortcuts and maximized window
- **Logging**: Color-coded console output and file logging

## Dependencies

- PyQt5 (v5.15.10): UI framework
- pyueye (v4.1.0): IDS camera interface
- OpenCV (v4.9.0): Image processing
- NumPy (v2.3.2): Numerical operations
- colorama (v0.4.6): Colored console output

## Hardware Requirements

- IDS uEye camera
- MCL MicroDrive XY stage

_Last updated: 2025-09-11_


