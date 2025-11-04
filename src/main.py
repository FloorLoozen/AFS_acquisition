"""
Main application entry point for AFS Acquisition.

This module provides the main entry point for the AFS (Acoustic Force Spectroscopy)
Acquisition, a comprehensive laboratory instrumentation control and data
acquisition software. The system interfaces with cameras, stages, function
generators, and oscilloscopes to provide real-time experimental control and
data recording capabilities.

The application is built on PyQt5 and provides:
- Real-time camera feed and recording to HDF5 format
- Precision XY stage control with automated positioning
- Function generator control for frequency sweeps and resonance detection
- Oscilloscope integration for signal monitoring
- Force path design and execution for automated measurements
- Comprehensive data logging and experimental metadata storage

Dependencies:
    - PyQt5: GUI framework
    - NumPy: Numerical computations
    - OpenCV: Image processing
    - H5PY: HDF5 data storage
    - PyVISA: Instrument communication
    - matplotlib: Data visualization

Authors:
    Floor Loozen - Primary developer and maintainer

Version:
    3.0.0 - Major rewrite with improved architecture and HDF5 integration
"""

import sys
import os
import warnings
from typing import NoReturn, Optional

# Suppress syntax warnings from pyueye library (harmless documentation issues)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pyueye")

# Application metadata
__version__ = "3.0.0"
__author__ = "Floor Loozen"
__description__ = "AFS Acquisition - Laboratory instrumentation control and data acquisition"


def setup_python_path() -> None:
    """
    Setup Python path for module imports.
    
    Ensures that the project root directory is in the Python path so that
    modules can be imported correctly regardless of the current working directory.
    This is essential for proper module resolution when the application is
    launched from different locations.
    
    Raises:
        OSError: If the project directory structure is invalid.
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists(project_root):
            raise OSError(f"Project root directory not found: {project_root}")
        
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    except Exception as e:
        print(f"Error setting up Python path: {e}")
        sys.exit(1)


def create_application() -> 'QApplication':
    """
    Create and configure the PyQt5 application.
    
    Initializes the PyQt5 QApplication with appropriate settings for high-DPI
    displays and sets application metadata. This ensures consistent appearance
    across different display configurations and provides proper application
    identification for the operating system.
    
    Returns:
        QApplication: Configured PyQt5 application instance.
        
    Raises:
        ImportError: If PyQt5 is not available.
        RuntimeError: If application creation fails.
    """
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        
        # Enable high DPI support for modern displays
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        app = QApplication(sys.argv)
        
        # Set application metadata for system integration
        app.setApplicationName("AFS Acquisition")
        app.setApplicationVersion(__version__)
        app.setOrganizationName("AFS Lab")
        app.setApplicationDisplayName("AFS Acquisition")

        return app
        
    except ImportError as e:
        raise ImportError(f"PyQt5 not available: {e}. Please install with: pip install PyQt5") from e
    except Exception as e:
        raise RuntimeError(f"Failed to create application: {e}") from e


def main() -> NoReturn:
    """
    Initialize and run the AFS Acquisition application.
    
    This is the main entry point that orchestrates the application startup:
    1. Sets up the Python import path
    2. Initializes logging system
    3. Loads configuration
    4. Creates the PyQt5 application
    5. Instantiates and shows the main window
    6. Starts the event loop
    7. Handles cleanup on exit
    
    The function includes comprehensive error handling to provide meaningful
    error messages for common failure modes such as missing dependencies,
    configuration errors, or hardware initialization problems.
    
    Raises:
        SystemExit: Always called at the end with appropriate exit code.
    """
    exit_code = 0
    logger: Optional['Logger'] = None
    
    try:
        # Setup environment
        setup_python_path()
        
        # Initialize logging first (critical for error reporting)
        from src.utils.logger import get_logger
        logger = get_logger("main")
        logger.info(f"Starting {__description__} v{__version__}")
        
        # Check system requirements
        _check_system_requirements(logger)
        
        # Initialize configuration
        from src.utils.config_manager import get_config
        config = get_config()
        
        # Create PyQt5 application
        app = create_application()
        
        # Import and create main window
        from src.ui.main_window import MainWindow
        window = MainWindow()
        
        # Start in maximized mode for better visibility
        window.showMaximized()
        
        # Start the Qt event loop
        exit_code = app.exec_()
        
        # Save configuration on normal exit
        try:
            config.save_config()
        except Exception as e:
            logger.warning(f"Failed to save configuration: {e}")
        
    except ImportError as e:
        error_msg = (
            f"Import error: {e}\n"
            f"Please ensure all dependencies are installed:\n"
            f"  pip install -r requirements.txt\n"
            f"Required packages: PyQt5, numpy, opencv-python, h5py, pyvisa, matplotlib"
        )
        if logger:
            logger.error(error_msg)
        else:
            print(f"CRITICAL: {error_msg}")
        exit_code = 1
        
    except Exception as e:
        error_msg = f"Critical error during startup: {e}"
        if logger:
            logger.critical(error_msg)
            import traceback
            logger.critical(f"Full traceback:\n{traceback.format_exc()}")
        else:
            print(f"CRITICAL: {error_msg}")
            import traceback
            traceback.print_exc()
        exit_code = 1
        
    finally:
        if logger:
            logger.info(f"Exiting (code {exit_code})")
        sys.exit(exit_code)


def _check_system_requirements(logger: 'Logger') -> None:
    """
    Check system requirements and log warnings for potential issues.
    
    Args:
        logger: Logger instance for reporting issues.
        
    Note:
        This function logs warnings but does not prevent startup, allowing
        the application to attempt to run even with suboptimal conditions.
    """
    import platform
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.warning(f"Python 3.8+ recommended, but {sys.version} found")
    
    # Check operating system
    os_name = platform.system()
    if os_name not in ["Windows", "Linux", "Darwin"]:
        logger.warning(f"Untested operating system: {os_name}")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            logger.warning(f"Low system memory: {memory_gb:.1f} GB (4+ GB recommended)")
    except ImportError:
        pass  # psutil not available


if __name__ == "__main__":
    main()