"""AFS Acquisition - Main application entry point.

This module serves as the entry point for the AFS Acquisition application.
It handles application initialization, configuration, and the main event loop.

Usage:
    python src/main.py

The application will:
    1. Setup the Python path for imports
    2. Initialize logging and configuration
    3. Create and display the main window
    4. Run the Qt event loop
    5. Save configuration on exit
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pyueye")

__version__ = "1.0.0"


def main():
    """Initialize and run the AFS Acquisition application.
    
    This function:
    - Sets up the Python path for package imports
    - Configures PyQt5 for high DPI displays
    - Initializes logging and configuration
    - Creates and shows the main application window
    - Runs the Qt event loop
    - Handles graceful shutdown and config saving
    
    Returns:
        System exit code (0 for success, non-zero for errors)
    """
    # Setup Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from src.utils.logger import get_logger
    from src.utils.config_manager import get_config
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt, QLocale
    
    logger = get_logger("main")
    logger.info(f"Starting AFS Acquisition v{__version__}")
    
    # Configure PyQt5 for high DPI displays (Windows 11)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Set locale to use dot as decimal separator (not comma)
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    app.setApplicationName("AFS Acquisition")
    app.setApplicationVersion(__version__)
    
    # Load configuration
    config = get_config()
    
    # Create and show main window
    try:
        from src.ui.main_window import MainWindow
        window = MainWindow()
        window.showMaximized()
    except Exception as e:
        logger.critical(f"Failed to create main window: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        return 1
    
    # Run application event loop
    exit_code = app.exec_()
    
    # Save config on exit
    try:
        config.save_config()
        logger.info(f"Exiting (code {exit_code})")
    except Exception as e:
        logger.error(f"Error saving configuration on exit: {e}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()