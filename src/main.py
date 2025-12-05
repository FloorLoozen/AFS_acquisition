"""AFS Acquisition - Main application entry point."""

import sys
import os
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pyueye")

__version__ = "1.0.0"


def main():
    """Initialize and run the AFS Acquisition application."""
    # Setup Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from src.utils.logger import get_logger
    from src.utils.config_manager import get_config
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt, QLocale
    
    logger = get_logger("main")
    logger.info(f"Starting AFS Acquisition v{__version__}")
    
    # Configure PyQt5 for high DPI
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
    from src.ui.main_window import MainWindow
    window = MainWindow()
    window.showMaximized()
    
    # Run application
    exit_code = app.exec_()
    
    # Save config on exit
    config.save_config()
    logger.info(f"Exiting (code {exit_code})")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()