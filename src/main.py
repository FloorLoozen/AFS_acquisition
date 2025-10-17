"""Main application entry point for AFS Tracking System."""

import sys
import os
import warnings
from typing import NoReturn

# Suppress syntax warnings from pyueye library (harmless documentation issues)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pyueye")


def setup_python_path() -> None:
    """Setup Python path for module imports."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def create_application() -> 'QApplication':
    """Create and configure the PyQt5 application."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    
    # Enable high DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("AFS Tracking System")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("AFS Lab")
    
    return app


def main() -> NoReturn:
    """Initialize and run the AFS Tracking application."""
    # Setup environment
    setup_python_path()
    
    # Initialize logging first
    from src.utils.logger import get_logger
    logger = get_logger("main")
    
    try:
        # Initialize configuration
        from src.utils.config_manager import get_config
        config = get_config()
        
        # Create application
        app = create_application()
        
        # Import and create main window
        from src.ui.main_window import MainWindow
        window = MainWindow()
        
        # Start in maximized mode
        window.showMaximized()
        logger.info("AFS Tracking System started successfully")
        
        # Start event loop
        rc = app.exec_()
        
        # Save configuration on exit
        config.save_config()
        
        logger.info(f"Application exited with code {rc}")
        sys.exit(rc)
        
    except ImportError as e:
        error_msg = f"Import error: {e}\nPlease ensure all dependencies are installed: pip install -r requirements.txt"
        try:
            logger.error(error_msg)
        except:
            print(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"Critical error during startup: {e}"
        try:
            logger.error(error_msg)
            print(error_msg)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        except:
            print(error_msg)
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()