"""Main application entry point for AFS Tracking System."""

import sys
import os
from typing import NoReturn
from PyQt5.QtWidgets import QApplication

# Add the project root directory to the Python path to enable imports
# regardless of whether the script is run directly or as a module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ui.main_window import MainWindow
from src.utils.logger import get_logger

logger = get_logger("main")


def main() -> NoReturn:
    """Initialize and run the AFS Tracking application.
    
    Creates the PyQt5 application instance, initializes the main window,
    and starts the event loop. Exits when the application is closed.
    
    Raises:
        SystemExit: Always exits with the application return code.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    rc = app.exec_()
    logger.info(f"Application exited with code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()