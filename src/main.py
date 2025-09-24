import sys
import os
from PyQt5.QtWidgets import QApplication

# Add the project root directory to the Python path to enable imports
# regardless of whether the script is run directly or as a module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now imports should work correctly

from src.ui.main_window import MainWindow
from src.utils.logger import get_logger

logger = get_logger("main")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    rc = app.exec_()
    logger.info(f"Application exited with code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()