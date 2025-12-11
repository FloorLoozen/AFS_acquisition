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
    from PyQt5.QtWidgets import QApplication, QSplashScreen
    from PyQt5.QtCore import Qt, QLocale, QRect
    from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QColor, QPainterPath
    from pathlib import Path
    
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
    
    # Custom splash screen class with outlined text
    class CustomSplashScreen(QSplashScreen):
        def __init__(self, pixmap):
            super().__init__(pixmap, Qt.WindowStaysOnTopHint)
            self.message = ""
            
        def showMessage(self, message, alignment=Qt.AlignCenter, color=QColor("#004667")):
            self.message = message
            self.repaint()
            
        def drawContents(self, painter):
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Set font
            font = QFont()
            font.setPointSize(32)
            font.setBold(True)
            painter.setFont(font)
            
            # Create text path for outlined text
            path = QPainterPath()
            rect = self.rect()
            path.addText(rect.center().x() - 80, rect.center().y() + 15, font, self.message)
            
            # Draw outline (light blue)
            painter.setPen(QPen(QColor("#A6CAEC"), 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
            
            # Draw inner text (dark blue)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#004667"))
            painter.drawPath(path)
    
    # Show splash screen with loading icon
    splash = None
    loading_icon_paths = [
        r"C:\Users\AFS\Documents\Software\Icons\acquistion_loading.png",  # Standalone exe location
        Path(project_root) / "AFS_loading.png",  # Development location
    ]
    
    for icon_path in loading_icon_paths:
        if Path(icon_path).exists():
            pixmap = QPixmap(str(icon_path))
            # Scale icon to smaller splash screen size
            pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)  # Use regular QSplashScreen, no text needed
            splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
            splash.show()
            # No showMessage call - text is already in the image
            app.processEvents()
            logger.debug(f"Splash screen shown with icon from: {icon_path}")
            break
    
    # Load configuration
    config = get_config()
    
    # Create and show main window
    try:
        from src.ui.main_window import MainWindow
        window = MainWindow()
        
        # Close splash screen before showing main window
        if splash:
            splash.finish(window)
        
        window.showMaximized()
    except Exception as e:
        logger.critical(f"Failed to create main window: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        if splash:
            splash.close()
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