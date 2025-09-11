from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QMenuBar, QAction, QLabel, QMessageBox, QSizePolicy, 
    QGroupBox, QPushButton, QSlider, QComboBox, QLineEdit, QFormLayout,
    QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor


# Use only absolute imports with src. prefix
from src.logger import get_logger
from src.ui.widgets.camera_widget import CameraWidget
from src.ui.keyboard_shortcuts import KeyboardShortcutManager

logger = get_logger("ui")


class MainWindow(QMainWindow):
    """
    Main application window with camera view, control panels, and menus.
    Features:
    - Automatically maximized window
    - Two-column layout (60% controls, 40% camera)
    - Automatic hardware initialization
    - Simplified menu structure with all tools under one menu
    """

    def __init__(self):
        super().__init__()
        logger.info("Initializing main window")
        self.camera_widget = None
        self.init_ui()
        self.keyboard_shortcuts = KeyboardShortcutManager(self)
        
        # Automatically initialize hardware on startup
        QTimer.singleShot(100, self.initialize_hardware)

    def init_ui(self):
        self.setWindowTitle("AFS Tracking System")
        self.setGeometry(100, 100, 1280, 800)  # Default size if not maximized
        
        # Start application in maximized window
        self.showMaximized()

        self._create_menu_bar()
        self._create_central_layout()

        # Ensure normal cursor is set at startup
        QApplication.restoreOverrideCursor()
        
        self.statusBar().showMessage("Ready")

    def _create_menu_bar(self):
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        # File
        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu - simplified menu structure as requested
        tools_menu = menubar.addMenu("Tools")
        
        # Camera Settings
        camera_action = QAction("Camera Settings", self)
        camera_action.triggered.connect(self.open_camera_settings)
        tools_menu.addAction(camera_action)
        
        # Stage Controller
        stage_action = QAction("Stage Controller", self)
        stage_action.triggered.connect(self.open_stage_controls)
        tools_menu.addAction(stage_action)
        
        # Resonance Finder
        resonance_action = QAction("Resonance Finder", self)
        resonance_action.triggered.connect(self.open_resonance_finder)
        tools_menu.addAction(resonance_action)
        
        # Force Path Maker with better name
        freq_path_action = QAction("Force Path Designer", self)
        freq_path_action.triggered.connect(self.open_frequency_path_designer)
        tools_menu.addAction(freq_path_action)

        # Help
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.open_about)
        help_menu.addAction(about_action)

    def _create_central_layout(self):
        """
        Create the main window layout with a 2-column grid:
        - Left column (60%): Control panels split into two rows
          - Top row: Measurement settings
          - Bottom row: Control buttons
        - Right column (40%): Camera view spanning both rows
        """
        central = QWidget(self)
        main_layout = QGridLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Left column - split into two rows
        measurement_settings = self._create_measurement_settings_widget()
        main_layout.addWidget(measurement_settings, 0, 0)
        
        measurement_controls = self._create_measurement_controls_widget()
        main_layout.addWidget(measurement_controls, 1, 0)
        
        # Right column - Camera widget spanning both rows
        self.camera_widget = CameraWidget()
        self.camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_widget.setMinimumWidth(400)
        main_layout.addWidget(self.camera_widget, 0, 1, 2, 1)  # Span 2 rows
        
        # Set column stretch factors to 60:40
        main_layout.setColumnStretch(0, 3)  # Control column (60%)
        main_layout.setColumnStretch(1, 2)  # Camera column (40%)
        
        # Set row stretch factors
        main_layout.setRowStretch(0, 1)  # Measurement settings
        main_layout.setRowStretch(1, 1)  # Measurement controls
        
        self.setCentralWidget(central)
        
    def _create_measurement_settings_widget(self):
        """Create the measurement settings widget (top-left)."""
        group = QGroupBox("Measurement Settings")
        # Just create an empty layout with no content for now
        layout = QVBoxLayout(group)
        layout.addStretch(1)
        
        return group
        
    def _create_measurement_controls_widget(self):
        """Create the measurement controls widget (bottom-left)."""
        group = QGroupBox("Measurement Controls")
        layout = QVBoxLayout(group)
        
        # Just an empty layout for now
        layout.addStretch(1)
        
        return group

    # Menu handlers
    def open_camera_settings(self):
        QMessageBox.information(self, "Camera Settings", "This feature will be implemented later.")

    def open_stage_controls(self):
        # Open the XY Stage dialog
        try:
            # Always ensure normal cursor before opening the dialog
            QApplication.restoreOverrideCursor()
            
            from src.ui.widgets.xy_stage_widget import XYStageWidget
            
            # Store reference to prevent garbage collection
            self._stage_dialog = getattr(self, "_stage_dialog", None)
            
            if self._stage_dialog is None or not self._stage_dialog.isVisible():
                # Create a new dialog instance
                self._stage_dialog = XYStageWidget(self)
                self._stage_dialog.show()
            else:
                # If dialog already exists, bring it to front
                self._stage_dialog.activateWindow()
                self._stage_dialog.raise_()
        except Exception as e:
            logger.error(f"Failed to open stage dialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open stage controls: {e}")
    
    def open_resonance_finder(self):
        QMessageBox.information(self, "Resonance Finder", "This feature will be implemented later.")
    
    def open_frequency_path_designer(self):
        QMessageBox.information(self, "Force Path Designer", "This feature will be implemented later.")
    
    def open_spectrum_analyzer(self):
        QMessageBox.information(self, "Spectrum Analyzer", "This feature will be implemented later.")
    
    def open_data_export(self):
        QMessageBox.information(self, "Export Data", "This feature will be implemented later.")

    def open_about(self):
        QMessageBox.information(self, "About", "AFS Tracking System v3\n\nAutomated tracking system for AFS using IDS cameras and MCL MicroDrive XY stage hardware.")
        
    def initialize_hardware(self):
        """
        Initialize all hardware components automatically at startup.
        Called with a short delay after the UI is displayed to ensure
        all widgets are properly set up before hardware connection begins.
        """
        logger.info("Initializing hardware components...")
        
        # Update status bar
        self.statusBar().showMessage("Initializing hardware...")
        
        try:
            # Initialize all hardware components in sequence
            # The system will continue even if some components fail to initialize
            self._initialize_camera()
            self._initialize_xy_stage()
            self._initialize_function_generator()
        except Exception as e:
            logger.error(f"Hardware initialization error: {e}")
            self.statusBar().showMessage(f"Hardware initialization error: {str(e)}")
        finally:
            # Always ensure the cursor is restored to normal
            QApplication.restoreOverrideCursor()
        
        # Update status after a short delay
        # This gives the user time to see the status message
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Hardware initialization complete"))
    
    def _initialize_camera(self):
        """Initialize camera hardware."""
        logger.info("Initializing camera...")
        
        # Show camera-specific waiting message
        self.statusBar().showMessage("Initializing camera... Please wait")
        
        try:
            # Camera widget should auto-initialize, just check its status
            if self.camera_widget:
                # Camera widget exists but may not be fully initialized yet
                if hasattr(self.camera_widget, 'is_running') and self.camera_widget.is_running:
                    # Camera is already running
                    if hasattr(self.camera_widget, 'use_test_pattern') and self.camera_widget.use_test_pattern:
                        # Camera is in test pattern mode
                        logger.info("Camera initialized in test pattern mode")
                        self.statusBar().showMessage("Camera running in test pattern mode")
                    else:
                        # Camera is running with actual hardware
                        logger.info("Camera hardware initialized successfully")
                        self.statusBar().showMessage("Camera hardware initialized")
                else:
                    # Camera widget exists but hasn't started running yet
                    logger.info("Camera widget starting initialization process")
                    self.statusBar().showMessage("Camera initializing")
            else:
                # Camera widget doesn't exist at all
                logger.warning("Camera widget creation failed")
                self.statusBar().showMessage("Camera initialization failed, using fallback")
        finally:
            # Ensure cursor is normal
            QApplication.restoreOverrideCursor()
            
    def _initialize_xy_stage(self):
        """Initialize XY stage hardware."""
        logger.info("Initializing XY stage...")
        self.statusBar().showMessage("Initializing XY stage...")
        
        # Will be implemented later
        try:
            # Stage controller initialization
            pass
        except Exception as e:
            logger.error(f"XY stage initialization error: {e}")
            logger.info("Continuing with limited functionality")
            self.statusBar().showMessage("XY stage initialization failed")
        else:
            self.statusBar().showMessage("XY stage initialization complete")
    
    def _initialize_function_generator(self):
        """Initialize function generator hardware."""
        logger.info("Initializing function generator...")
        self.statusBar().showMessage("Initializing function generator...")
        
        # Will be implemented later
        try:
            # Function generator initialization
            pass
        except Exception as e:
            logger.error(f"Function generator initialization error: {e}")
            logger.info("Continuing with limited functionality")
            self.statusBar().showMessage("Function generator initialization failed")
        else:
            self.statusBar().showMessage("Function generator initialization complete")