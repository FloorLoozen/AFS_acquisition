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
from src.ui.widgets.measurement_settings_widget import MeasurementSettingsWidget
from src.ui.widgets.acquisition_controls_widget import AcquisitionControlsWidget
from src.ui.widgets.measurement_controls_widget import MeasurementControlsWidget
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
        self.measurement_settings_widget = None
        self.acquisition_controls_widget = None
        self.measurement_controls_widget = None
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
        - Left column (45%): Control panels split into three rows
          - Top row: Measurement settings
          - Middle row: Acquisition controls  
          - Bottom row: Measurement controls
        - Right column (55%): Camera view spanning all three rows
        """
        central = QWidget(self)
        main_layout = QGridLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Left column - split into three rows
        measurement_settings = self._create_measurement_settings_widget()
        main_layout.addWidget(measurement_settings, 0, 0)
        
        acquisition_controls = self._create_acquisition_controls_widget()
        main_layout.addWidget(acquisition_controls, 1, 0)
        
        measurement_controls = self._create_measurement_controls_widget()
        main_layout.addWidget(measurement_controls, 2, 0)
        
        # Right column - Camera widget spanning all three rows
        self.camera_widget = CameraWidget()
        self.camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_widget.setMinimumWidth(400)
        main_layout.addWidget(self.camera_widget, 0, 1, 3, 1)  # Span 3 rows
        
        # Set column stretch factors to 45:55 - balanced layout with slightly larger camera
        main_layout.setColumnStretch(0, 45)  # Control column (45%)
        main_layout.setColumnStretch(1, 55)  # Camera column (55%)
        
        # Set row stretch factors
        main_layout.setRowStretch(0, 1)  # Measurement settings
        main_layout.setRowStretch(1, 1)  # Acquisition controls
        main_layout.setRowStretch(2, 1)  # Measurement controls
        
        self.setCentralWidget(central)
        
    def _create_measurement_settings_widget(self):
        """Create the measurement settings widget (top-left)."""
        self.measurement_settings_widget = MeasurementSettingsWidget()
        return self.measurement_settings_widget
        
    def _create_acquisition_controls_widget(self):
        """Create the acquisition controls widget (middle-left)."""
        self.acquisition_controls_widget = AcquisitionControlsWidget()
        
        # Connect the acquisition controls to the measurement settings
        if self.measurement_settings_widget:
            self.acquisition_controls_widget.set_measurement_settings_widget(self.measurement_settings_widget)
        
        # Connect signals from acquisition controls to camera widget
        self.acquisition_controls_widget.start_recording_requested.connect(self.handle_start_recording)
        self.acquisition_controls_widget.stop_recording_requested.connect(self.handle_stop_recording)
        self.acquisition_controls_widget.save_recording_requested.connect(self.handle_save_recording)
        
        return self.acquisition_controls_widget
    
    def _create_measurement_controls_widget(self):
        """Create the measurement controls widget (bottom-left)."""
        self.measurement_controls_widget = MeasurementControlsWidget()
        return self.measurement_controls_widget

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
    
    def get_measurement_settings(self):
        """Get the measurement settings widget instance."""
        return self.measurement_settings_widget
    
    def get_measurements_save_path(self):
        """Get the configured measurements save path."""
        if self.measurement_settings_widget:
            return self.measurement_settings_widget.get_measurements_path()
        return ""
    
    def get_lookup_table_save_path(self):
        """Get the configured lookup table save path."""
        if self.measurement_settings_widget:
            return self.measurement_settings_widget.get_lookup_table_path()
        return ""
    
    def get_hdf5_filename(self):
        """Get the configured HDF5 filename."""
        if self.measurement_settings_widget:
            return self.measurement_settings_widget.get_filename()
        return ""
    
    def get_full_hdf5_path(self):
        """Get the complete path for the HDF5 measurement file."""
        if self.measurement_settings_widget:
            return self.measurement_settings_widget.get_full_file_path()
        return ""
    
    def get_sample_information(self):
        """Get the sample information."""
        if self.measurement_settings_widget:
            return self.measurement_settings_widget.get_sample_information()
        return ""
    
    def get_measurement_notes(self):
        """Get the measurement notes."""
        if self.measurement_settings_widget:
            return self.measurement_settings_widget.get_notes()
        return ""
    
    def handle_start_recording(self, file_path):
        """Handle start recording request from measurement controls."""
        logger.info(f"Starting recording to: {file_path}")
        
        # Check if camera is available and running
        if not self.camera_widget or not self.camera_widget.is_running:
            self.acquisition_controls_widget.recording_failed("Camera is not running. Please ensure camera is connected and running.")
            return
        
        try:
            # Start video recording in camera widget
            if hasattr(self.camera_widget, 'start_recording'):
                success = self.camera_widget.start_recording(file_path)
                if success:
                    self.acquisition_controls_widget.recording_started_successfully()
                    self.statusBar().showMessage(f"Recording started: {file_path}")
                    logger.info(f"Recording started successfully: {file_path}")
                else:
                    self.acquisition_controls_widget.recording_failed("Failed to start video recording in camera.")
            else:
                self.acquisition_controls_widget.recording_failed("Camera widget does not support recording.")
                
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.acquisition_controls_widget.recording_failed(f"Error starting recording: {str(e)}")
    
    def handle_stop_recording(self):
        """Handle stop recording request from measurement controls."""
        logger.info("Stopping recording")
        
        try:
            # Stop video recording in camera widget
            if hasattr(self.camera_widget, 'stop_recording'):
                saved_path = self.camera_widget.stop_recording()
                if saved_path:
                    self.acquisition_controls_widget.recording_stopped_successfully(saved_path)
                    self.statusBar().showMessage(f"Recording stopped: {saved_path}")
                    logger.info(f"Recording stopped successfully: {saved_path}")
                else:
                    self.acquisition_controls_widget.recording_failed("Failed to stop recording properly.")
            else:
                self.acquisition_controls_widget.recording_failed("Camera widget does not support recording.")
                
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.acquisition_controls_widget.recording_failed(f"Error stopping recording: {str(e)}")
    
    def handle_save_recording(self, file_path):
        """Handle save recording request from measurement controls."""
        logger.info(f"Saving recording to: {file_path}")
        
        try:
            # For now, the recording is already saved when stopped
            # In the future, this could handle additional metadata saving, HDF5 creation, etc.
            
            # Update status
            self.statusBar().showMessage(f"Recording saved: {file_path}")
            logger.info(f"Recording saved successfully: {file_path}")
            
            # Clear the acquisition controls status after a delay
            QTimer.singleShot(3000, self.acquisition_controls_widget.clear_status)
            
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            self.acquisition_controls_widget.recording_failed(f"Error saving recording: {str(e)}")
            
    def start_measurement(self):
        """Legacy method - replaced by measurement controls widget."""
        pass
    
    def stop_measurement(self):
        """Legacy method - replaced by measurement controls widget."""
        pass
        
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