from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QMenuBar, QAction, 
    QMessageBox, QSizePolicy, QApplication
)
from PyQt5.QtCore import QTimer

from src.utils.logger import get_logger
from src.ui.widgets.camera_widget import CameraWidget
from src.ui.widgets.measurement_settings_widget import MeasurementSettingsWidget
from src.ui.widgets.acquisition_controls_widget import AcquisitionControlsWidget
from src.ui.widgets.measurement_controls_widget import MeasurementControlsWidget
from src.utils.keyboard_shortcuts import KeyboardShortcutManager

logger = get_logger("ui")


class MainWindow(QMainWindow):
    """Main application window with 3-row layout and camera view."""

    def __init__(self):
        super().__init__()
        logger.info("AFS Tracking started")
        
        # Initialize widget references
        self.camera_widget = None
        self.measurement_settings_widget = None
        self.acquisition_controls_widget = None
        self.measurement_controls_widget = None
        
        self._init_ui()
        self.keyboard_shortcuts = KeyboardShortcutManager(self)
        
        # Auto-initialize hardware after UI is ready
        QTimer.singleShot(100, self._initialize_hardware)

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AFS Tracking System")
        self.setGeometry(100, 100, 1280, 800)
        self.showMaximized()

        self._create_menu_bar()
        self._create_central_layout()

        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Ready")

    def _create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        self._add_action(file_menu, "Exit", "Ctrl+Q", self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        self._add_action(tools_menu, "Camera Settings", None, self._show_not_implemented)
        self._add_action(tools_menu, "Stage Controller", None, self._open_stage_controls)
        self._add_action(tools_menu, "Resonance Finder", None, self._show_not_implemented)
        self._add_action(tools_menu, "Force Path Designer", None, self._show_not_implemented)

        # Help menu
        help_menu = menubar.addMenu("Help")
        self._add_action(help_menu, "About", None, self._open_about)

    def _add_action(self, menu, text, shortcut, callback):
        """Helper to add menu actions."""
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        menu.addAction(action)

    def _create_central_layout(self):
        """Create main layout: left column (3 rows of controls) + right column (camera)."""
        central = QWidget(self)
        layout = QGridLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # Left column widgets
        self._create_measurement_settings_widget(layout, 0, 0)
        self._create_acquisition_controls_widget(layout, 1, 0)
        self._create_measurement_controls_widget(layout, 2, 0)
        
        # Right column camera (spans all 3 rows)
        self.camera_widget = CameraWidget()
        self.camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_widget.setMinimumWidth(400)
        layout.addWidget(self.camera_widget, 0, 1, 3, 1)
        
        # Set proportions: 45% controls, 55% camera
        layout.setColumnStretch(0, 45)
        layout.setColumnStretch(1, 55)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 1)
        
        self.setCentralWidget(central)
        
    def _create_measurement_settings_widget(self, layout, row, col):
        """Create and add measurement settings widget."""
        self.measurement_settings_widget = MeasurementSettingsWidget()
        layout.addWidget(self.measurement_settings_widget, row, col)
        
    def _create_acquisition_controls_widget(self, layout, row, col):
        """Create and add acquisition controls widget."""
        self.acquisition_controls_widget = AcquisitionControlsWidget()
        
        # Set measurement settings reference
        if self.measurement_settings_widget:
            self.acquisition_controls_widget.set_measurement_settings_widget(self.measurement_settings_widget)
        
        # Connect recording signals
        self.acquisition_controls_widget.start_recording_requested.connect(self._handle_start_recording)
        self.acquisition_controls_widget.stop_recording_requested.connect(self._handle_stop_recording)
        self.acquisition_controls_widget.save_recording_requested.connect(self._handle_save_recording)
        
        layout.addWidget(self.acquisition_controls_widget, row, col)
    
    def _create_measurement_controls_widget(self, layout, row, col):
        """Create and add measurement controls widget."""
        self.measurement_controls_widget = MeasurementControlsWidget()
        layout.addWidget(self.measurement_controls_widget, row, col)

    # Menu handlers
    def _show_not_implemented(self):
        """Show not implemented message."""
        QMessageBox.information(self, "Not Implemented", "This feature will be implemented later.")

    def _open_stage_controls(self):
        """Open XY Stage control dialog."""
        try:
            QApplication.restoreOverrideCursor()
            
            from src.ui.widgets.xy_stage_widget import XYStageWidget
            
            # Store reference to prevent garbage collection
            if not hasattr(self, '_stage_dialog') or not self._stage_dialog.isVisible():
                self._stage_dialog = XYStageWidget(self)
                self._stage_dialog.show()
            else:
                self._stage_dialog.activateWindow()
                self._stage_dialog.raise_()
        except Exception as e:
            logger.error(f"Failed to open stage dialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open stage controls: {e}")

    def _open_about(self):
        """Show about dialog."""
        QMessageBox.information(self, "About", 
            "AFS Tracking System v3\n\n"
            "Automated tracking system for AFS using IDS cameras "
            "and MCL MicroDrive XY stage hardware.")
    
    def get_measurement_settings(self):
        """Get the measurement settings widget instance."""
        return self.measurement_settings_widget
    
    def get_measurements_save_path(self):
        """Get the configured save path (compatibility method)."""
        if self.measurement_settings_widget:
            return self.measurement_settings_widget.get_save_path()
        return ""
    
    def get_save_path(self):
        """Get the configured save path."""
        if self.measurement_settings_widget:
            return self.measurement_settings_widget.get_save_path()
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
    
    def _handle_start_recording(self, file_path):
        """Handle start recording request."""
# Recording start logged by acquisition controls
        
        if not self.camera_widget or not self.camera_widget.is_running:
            self.acquisition_controls_widget.recording_failed(
                "Camera is not running. Please ensure camera is connected and running.")
            return
        
        try:
            if hasattr(self.camera_widget, 'start_recording'):
                success = self.camera_widget.start_recording(file_path)
                if success:
                    self.acquisition_controls_widget.recording_started_successfully()
                    self.statusBar().showMessage(f"Recording started: {file_path}")
# Success already logged by acquisition controls
                else:
                    self.acquisition_controls_widget.recording_failed("Failed to start video recording in camera.")
            else:
                self.acquisition_controls_widget.recording_failed("Camera widget does not support recording.")
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.acquisition_controls_widget.recording_failed(f"Error starting recording: {str(e)}")
    
    def _handle_stop_recording(self):
        """Handle stop recording request."""
# Stop logged by acquisition controls
        
        try:
            if hasattr(self.camera_widget, 'stop_recording'):
                saved_path = self.camera_widget.stop_recording()
                if saved_path:
                    self.acquisition_controls_widget.recording_stopped_successfully(saved_path)
                    self.statusBar().showMessage(f"Recording stopped: {saved_path}")
# Success already logged by acquisition controls
                else:
                    self.acquisition_controls_widget.recording_failed("Failed to stop recording properly.")
            else:
                self.acquisition_controls_widget.recording_failed("Camera widget does not support recording.")
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.acquisition_controls_widget.recording_failed(f"Error stopping recording: {str(e)}")
    
    def _handle_save_recording(self, file_path):
        """Handle save recording request."""
        import os
        import shutil
        
# Save result will be logged
        
        try:
            # Get the original recorded file path for potential renaming
            if hasattr(self.acquisition_controls_widget, 'original_recording_path'):
                actual_recorded_path = self.acquisition_controls_widget.original_recording_path
                if actual_recorded_path and actual_recorded_path != file_path and os.path.exists(actual_recorded_path):
                    # Need to move/rename the file to the desired path
                    shutil.move(actual_recorded_path, file_path)
# Rename handled silently
                elif not os.path.exists(file_path):
                    # File doesn't exist - this shouldn't happen in auto-save
                    logger.warning(f"Expected recording file not found: {file_path}")
            
            self.statusBar().showMessage(f"Recording saved: {file_path}")
# Save logged by acquisition controls
            
            # Clear status after delay
            QTimer.singleShot(3000, self.acquisition_controls_widget.clear_status)
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            self.acquisition_controls_widget.recording_failed(f"Error saving recording: {str(e)}")
        
    def _initialize_hardware(self):
        """Initialize all hardware components at startup."""
# Hardware initialization starts
        self.statusBar().showMessage("Initializing hardware...")
        
        try:
            self._init_camera()
            self._init_xy_stage() 
            self._init_function_generator()
        except Exception as e:
            logger.error(f"Hardware initialization error: {e}")
            self.statusBar().showMessage(f"Hardware initialization error: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
        
        # Show completion message after delay
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Hardware initialization complete"))
    
    def _init_camera(self):
        """Initialize camera hardware."""
# Camera initialization
        self.statusBar().showMessage("Initializing camera... Please wait")
        
        try:
            if self.camera_widget and hasattr(self.camera_widget, 'is_running'):
                if self.camera_widget.is_running:
                    mode = "test pattern" if getattr(self.camera_widget, 'use_test_pattern', False) else "hardware"
# Camera mode info shown in status bar
                    self.statusBar().showMessage(f"Camera running in {mode} mode")
                else:
                    logger.info("Camera widget starting initialization")
                    self.statusBar().showMessage("Camera initializing")
            else:
                logger.warning("Camera widget creation failed")
                self.statusBar().showMessage("Camera initialization failed, using fallback")
        finally:
            QApplication.restoreOverrideCursor()
            
    def _init_xy_stage(self):
        """Initialize XY stage hardware."""
        try:
            from src.controllers.stage_manager import StageManager
            stage_manager = StageManager.get_instance()
            if stage_manager.connect():
                logger.info("XY stage connected")
            else:
                logger.warning("XY stage connection failed")
        except Exception as e:
            logger.warning(f"XY stage initialization failed: {e}")
    
    def _init_function_generator(self):
        """Initialize function generator hardware (placeholder).""" 
# Function generator initialization placeholder