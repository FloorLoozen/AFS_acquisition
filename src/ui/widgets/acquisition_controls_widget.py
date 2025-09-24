"""
Acquisition Controls Widget for the AFS Tracking System.
Provides Start, Stop, and Save recording controls for data acquisition.
"""

import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, 
    QMessageBox
)
from PyQt5.QtCore import pyqtSignal

from src.logger import get_logger
from src.ui.components.status_display import StatusDisplay

logger = get_logger("acquisition_controls")


class AcquisitionControlsWidget(QGroupBox):
    """Widget for data acquisition recording controls with Start, Stop, and Save buttons."""
    
    # Signals to communicate with main window and camera widget
    start_recording_requested = pyqtSignal(str)  # Emits the file path
    stop_recording_requested = pyqtSignal()
    save_recording_requested = pyqtSignal(str)  # Emits the save path

    def __init__(self, parent=None):
        super().__init__("Acquisition Controls", parent)
        
        logger.info("Initializing AcquisitionControlsWidget")
        
        # Recording state
        self.is_recording = False
        self.current_recording_path = ""
        self.measurement_settings_widget = None  # Will be set from main window
        
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Button and status layout
        controls_layout = QHBoxLayout()
        
        # Start measurement button
        self.start_btn = QPushButton("🔴 Start Recording")
        self.start_btn.clicked.connect(self.start_recording)
        
        # Stop measurement button
        self.stop_btn = QPushButton("⏹️ Stop Recording")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)  # Initially disabled
        
        # Save measurement button
        self.save_btn = QPushButton("💾 Save Recording")
        self.save_btn.clicked.connect(self.save_recording)
        self.save_btn.setEnabled(False)  # Initially disabled
        
        # Status display for recording feedback (next to buttons)
        self.status_display = StatusDisplay()
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addWidget(self.status_display, 1)  # Give status display more space
        
        main_layout.addLayout(controls_layout)
        main_layout.addStretch(1)

    def set_measurement_settings_widget(self, widget):
        """Set reference to the measurement settings widget to check paths."""
        self.measurement_settings_widget = widget
        # Update save button state based on current settings
        self.update_save_button_state()

    def update_save_button_state(self):
        """Update the save button enabled state based on path configuration."""
        if self.measurement_settings_widget:
            has_valid_path = (
                self.measurement_settings_widget.is_configured() and 
                bool(self.measurement_settings_widget.get_measurements_path())
            )
            # Save button is enabled if we have a valid path AND we're not currently recording
            self.save_btn.setEnabled(has_valid_path and not self.is_recording)
        else:
            self.save_btn.setEnabled(False)

    def start_recording(self):
        """Start recording measurement data and video."""
        logger.info("Start recording button clicked")
        
        # Check if measurement settings are configured
        if not self.measurement_settings_widget or not self.measurement_settings_widget.is_configured():
            QMessageBox.warning(self, "Configuration Required", 
                              "Please configure measurement and lookup table save paths before starting recording.")
            return
        
        # Get the full file path for the measurement
        if not self.measurement_settings_widget.get_measurements_path():
            QMessageBox.warning(self, "Invalid Path", 
                              "Could not determine measurement file path. Please check your settings.")
            return
        
        # Generate the full path for recording
        filename = self.measurement_settings_widget.get_filename()
        measurements_path = self.measurement_settings_widget.get_measurements_path()
        full_path = os.path.join(measurements_path, filename)
        
        try:
            # Emit signal to start recording
            self.start_recording_requested.emit(full_path)
            
            # Update internal state
            self.is_recording = True
            self.current_recording_path = full_path
            
            # Update button states
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.update_save_button_state()  # This will disable save during recording
            
            # Update status
            self.status_display.set_status("Recording")
            
            logger.info(f"Recording started - target path: {full_path}")
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            QMessageBox.critical(self, "Recording Error", 
                               f"Failed to start recording:\n{str(e)}")

    def stop_recording(self):
        """Stop recording measurement data."""
        logger.info("Stop recording button clicked")
        
        try:
            # Emit signal to stop recording
            self.stop_recording_requested.emit()
            
            # Update internal state
            self.is_recording = False
            
            # Update button states
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.update_save_button_state()  # This will enable save if path is valid
            
            # Update status
            self.status_display.set_status("Stopped")
            
            logger.info("Recording stopped")
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            QMessageBox.critical(self, "Stop Recording Error", 
                               f"Failed to stop recording:\n{str(e)}")

    def save_recording(self):
        """Save the recorded measurement data."""
        logger.info("Save recording button clicked")
        
        if not self.current_recording_path:
            QMessageBox.warning(self, "No Recording", 
                              "No recording to save. Please record something first.")
            return
        
        if not self.measurement_settings_widget or not self.measurement_settings_widget.get_measurements_path():
            QMessageBox.warning(self, "No Save Path", 
                              "No valid save path configured. Please check your measurement settings.")
            return
        
        try:
            # Emit signal to save recording
            save_path = self.current_recording_path
            self.save_recording_requested.emit(save_path)
            
            # Update status
            self.status_display.set_status("Saved")
            
            # Show success message
            QMessageBox.information(self, "Recording Saved", 
                                  f"Recording saved successfully to:\n{save_path}")
            
            # Reset for next recording
            self.current_recording_path = ""
            
            logger.info(f"Recording saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            QMessageBox.critical(self, "Save Error", 
                               f"Failed to save recording:\n{str(e)}")

    def recording_started_successfully(self):
        """Called when recording actually starts successfully in the camera."""
        logger.info("Recording confirmed as started successfully")
        # Status is already set in start_recording, but we could update here if needed

    def recording_stopped_successfully(self, saved_path=None):
        """Called when recording stops successfully in the camera."""
        logger.info(f"Recording confirmed as stopped successfully, path: {saved_path}")
        if saved_path:
            self.current_recording_path = saved_path

    def recording_failed(self, error_message):
        """Called when recording fails."""
        logger.error(f"Recording failed: {error_message}")
        
        # Reset state
        self.is_recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_save_button_state()
        
        # Update status
        self.status_display.set_status("Error")

    def clear_status(self):
        """Clear the status display."""
        self.status_display.clear()

    def get_recording_state(self):
        """Get the current recording state."""
        return {
            'is_recording': self.is_recording,
            'current_path': self.current_recording_path
        }


# Example usage if this file is run directly
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    widget = AcquisitionControlsWidget()
    widget.show()
    sys.exit(app.exec_())