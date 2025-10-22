"""
Acquisition Controls Widget for AFS Acquisition.
Provides Start, Stop, and Save recording controls for data acquisition.
"""

import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, 
    QMessageBox, QFrame, QWidget
)
from PyQt5.QtCore import pyqtSignal, QTimer

from src.utils.logger import get_logger
from src.utils.status_display import StatusDisplay

logger = get_logger("acquisition_controls")


class AcquisitionControlsWidget(QGroupBox):
    """Widget for data acquisition recording controls with Start, Stop, and Save buttons."""
    
    # Signals to communicate with main window and camera widget
    start_recording_requested = pyqtSignal(str)  # Emits the file path
    stop_recording_requested = pyqtSignal()
    save_recording_requested = pyqtSignal(str)  # Emits the save path

    def __init__(self, parent=None):
        super().__init__("Acquisition Controls", parent)
        
# Removed verbose initialization logging
        
        # Recording state
        self.is_recording = False
        self.current_recording_path = ""
        self.original_recording_path = ""  # Store original path for renaming
        self.frequency_settings_widget = None  # Will be set from main window
        
        # Timers for automatic saving workflow
        self.auto_save_timer = QTimer()
        self.auto_save_timer.setSingleShot(True)
        self.auto_save_timer.timeout.connect(self._auto_save_recording)
        
        self.status_clear_timer = QTimer()
        self.status_clear_timer.setSingleShot(True)
        self.status_clear_timer.timeout.connect(self._clear_saved_status)
        
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Create controls frame for consistent styling
        controls_frame = self._create_controls_frame()
        main_layout.addWidget(controls_frame)

    def _create_controls_frame(self):
        """Create the main controls frame with consistent styling."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Add controls section
        controls_section = self._create_controls_section()
        layout.addWidget(controls_section)
        layout.addStretch(1)
        
        return frame

    def _create_controls_section(self):
        """Create the recording controls section."""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setSpacing(8)  # Consistent spacing between rows
        
        # Button and status layout
        controls_layout = QHBoxLayout()
        
        # Start measurement button
        self.start_btn = QPushButton("🔴 Start Recording")
        self.start_btn.clicked.connect(self.start_recording)
        
        # Stop measurement button
        self.stop_btn = QPushButton("⏹️ Stop Recording")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)  # Initially disabled
        
        # Save measurement button (now hidden since saving is automatic)
        self.save_btn = QPushButton("💾 Save Recording")
        self.save_btn.clicked.connect(self.save_recording)
        self.save_btn.setEnabled(False)  # Initially disabled
        self.save_btn.hide()  # Hide since saving is now automatic
        
        # Status display for recording feedback (next to buttons)
        self.status_display = StatusDisplay()
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        # Save button is hidden since saving is automatic
        controls_layout.addWidget(self.status_display, 1)  # Give status display more space
        
        layout.addLayout(controls_layout)
        
        return section

    def set_frequency_settings_widget(self, widget):
        """Set reference to the frequency settings widget to check paths."""
        self.frequency_settings_widget = widget
        # Update save button state based on current settings
        self.update_save_button_state()

    def update_save_button_state(self):
        """Update the save button enabled state based on path configuration."""
        if self.frequency_settings_widget:
            has_valid_path = (
                self.frequency_settings_widget.is_configured() and 
                bool(self.frequency_settings_widget.get_save_path())
            )
            # Save button is enabled if we have a valid path AND we're not currently recording
            self.save_btn.setEnabled(has_valid_path and not self.is_recording)
        else:
            self.save_btn.setEnabled(False)

    def start_recording(self):
        """Start recording measurement data and video."""
        logger.info("Recording started")
        
        # Check if frequency settings are configured
        if not self.frequency_settings_widget or not self.frequency_settings_widget.is_configured():
            QMessageBox.warning(self, "Configuration Required", 
                              "Please configure save path before starting recording.")
            return
        
        # Get the full file path for the measurement
        if not self.frequency_settings_widget.get_save_path():
            QMessageBox.warning(self, "Invalid Path", 
                              "Could not determine save file path. Please check your settings.")
            return
        
        # Generate the full path for recording
        filename = self.frequency_settings_widget.get_filename()
        save_path = self.frequency_settings_widget.get_save_path()
        full_path = os.path.join(save_path, filename)
        
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
            
# Target path logged in main window
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            QMessageBox.critical(self, "Recording Error", 
                               f"Failed to start recording:\n{str(e)}")

    def stop_recording(self):
        """Stop recording measurement data and trigger automatic saving."""
        logger.info("Recording stopped")
        
        try:
            # Emit signal to stop recording
            self.stop_recording_requested.emit()
            
            # Update internal state
            self.is_recording = False
            
            # Update button states
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.update_save_button_state()
            
            # Update status to stopped
            self.status_display.set_status("Stopped")
            
            # Start automatic saving after 1 second
            self.auto_save_timer.start(1000)  # 1 second delay
            
# Auto-saving will be logged when it happens
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            QMessageBox.critical(self, "Stop Recording Error", 
                               f"Failed to stop recording:\n{str(e)}")

    def save_recording(self):
        """Save the recorded measurement data."""
# Save action will be logged when completed
        
        if not self.current_recording_path:
            QMessageBox.warning(self, "No Recording", 
                              "No recording to save. Please record something first.")
            return
        
        if not self.frequency_settings_widget or not self.frequency_settings_widget.get_save_path():
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
# Success already logged
        # Status is already set in start_recording, but we could update here if needed

    def recording_stopped_successfully(self, saved_path=None):
        """Called when recording stops successfully in the camera."""
# Success already logged in main window
        if saved_path:
            self.current_recording_path = saved_path
            self.original_recording_path = saved_path  # Store for potential renaming

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

    def _auto_save_recording(self):
        """Automatically save recording and update status."""
# Final save result will be logged
        
        if not self.current_recording_path:
            logger.warning("No recording path available for auto-save")
            return
        
        try:
            # Check if the current file needs to be renamed to avoid conflicts
            final_save_path = self._get_unique_filename(self.current_recording_path)
            
            # If the unique filename is different, we need to rename the file
            if final_save_path != self.current_recording_path:
                pass  # File rename logged in main window
            
            # Emit signal to save recording
            self.save_recording_requested.emit(final_save_path)
            
            # Update status to saved
            self.status_display.set_status("Saved")
            
            # Clear status after 3 seconds
            self.status_clear_timer.start(3000)
            
            logger.info(f"Saved: {os.path.basename(final_save_path)}")
            
        except Exception as e:
            logger.error(f"Error in auto-save: {e}")
            self.status_display.set_status("Error")

    def _get_unique_filename(self, original_path):
        """Generate unique filename by adding _1, _2, etc. if file exists."""
        if not os.path.exists(original_path):
            return original_path
        
        # Split path into directory, name, and extension
        dir_path, filename = os.path.split(original_path)
        name, ext = os.path.splitext(filename)
        
        counter = 1
        while True:
            new_name = f"{name}_{counter}{ext}"
            new_path = os.path.join(dir_path, new_name)
            if not os.path.exists(new_path):
# File increment handled silently
                return new_path
            counter += 1

    def _clear_saved_status(self):
        """Clear the saved status display."""
        self.status_display.clear()
        
        # Reset for next recording
        self.current_recording_path = ""
        self.original_recording_path = ""

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