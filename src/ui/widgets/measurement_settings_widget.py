"""
Measurement Settings Widget for the AFS Tracking System.
Provides configuration options for measurement paths and settings.
"""

import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFileDialog, QFormLayout, QFrame, QWidget
)
from PyQt5.QtCore import Qt

from src.logger import get_logger

logger = get_logger("measurement_settings")


class MeasurementSettingsWidget(QGroupBox):
    """Widget for configuring measurement settings including save paths."""

    def __init__(self, parent=None):
        super().__init__("Measurement Settings", parent)
        
        logger.info("Initializing MeasurementSettingsWidget")
        
        # Default paths
        self.measurements_save_path = ""
        self.lookup_table_save_path = ""
        
        # Default filename with today's date
        today = datetime.now().strftime("%Y%m%d")
        self.default_filename = today
        
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Create a frame for the settings
        settings_frame = QFrame()
        settings_frame.setFrameShape(QFrame.StyledPanel)
        settings_frame.setFrameShadow(QFrame.Raised)
        settings_frame.setLineWidth(1)
        settings_layout = QVBoxLayout(settings_frame)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        
        # Path settings section
        path_section = self.create_path_settings_section()
        settings_layout.addWidget(path_section)
        
        # Add some stretch to push content to top
        settings_layout.addStretch(1)
        
        # Add the settings frame to main layout
        main_layout.addWidget(settings_frame)

    def create_path_settings_section(self):
        """Create the path settings section."""
        # Create a simple widget instead of a group box
        section = QWidget()
        layout = QVBoxLayout(section)
        
        # Measurements save path
        measurements_layout = QHBoxLayout()
        measurements_label = QLabel("Measurements Save Path:")
        measurements_label.setMinimumWidth(150)
        self.measurements_path_edit = QLineEdit()
        self.measurements_path_edit.setPlaceholderText("Select folder to save measurements...")
        self.measurements_browse_btn = QPushButton("Browse...")
        self.measurements_browse_btn.setFixedWidth(60)  # Set fixed width to match .hdf5 label
        self.measurements_browse_btn.clicked.connect(self.browse_measurements_path)
        
        measurements_layout.addWidget(measurements_label)
        measurements_layout.addWidget(self.measurements_path_edit, 1)
        measurements_layout.addWidget(self.measurements_browse_btn)
        
        # Lookup table save path
        lookup_layout = QHBoxLayout()
        lookup_label = QLabel("Lookup Table Save Path:")
        lookup_label.setMinimumWidth(150)
        self.lookup_path_edit = QLineEdit()
        self.lookup_path_edit.setPlaceholderText("Select folder to save lookup tables...")
        self.lookup_browse_btn = QPushButton("Browse...")
        self.lookup_browse_btn.setFixedWidth(60)  # Set fixed width to match .hdf5 label
        self.lookup_browse_btn.clicked.connect(self.browse_lookup_path)
        
        lookup_layout.addWidget(lookup_label)
        lookup_layout.addWidget(self.lookup_path_edit, 1)
        lookup_layout.addWidget(self.lookup_browse_btn)
        
        # Filename
        filename_layout = QHBoxLayout()
        filename_label = QLabel("Filename:")
        filename_label.setMinimumWidth(150)
        self.filename_edit = QLineEdit()
        self.filename_edit.setText(self.default_filename)
        self.filename_edit.setPlaceholderText("measurement_file")
        hdf5_label = QLabel(".hdf5")
        hdf5_label.setStyleSheet("font-weight: bold; color: #666;")
        hdf5_label.setFixedWidth(60)  # Set fixed width to match browse buttons
        filename_layout.addWidget(filename_label)
        filename_layout.addWidget(self.filename_edit, 1)
        filename_layout.addWidget(hdf5_label)
        
        # Add layouts to section
        layout.addLayout(measurements_layout)
        layout.addLayout(lookup_layout)
        layout.addLayout(filename_layout)
        
        # Add spacing to separate file paths from metadata
        layout.addSpacing(20)
        
        # Sample information
        sample_layout = QHBoxLayout()
        sample_label = QLabel("Sample:")
        sample_label.setMinimumWidth(150)
        self.sample_edit = QLineEdit()
        # Add a dummy widget to match the width of browse buttons
        sample_spacer = QLabel("")
        sample_spacer.setFixedWidth(60)
        sample_layout.addWidget(sample_label)
        sample_layout.addWidget(self.sample_edit, 1)
        sample_layout.addWidget(sample_spacer)
        
        # Notes field
        notes_layout = QHBoxLayout()
        notes_label = QLabel("Notes:")
        notes_label.setMinimumWidth(150)
        self.notes_edit = QLineEdit()
        # Add a dummy widget to match the width of browse buttons
        notes_spacer = QLabel("")
        notes_spacer.setFixedWidth(60)
        notes_layout.addWidget(notes_label)
        notes_layout.addWidget(self.notes_edit, 1)
        notes_layout.addWidget(notes_spacer)
        
        layout.addLayout(sample_layout)
        layout.addLayout(notes_layout)
        
        return section

    def browse_measurements_path(self):
        """Browse for measurements save path."""
        current_path = self.measurements_path_edit.text() or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(
            self, 
            "Select Measurements Save Directory", 
            current_path
        )
        
        if path:
            self.measurements_path_edit.setText(path)
            self.measurements_save_path = path
            logger.info(f"Measurements save path set to: {path}")

    def browse_lookup_path(self):
        """Browse for lookup table save path."""
        current_path = self.lookup_path_edit.text() or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(
            self, 
            "Select Lookup Table Save Directory", 
            current_path
        )
        
        if path:
            self.lookup_path_edit.setText(path)
            self.lookup_table_save_path = path
            logger.info(f"Lookup table save path set to: {path}")

    def get_measurements_path(self):
        """Get the current measurements save path."""
        return self.measurements_save_path

    def get_lookup_table_path(self):
        """Get the current lookup table save path."""
        return self.lookup_table_save_path

    def get_sample_information(self):
        """Get the sample information."""
        return self.sample_edit.text().strip()

    def get_notes(self):
        """Get the measurement notes."""
        return self.notes_edit.text().strip()

    def get_filename(self):
        """Get the current HDF5 filename with .hdf5 extension."""
        filename = self.filename_edit.text().strip()
        if not filename:
            filename = self.default_filename
        return f"{filename}.hdf5"

    def get_full_file_path(self):
        """Get the complete path for the HDF5 file."""
        if not self.measurements_save_path:
            return ""
        return os.path.join(self.measurements_save_path, self.get_filename())

    def set_measurements_path(self, path):
        """Set the measurements save path."""
        if path and os.path.isdir(path):
            self.measurements_save_path = path
            self.measurements_path_edit.setText(path)
            logger.info(f"Measurements path set to: {path}")

    def set_lookup_table_path(self, path):
        """Set the lookup table save path."""
        if path and os.path.isdir(path):
            self.lookup_table_save_path = path
            self.lookup_path_edit.setText(path)
            logger.info(f"Lookup table path set to: {path}")

    def set_filename(self, filename):
        """Set the filename (without .hdf5 extension)."""
        if filename:
            # Remove .hdf5 extension if provided
            if filename.lower().endswith('.hdf5'):
                filename = filename[:-5]
            self.filename_edit.setText(filename)
            logger.info(f"Filename set to: {filename}")

    def set_sample_information(self, sample_info):
        """Set the sample information."""
        if sample_info:
            self.sample_edit.setText(sample_info)
            logger.info(f"Sample information set: {sample_info}")

    def set_notes(self, notes):
        """Set the measurement notes."""
        if notes:
            self.notes_edit.setText(notes)
            logger.info(f"Notes set: {notes}")

    def is_configured(self):
        """Check if all required paths are configured."""
        return bool(self.measurements_save_path and self.lookup_table_save_path)


# Example usage if this file is run directly
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    widget = MeasurementSettingsWidget()
    widget.show()
    sys.exit(app.exec_())