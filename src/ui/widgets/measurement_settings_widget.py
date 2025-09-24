"""
Measurement Settings Widget for the AFS Tracking System.
Provides configuration options for measurement paths and settings.
"""

import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFileDialog, QFrame, QWidget
)

from src.logger import get_logger

logger = get_logger("measurement_settings")


class MeasurementSettingsWidget(QGroupBox):
    """Widget for configuring measurement settings including save paths."""

    def __init__(self, parent=None):
        super().__init__("Measurement Settings", parent)
        
        logger.info("Initializing MeasurementSettingsWidget")
        
        # Initialize paths
        self.measurements_save_path = ""
        self.lookup_table_save_path = ""
        self.default_filename = datetime.now().strftime("%Y%m%d")
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Create settings frame
        settings_frame = self._create_settings_frame()
        main_layout.addWidget(settings_frame)

    def _create_settings_frame(self):
        """Create the main settings frame."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Add path settings
        path_section = self._create_path_settings_section()
        layout.addWidget(path_section)
        layout.addStretch(1)
        
        return frame

    def _create_path_settings_section(self):
        """Create the path settings section."""
        section = QWidget()
        layout = QVBoxLayout(section)
        
        # Create all form rows
        self._add_path_row(layout, "Measurements Save Path:", "measurements", 
                          "Select folder to save measurements...", self._browse_measurements_path)
        
        self._add_path_row(layout, "Lookup Table Save Path:", "lookup", 
                          "Select folder to save lookup tables...", self._browse_lookup_path)
        
        self._add_filename_row(layout)
        
        layout.addSpacing(20)  # Separator
        
        self._add_text_row(layout, "Sample:", "sample", "")
        self._add_text_row(layout, "Notes:", "notes", "")
        
        return section
    
    def _add_path_row(self, layout, label_text, attr_name, placeholder, browse_callback):
        """Add a path selection row with label, line edit, and browse button."""
        row_layout = QHBoxLayout()
        
        label = QLabel(label_text)
        label.setMinimumWidth(150)
        
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        setattr(self, f"{attr_name}_path_edit", line_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(60)
        browse_btn.clicked.connect(browse_callback)
        
        row_layout.addWidget(label)
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse_btn)
        
        layout.addLayout(row_layout)
    
    def _add_filename_row(self, layout):
        """Add the filename row with .hdf5 extension label."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Filename:")
        label.setMinimumWidth(150)
        
        self.filename_edit = QLineEdit()
        self.filename_edit.setText(self.default_filename)
        self.filename_edit.setPlaceholderText("measurement_file")
        
        hdf5_label = QLabel(".hdf5")
        hdf5_label.setStyleSheet("font-weight: bold; color: #666;")
        hdf5_label.setFixedWidth(60)
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.filename_edit, 1)
        row_layout.addWidget(hdf5_label)
        
        layout.addLayout(row_layout)
    
    def _add_text_row(self, layout, label_text, attr_name, placeholder):
        """Add a text input row with label and spacer."""
        row_layout = QHBoxLayout()
        
        label = QLabel(label_text)
        label.setMinimumWidth(150)
        
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        setattr(self, f"{attr_name}_edit", line_edit)
        
        spacer = QLabel("")
        spacer.setFixedWidth(60)
        
        row_layout.addWidget(label)
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(spacer)
        
        layout.addLayout(row_layout)

    def _browse_measurements_path(self):
        """Browse for measurements save path."""
        path = self._browse_for_directory("Select Measurements Save Directory", 
                                         self.measurements_path_edit.text())
        if path:
            self.measurements_path_edit.setText(path)
            self.measurements_save_path = path
            logger.info(f"Measurements save path set to: {path}")

    def _browse_lookup_path(self):
        """Browse for lookup table save path."""
        path = self._browse_for_directory("Select Lookup Table Save Directory", 
                                         self.lookup_path_edit.text())
        if path:
            self.lookup_path_edit.setText(path)
            self.lookup_table_save_path = path
            logger.info(f"Lookup table save path set to: {path}")
    
    def _browse_for_directory(self, title, current_path):
        """Common directory browsing functionality."""
        start_path = current_path or os.path.expanduser("~")
        return QFileDialog.getExistingDirectory(self, title, start_path)

    # Getter methods
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
        filename = self.filename_edit.text().strip() or self.default_filename
        return f"{filename}.hdf5"

    def get_full_file_path(self):
        """Get the complete path for the HDF5 file."""
        if not self.measurements_save_path:
            return ""
        return os.path.join(self.measurements_save_path, self.get_filename())

    # Setter methods
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
            filename = filename[:-5] if filename.lower().endswith('.hdf5') else filename
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

    # Utility methods
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