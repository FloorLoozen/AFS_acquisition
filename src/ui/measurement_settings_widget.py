"""
Measurement Settings Widget for AFS Acquisition.
Provides configuration options for measurement paths and settings.
"""

import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFileDialog, QFrame, QWidget
)

from src.utils.logger import get_logger

logger = get_logger("measurement_settings")


class MeasurementSettingsWidget(QGroupBox):
    """Widget for configuring measurement settings including save paths."""

    def __init__(self, parent=None):
        super().__init__("Measurement Settings", parent)
        
# Frequency settings initialized
        
        # Initialize paths - use specified tmp folder
        import os
        # Use the specified path: C:/Users/fAFS/Documents/Floor/tmp
        default_dir = "C:/Users/fAFS/Documents/Floor/tmp"
        self.save_path = default_dir
        # Use simple date format for filename: YYYYMMDD
        self.default_filename = datetime.now().strftime("%Y%m%d")
        
        # Create default directory if it doesn't exist
        self._create_default_directory()
        
        self._init_ui()

    def _create_default_directory(self):
        """Create the default save directory if it doesn't exist."""
        try:
            os.makedirs(self.save_path, exist_ok=True)
# Save directory ready
        except Exception as e:
            logger.warning(f"Could not create default directory {self.save_path}: {e}")

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
        layout.setSpacing(8)  # Consistent spacing between rows
        
        # Create all form rows
        self._add_path_row(layout, "Save Path:", "save", 
                          "Select folder to save files...", self._browse_save_path)
        
        self._add_filename_row(layout)
        
        layout.addSpacing(12)  # Separator
        
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
        
        # Set default value if it's the save path
        if attr_name == "save":
            line_edit.setText(self.save_path)
            # Connect signal to update path when user types manually
            line_edit.textChanged.connect(self._on_save_path_changed)
        
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

    def _browse_save_path(self):
        """Browse for save path."""
        path = self._browse_for_directory("Select Save Directory", 
                                         self.save_path_edit.text())
        if path:
            self.save_path_edit.setText(path)
            self.save_path = path
            logger.info(f"Save path set to: {path}")
    
    def _on_save_path_changed(self):
        """Handle manual changes to save path text field."""
        path = self.save_path_edit.text().strip()
        if path and os.path.isdir(path):
            self.save_path = path
            logger.debug(f"Save path updated to: {path}")
        elif path:
            # Path exists in text field but may not be valid directory yet
            self.save_path = path  # Store it anyway, user might be typing
    
    def _browse_for_directory(self, title, current_path):
        """Common directory browsing functionality."""
        start_path = current_path or os.path.expanduser("~")
        return QFileDialog.getExistingDirectory(self, title, start_path)

    # Getter methods
    def get_save_path(self):
        """Get the current save path."""
        # Always return the text field value if it exists, otherwise fallback to internal path
        if hasattr(self, 'save_path_edit'):
            text_path = self.save_path_edit.text().strip()
            if text_path:
                self.save_path = text_path  # Update internal path
                return text_path
        return self.save_path

    def get_measurements_path(self):
        """Get the current save path (compatibility method)."""
        return self.save_path

    def get_sample_information(self):
        """Get the sample information."""
        return self.sample_edit.text().strip()

    def get_notes(self):
        """Get the measurement notes."""
        return self.notes_edit.text().strip()

    def get_filename(self):
        """Get the current HDF5 filename with .hdf5 extension and automatic numbering."""
        import os
        base_filename = self.filename_edit.text().strip() or self.default_filename
        
        # Check if file already exists and add numbering
        if self.save_path:
            # Start with just the date (no _1)
            filename = f"{base_filename}.hdf5"
            full_path = os.path.join(self.save_path, filename)
            
            if not os.path.exists(full_path):
                return filename
            
            # If base filename exists, start numbering from 1
            counter = 1
            while True:
                filename = f"{base_filename}_{counter}.hdf5"
                full_path = os.path.join(self.save_path, filename)
                if not os.path.exists(full_path):
                    return filename
                counter += 1
        else:
            return f"{base_filename}.hdf5"

    def get_full_file_path(self):
        """Get the complete path for the HDF5 file."""
        if not self.save_path:
            return ""
        return os.path.join(self.save_path, self.get_filename())

    # Setter methods
    def set_save_path(self, path):
        """Set the save path."""
        if path and os.path.isdir(path):
            self.save_path = path
            self.save_path_edit.setText(path)
            logger.info(f"Save path set to: {path}")

    def set_measurements_path(self, path):
        """Set the save path (compatibility method)."""
        self.set_save_path(path)

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
        """Check if save path is configured."""
        return bool(self.save_path)


# Example usage if this file is run directly
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    widget = FrequencySettingsWidget()
    widget.show()
    sys.exit(app.exec_())