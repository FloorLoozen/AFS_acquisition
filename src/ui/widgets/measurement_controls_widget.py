"""
Measurement Controls Widget for the AFS Tracking System.
Placeholder widget - to be implemented later.
"""

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt


class MeasurementControlsWidget(QGroupBox):
    """Placeholder widget for measurement controls - to be implemented later."""

    def __init__(self, parent=None):
        super().__init__("Measurement Controls", parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 24, 8, 8)
        
        # Add stretch to center the content
        layout.addStretch(1)