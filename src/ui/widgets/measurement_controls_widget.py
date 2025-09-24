"""
Measurement Controls Widget for the AFS Tracking System.
Provides measurement-specific controls and settings.
"""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout
)
from PyQt5.QtCore import Qt

from src.logger import get_logger

logger = get_logger("measurement_controls")


class MeasurementControlsWidget(QGroupBox):
    """Widget for measurement-specific controls and parameters."""

    def __init__(self, parent=None):
        super().__init__("Measurement Controls", parent)
        
        logger.info("Initializing MeasurementControlsWidget")
        
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Keep it simple for now - will add controls later
        main_layout.addStretch(1)

    # Methods will be added later when controls are implemented
    pass


# Example usage if this file is run directly
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    widget = MeasurementControlsWidget()
    widget.show()
    sys.exit(app.exec_())