"""Shared status components for consistent display across widgets."""

from PyQt5.QtWidgets import QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter
from typing import Dict, ClassVar

from src.utils.logger import get_logger

logger = get_logger("status_display")


class StatusIndicator(QLabel):
    """Circle indicator that changes color based on status."""
    
    # Consolidated color scheme for consistency
    STATUS_COLORS: ClassVar[Dict[str, QColor]] = {
        # Success states (Green)
        'live': QColor(0, 255, 0),
        'connected': QColor(0, 255, 0),
        'saved': QColor(0, 255, 0),
        'on': QColor(0, 255, 0),
        'completed': QColor(0, 255, 0),
        
        # Active/Executing states (Blue) - for running processes
        'executing': QColor(0, 120, 255),     # Bright blue for active execution
        'starting': QColor(0, 120, 255),     # Blue for starting execution
        'running': QColor(0, 120, 255),      # Blue for active processes
        'recording': QColor(0, 120, 255),    # Blue for recording (active process)
        
        # Error/Critical states (Red)
        'error': QColor(255, 0, 0),
        'disconnected': QColor(255, 0, 0),
        'connection_failed': QColor(255, 0, 0),
        'connection_error': QColor(255, 0, 0),
        'camera_error': QColor(255, 0, 0),
        'failed': QColor(255, 0, 0),
        
        # Warning/Transitional states (Orange)
        'initializing': QColor(255, 165, 0),
        'reinitializing': QColor(255, 165, 0),
        'stopped': QColor(255, 165, 0),       # Orange for user-stopped actions
        'connecting': QColor(255, 165, 0),
        'reconnecting': QColor(255, 165, 0),
        
        # Inactive/Off states (Gray)
        'off': QColor(128, 128, 128),
        'ready': QColor(128, 128, 128),      # Gray for ready/idle state
        
        # Special states
        'paused': QColor(255, 255, 0),       # Yellow
        'test_pattern': QColor(100, 149, 237), # Cornflower blue
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.status_color = self.STATUS_COLORS['ready']
        
    def set_status(self, status: str) -> None:
        """Set status by name or use default gray."""
        # Normalize status string for consistent matching
        status_key = status.lower().replace(' ', '_').replace('-', '_')
        
        # Use direct lookup for exact matches, fallback to substring matching
        if status_key in self.STATUS_COLORS:
            self.status_color = self.STATUS_COLORS[status_key]
        else:
            # Fallback to substring matching for backward compatibility
            self.status_color = next(
                (color for key, color in self.STATUS_COLORS.items() if key in status_key),
                self.STATUS_COLORS['ready']
            )
        self.update()
        
    def paintEvent(self, event) -> None:
        """Draw the status indicator circle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.status_color)
        painter.setPen(Qt.black)
        painter.drawEllipse(2, 2, 12, 12)


class StatusDisplay(QWidget):
    """Standardized status display with text and indicator."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self) -> None:
        """Initialize the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        self.indicator = StatusIndicator()
        self.text_label = QLabel("")
        
        layout.addWidget(self.indicator)
        layout.addWidget(self.text_label)
        layout.addStretch()
        
    def set_status(self, text: str) -> None:
        """Set both text and indicator color based on status."""
        # If status is unknown, show nothing
        if text.lower() in ['unknown', '']:
            self.text_label.setText("")
        else:
            self.text_label.setText(text)
        self.indicator.set_status(text)
        
    def clear(self) -> None:
        """Clear the status display."""
        self.text_label.setText("")
        self.indicator.set_status("ready")


# Example usage if this file is run directly
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QPushButton
    
    app = QApplication(sys.argv)
    
    # Test window
    window = QWidget()
    layout = QVBoxLayout(window)
    
    # Create status display
    status = StatusDisplay()
    layout.addWidget(status)
    
    # Test buttons
    def test_ready():
        status.set_status("Ready")
    
    def test_recording():
        status.set_status("Recording")
    
    def test_error():
        status.set_status("Error")
    
    def test_on():
        status.set_status("ON @ 14.0 MHz")
    
    ready_btn = QPushButton("Test Ready")
    ready_btn.clicked.connect(test_ready)
    layout.addWidget(ready_btn)
    
    recording_btn = QPushButton("Test Recording")
    recording_btn.clicked.connect(test_recording)
    layout.addWidget(recording_btn)
    
    error_btn = QPushButton("Test Error")
    error_btn.clicked.connect(test_error)
    layout.addWidget(error_btn)
    
    on_btn = QPushButton("Test ON")
    on_btn.clicked.connect(test_on)
    layout.addWidget(on_btn)
    
    window.show()
    sys.exit(app.exec_())