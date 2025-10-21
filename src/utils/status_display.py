"""
Status Display Widget for the AFS Tracking System.
Provides a standardized status indicator with colored circular status and text.
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QBrush, QColor

from src.utils.logger import get_logger

logger = get_logger("status_display")


class StatusDisplay(QWidget):
    """Widget that displays a colored status indicator with text."""
    
    # Status colors
    STATUS_COLORS = {
        'ready': '#00AA00',       # Green
        'connected': '#00AA00',   # Green  
        'on': '#00AA00',          # Green
        'completed': '#00AA00',   # Green
        'recording': '#FF6600',   # Orange
        'executing': '#FF6600',   # Orange
        'starting': '#FFAA00',    # Yellow
        'stopped': '#FFAA00',     # Yellow
        'disconnected': '#AA0000', # Red
        'error': '#AA0000',       # Red
        'offline': '#AA0000',     # Red
        'unknown': '#888888'      # Gray
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status_text = "Unknown"
        self.status_color = self.STATUS_COLORS['unknown']
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Status indicator (colored circle)
        self.indicator = StatusIndicator()
        self.indicator.setFixedSize(12, 12)
        layout.addWidget(self.indicator)
        
        # Status text
        self.label = QLabel(self.status_text)
        # No bold styling - just normal text
        layout.addWidget(self.label)
        
        # Add stretch to prevent expansion
        layout.addStretch(1)
    
    def set_status(self, status_text: str):
        """Set the status text and update the indicator color."""
        self.status_text = status_text
        self.label.setText(status_text)
        
        # Determine color based on status text
        status_lower = status_text.lower()
        
        # Check for specific patterns in status text
        if any(word in status_lower for word in ['ready', 'connected']):
            self.status_color = self.STATUS_COLORS['ready']
        elif any(word in status_lower for word in ['on @', 'on']):
            self.status_color = self.STATUS_COLORS['on']
        elif any(word in status_lower for word in ['completed', 'saved']):
            self.status_color = self.STATUS_COLORS['completed']
        elif any(word in status_lower for word in ['recording', 'executing']):
            self.status_color = self.STATUS_COLORS['recording']
        elif any(word in status_lower for word in ['starting', 'loading']):
            self.status_color = self.STATUS_COLORS['starting']
        elif any(word in status_lower for word in ['stopped', 'stopping']):
            self.status_color = self.STATUS_COLORS['stopped']
        elif any(word in status_lower for word in ['disconnected', 'offline']):
            self.status_color = self.STATUS_COLORS['disconnected']
        elif any(word in status_lower for word in ['error', 'failed', 'fault']):
            self.status_color = self.STATUS_COLORS['error']
        else:
            # For unknown status, just show gray circle
            self.status_color = self.STATUS_COLORS['unknown']
        
        # Update indicator color
        self.indicator.set_color(self.status_color)
    
    def get_status(self) -> str:
        """Get the current status text."""
        return self.status_text
    
    def clear(self):
        """Clear the status display."""
        self.set_status("Ready")


class StatusIndicator(QWidget):
    """A simple colored circle indicator."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.color = QColor('#888888')  # Default gray
    
    def set_color(self, color_str: str):
        """Set the indicator color."""
        self.color = QColor(color_str)
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the colored circle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw filled circle
        brush = QBrush(self.color)
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)
        
        # Draw circle to fit widget size
        size = min(self.width(), self.height())
        x = (self.width() - size) // 2
        y = (self.height() - size) // 2
        painter.drawEllipse(x, y, size, size)


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