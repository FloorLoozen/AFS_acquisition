"""Shared status components for consistent display across widgets."""

from PyQt5.QtWidgets import QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter

STATUS_COLORS = {
    'live': QColor(0, 255, 0),
    'connected': QColor(0, 255, 0),
    'saved': QColor(0, 255, 0),
    'on': QColor(0, 255, 0),
    'completed': QColor(0, 255, 0),
    'executing': QColor(0, 120, 255),
    'starting': QColor(0, 120, 255),
    'running': QColor(0, 120, 255),
    'recording': QColor(0, 120, 255),
    'moving': QColor(0, 120, 255),
    'sweeping': QColor(0, 120, 255),
    'retrieving_data': QColor(0, 120, 255),
    'error': QColor(255, 0, 0),
    'disconnected': QColor(255, 0, 0),
    'partially_connected': QColor(255, 0, 0),
    'connection_failed': QColor(255, 0, 0),
    'connection_error': QColor(255, 0, 0),
    'camera_error': QColor(255, 0, 0),
    'failed': QColor(255, 0, 0),
    'out_of_range': QColor(255, 0, 0),
    'initializing': QColor(255, 165, 0),
    'reinitializing': QColor(255, 165, 0),
    'stopped': QColor(255, 165, 0),
    'connecting': QColor(255, 165, 0),
    'reconnecting': QColor(255, 165, 0),
    'off': QColor(128, 128, 128),
    'ready': QColor(0, 255, 0),
    'paused': QColor(255, 255, 0),
    'test_pattern': QColor(100, 149, 237),
}


class StatusIndicator(QLabel):
    """Circle indicator that changes color based on status."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.status_color = STATUS_COLORS['off']  # Gray by default
        
    def set_status(self, status):
        """Set status by name or use default gray."""
        key = status.lower().replace(' ', '_').replace('-', '_')
        
        if key in STATUS_COLORS:
            self.status_color = STATUS_COLORS[key]
        else:
            self.status_color = next(
                (color for k, color in STATUS_COLORS.items() if k in key),
                STATUS_COLORS['off']  # Gray as fallback
            )
        self.update()
        
    def paintEvent(self, event):
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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        self.indicator = StatusIndicator()
        self.text_label = QLabel("")
        
        layout.addWidget(self.indicator)
        layout.addWidget(self.text_label)
        layout.addStretch()
        
    def set_status(self, text):
        """Set both text and indicator color based on status."""
        if text.lower() in ['unknown', '']:
            self.text_label.setText("")
        else:
            self.text_label.setText(text)
        self.indicator.set_status(text)
        
    def clear(self):
        """Clear the status display."""
        self.text_label.setText("")
        self.indicator.set_status("off")