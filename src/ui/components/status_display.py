"""
Shared status components for consistent display across widgets.
"""

from PyQt5.QtWidgets import QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter


class StatusIndicator(QLabel):
    """Circle indicator that changes color based on status."""
    
    STATUS_COLORS = {
        'live': QColor(0, 255, 0),          # Green
        'connected': QColor(0, 255, 0),     # Green
        'recording': QColor(255, 0, 0),     # Red
        'initializing': QColor(255, 165, 0), # Orange
        'paused': QColor(255, 255, 0),      # Yellow
        'stopped': QColor(255, 165, 0),     # Orange
        'saved': QColor(0, 255, 0),         # Green
        'disconnected': QColor(255, 0, 0),   # Red
        'error': QColor(255, 0, 0),         # Red
        'ready': QColor(128, 128, 128),     # Gray
        'test_pattern': QColor(100, 149, 237) # Cornflower blue
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.status_color = self.STATUS_COLORS['ready']
        
    def set_status(self, status):
        """Set status by name or use default gray."""
        status_key = status.lower().replace(' ', '_').replace('-', '_')
        for key, color in self.STATUS_COLORS.items():
            if key in status_key:
                self.status_color = color
                break
        else:
            self.status_color = self.STATUS_COLORS['ready']
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.status_color)
        painter.setPen(Qt.black)
        painter.drawEllipse(2, 2, 12, 12)


class StatusDisplay(QWidget):
    """Standardized status display with text and indicator."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        self.text_label = QLabel("")
        self.indicator = StatusIndicator()
        
        layout.addWidget(self.text_label)
        layout.addWidget(self.indicator)
        layout.addStretch()
        
    def set_status(self, text):
        """Set both text and indicator color based on status."""
        self.text_label.setText(text)
        self.indicator.set_status(text)
        
    def clear(self):
        """Clear the status display."""
        self.text_label.setText("")
        self.indicator.set_status("ready")