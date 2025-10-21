"""Shared status components for consistent display across widgets."""

from PyQt5.QtWidgets import QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter
from typing import Dict, ClassVar


class StatusIndicator(QLabel):
    """Circle indicator that changes color based on status."""
    
    # Consolidated color scheme for consistency
    STATUS_COLORS: ClassVar[Dict[str, QColor]] = {
        # Success states (Green)
        'live': QColor(0, 255, 0),
        'connected': QColor(0, 255, 0),
        'saved': QColor(0, 255, 0),
        'on': QColor(0, 255, 0),
        
        # Active/Recording states (Red)
        'recording': QColor(255, 0, 0),
        'error': QColor(255, 0, 0),
        'disconnected': QColor(255, 0, 0),
        'connection_failed': QColor(255, 0, 0),
        'connection_error': QColor(255, 0, 0),
        'camera_error': QColor(255, 0, 0),
        
        # Warning/Transitional states (Orange)
        'initializing': QColor(255, 165, 0),
        'stopped': QColor(255, 165, 0),
        'connecting': QColor(255, 165, 0),
        'reconnecting': QColor(255, 165, 0),
        
        # Inactive/Off states (Gray)
        'off': QColor(128, 128, 128),
        
        # Special states
        'paused': QColor(255, 255, 0),       # Yellow
        'ready': QColor(128, 128, 128),      # Gray
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
        self.text_label.setText(text)
        self.indicator.set_status(text)
        
    def clear(self) -> None:
        """Clear the status display."""
        self.text_label.setText("")
        self.indicator.set_status("ready")