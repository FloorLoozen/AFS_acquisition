"""
Hardware Status Warning Dialog for AFS Acquisition.
Shows which hardware components are not connected at startup.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QWidget, QMessageBox
)
from PyQt5.QtCore import Qt

from src.utils.logger import get_logger

logger = get_logger("hardware_status")


class HardwareStatusDialog(QDialog):
    """Minimal dialog to display hardware connection warnings."""
    
    def __init__(self, hardware_status, parent=None):
        """
        Initialize the hardware status dialog.
        
        Args:
            hardware_status: Dictionary with hardware connection status
            parent: Parent widget
        """
        super().__init__(parent)
        self.hardware_status = hardware_status
        self.setWindowTitle("Hardware Warning")
        self.setModal(True)
        self.setFixedSize(350, 200)
        
        # Count disconnected hardware
        self.disconnected_count = sum(1 for status in hardware_status.values() 
                                    if not status.get('connected', False))
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)
        
        # Warning message
        self._add_warning_message(layout)
        
        # Hardware status list
        self._add_hardware_status_list(layout)
        
        # Buttons
        self._add_buttons(layout)
    
    def _add_warning_message(self, layout):
        """Add warning message."""
        if self.disconnected_count == 1:
            message = "Hardware component not connected:"
        else:
            message = f"{self.disconnected_count} hardware components not connected:"
        
        message_label = QLabel(message)
        message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(message_label)
    
    def _add_hardware_status_list(self, layout):
        """Add simple list of hardware status."""
        # Create content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(4)
        
        # Add hardware status items
        for hw_name, hw_status in self.hardware_status.items():
            self._add_hardware_item(content_layout, hw_name, hw_status)
        
        layout.addWidget(content_widget)
    
    def _add_hardware_item(self, layout, hw_name, hw_status):
        """Add a single hardware status item."""
        item_layout = QHBoxLayout()
        
        # Hardware name
        name_label = QLabel(f"• {hw_name}:")
        name_label.setMinimumWidth(100)
        
        # Status message (only show disconnected items)
        connected = hw_status.get('connected', False)
        if not connected:
            status_message = hw_status.get('message', 'Connection failed')
            status_label = QLabel(status_message)
            status_label.setStyleSheet("color: #d32f2f;")
            
            item_layout.addWidget(name_label)
            item_layout.addWidget(status_label, 1)
            
            layout.addLayout(item_layout)
    
    def _add_buttons(self, layout):
        """Add dialog buttons."""
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Retry button
        retry_btn = QPushButton("Retry")
        retry_btn.clicked.connect(self.retry_connection)
        
        # Continue button
        continue_btn = QPushButton("Continue")
        continue_btn.clicked.connect(self.accept)
        continue_btn.setDefault(True)
        
        button_layout.addWidget(retry_btn)
        button_layout.addWidget(continue_btn)
        
        layout.addLayout(button_layout)
    
    def retry_connection(self):
        """Handle retry connection button click."""
        logger.info("User requested hardware connection retry")
        self.done(2)  # Return code 2 for retry


def show_hardware_status_warning(hardware_status, parent=None):
    """
    Show hardware status warning dialog if there are connection issues.
    
    Args:
        hardware_status: Dictionary with hardware connection status
        parent: Parent widget
        
    Returns:
        0: Dialog was rejected
        1: User chose to continue anyway  
        2: User chose to retry connection
        None: No dialog shown (all hardware connected)
    """
    # Check if there are any disconnected components
    disconnected_components = [name for name, status in hardware_status.items() 
                             if not status.get('connected', False)]
    
    if not disconnected_components:
        return None
    
    logger.info(f"Hardware connection issues detected: {', '.join(disconnected_components)}")
    
    # Show warning dialog
    dialog = HardwareStatusDialog(hardware_status, parent)
    return dialog.exec_()


# Example usage for testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Test hardware status
    test_status = {
        "Camera": {"connected": True, "message": "IDS camera detected"},
        "XY Stage": {"connected": False, "message": "Device not found"},
        "Function Generator": {"connected": False, "message": "VISA resource not available"},
    }
    
    result = show_hardware_status_warning(test_status)
    logger.info(f"Hardware status dialog result: {result}")
    
    sys.exit(app.exec_())