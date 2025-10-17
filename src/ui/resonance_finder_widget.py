"""
Resonance Finder Widget for the AFS Tracking System.
This feature will be implemented later.
"""

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QMessageBox, QPushButton
from PyQt5.QtCore import Qt

class ResonanceFinderWidget(QWidget):
    """Placeholder widget for resonance finder functionality - to be implemented later."""
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """Initialize placeholder UI indicating future implementation."""
        layout = QVBoxLayout()
        
        label = QLabel("Resonance Finder Widget")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 20px;")
        
        desc_label = QLabel("Click below to access resonance finder functionality")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("color: gray; font-size: 12px; margin: 10px;")
        
        # Button that shows the same popup as lookup table
        open_button = QPushButton("Open Resonance Finder")
        open_button.clicked.connect(self._show_not_implemented)
        open_button.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        
        layout.addStretch()
        layout.addWidget(label)
        layout.addWidget(desc_label)
        layout.addWidget(open_button)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def _show_not_implemented(self):
        """Show not implemented message - same as lookup table."""
        QMessageBox.information(self, "Not Implemented", "This feature will be implemented later.")