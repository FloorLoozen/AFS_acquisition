"""Styled dialogs for AFS Acquisition."""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class StyledProgressDialog(QDialog):
    """Uniform styled progress dialog for long-running operations."""
    
    def __init__(self, title, message, parent=None, show_progress=False):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setFixedSize(400, 150)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignCenter)
        message_font = QFont()
        message_font.setPointSize(10)
        self.message_label.setFont(message_font)
        layout.addWidget(self.message_label)
        
        if show_progress:
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setTextVisible(False)
            layout.addWidget(self.progress_bar)
        else:
            self.progress_bar = None
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.setStyleSheet("""
            QDialog { background-color: #f5f5f5; }
            QLabel { color: #333333; }
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #ffffff;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
    
    def update_message(self, message):
        self.message_label.setText(message)
    
    def set_progress(self, value, maximum=100):
        if self.progress_bar:
            self.progress_bar.setRange(0, maximum)
            self.progress_bar.setValue(value)
    
    def set_indeterminate(self):
        if self.progress_bar:
            self.progress_bar.setRange(0, 0)
    
    def closeEvent(self, event):
        event.accept()


class StyledMessageDialog(QDialog):
    """Styled message dialog for information, warnings, or confirmations."""
    
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(350, 120)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)
        
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignCenter)
        message_font = QFont()
        message_font.setPointSize(10)
        message_label.setFont(message_font)
        layout.addWidget(message_label)
        
        self.setLayout(layout)
        
        self.setStyleSheet("""
            QDialog { background-color: #f5f5f5; }
            QLabel { color: #333333; }
        """)
