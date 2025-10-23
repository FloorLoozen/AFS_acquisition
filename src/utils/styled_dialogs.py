"""
Styled dialogs for AFS Acquisition.

Provides uniform styled dialogs for operations like saving, loading,
and processing with consistent visual appearance across the application.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class StyledProgressDialog(QDialog):
    """
    Uniform styled progress dialog for long-running operations.
    
    Features:
    - Clean, modern appearance
    - Consistent styling with the rest of the application
    - Non-closeable during operation
    - Optional progress bar (determinate or indeterminate)
    """
    
    def __init__(self, title: str, message: str, parent=None, show_progress: bool = False):
        """
        Initialize styled progress dialog.
        
        Args:
            title: Dialog window title
            message: Main message to display
            parent: Parent widget
            show_progress: Whether to show a progress bar
        """
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        
        # Set fixed size for consistency
        self.setFixedSize(400, 150)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title label
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Message label
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignCenter)
        message_font = QFont()
        message_font.setPointSize(10)
        self.message_label.setFont(message_font)
        layout.addWidget(self.message_label)
        
        # Optional progress bar
        if show_progress:
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 0)  # Indeterminate by default
            self.progress_bar.setTextVisible(False)
            layout.addWidget(self.progress_bar)
        else:
            self.progress_bar = None
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Apply stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
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
    
    def update_message(self, message: str):
        """Update the message text."""
        self.message_label.setText(message)
    
    def set_progress(self, value: int, maximum: int = 100):
        """
        Set progress bar value (makes it determinate).
        
        Args:
            value: Current progress value
            maximum: Maximum progress value
        """
        if self.progress_bar:
            self.progress_bar.setRange(0, maximum)
            self.progress_bar.setValue(value)
    
    def set_indeterminate(self):
        """Set progress bar to indeterminate mode."""
        if self.progress_bar:
            self.progress_bar.setRange(0, 0)
    
    def closeEvent(self, event):
        """Allow closing the dialog."""
        event.accept()  # Allow it to close


class StyledMessageDialog(QDialog):
    """
    Styled message dialog for information, warnings, or confirmations.
    Uses the same styling as StyledProgressDialog for consistency.
    """
    
    def __init__(self, title: str, message: str, parent=None):
        """
        Initialize styled message dialog.
        
        Args:
            title: Dialog window title
            message: Message to display
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setModal(True)
        
        # Set minimum size
        self.setMinimumSize(350, 120)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # Message label
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignCenter)
        message_font = QFont()
        message_font.setPointSize(10)
        message_label.setFont(message_font)
        layout.addWidget(message_label)
        
        self.setLayout(layout)
        
        # Apply stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
        """)
