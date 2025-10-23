"""
Centralized imports for common dependencies to reduce redundancy and improve load times.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, ClassVar, Protocol
from pathlib import Path
import os
import sys
from datetime import datetime
import time
import json
import numpy as np
import h5py

# PyQt5 common imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QGroupBox, QFrame, QScrollArea,
    QFileDialog, QMessageBox, QDialog, QApplication, QMainWindow,
    QSlider, QProgressBar, QSplitter, QTabWidget, QListWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, QObject, pyqtSignal, pyqtSlot,
    QSize, QRect, QPoint, QSettings, QStandardPaths
)
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPixmap, QIcon, QPalette,
    QBrush, QPen, QIntValidator, QDoubleValidator
)

# Project-specific imports - importing here can cause circular dependencies
# Import these directly in modules that need them instead:
# from src.utils.logger import get_logger
# from src.utils.exceptions import ...
# from src.utils.validation import ...

# Keep a lazy import function for logger to avoid circular imports
def get_logger_safe(name: str = None):
    """Safe logger import to avoid circular dependencies."""
    from src.utils.logger import get_logger as _get_logger
    return _get_logger(name)

# Common constants
DEFAULT_TIMEOUT = 5000  # ms
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 0.5  # seconds

# Common data structures
ConfigDict = Dict[str, Any]
SettingsDict = Dict[str, Any]
MetadataDict = Dict[str, Any]

__all__ = [
    # Types
    'Dict', 'List', 'Optional', 'Union', 'Any', 'Tuple', 'ClassVar', 'Protocol',
    'Path', 'ConfigDict', 'SettingsDict', 'MetadataDict',
    
    # Standard library
    'os', 'sys', 'datetime', 'time', 'json', 'np', 'h5py',
    
    # PyQt5 widgets
    'QWidget', 'QVBoxLayout', 'QHBoxLayout', 'QGridLayout',
    'QLabel', 'QPushButton', 'QLineEdit', 'QTextEdit', 'QSpinBox', 'QDoubleSpinBox',
    'QComboBox', 'QCheckBox', 'QGroupBox', 'QFrame', 'QScrollArea',
    'QFileDialog', 'QMessageBox', 'QDialog', 'QApplication', 'QMainWindow',
    'QSlider', 'QProgressBar', 'QSplitter', 'QTabWidget', 'QListWidget',
    'QTableWidget', 'QTableWidgetItem', 'QHeaderView', 'QSizePolicy',
    
    # PyQt5 core
    'Qt', 'QTimer', 'QThread', 'QObject', 'pyqtSignal', 'pyqtSlot',
    'QSize', 'QRect', 'QPoint', 'QSettings', 'QStandardPaths',
    
    # PyQt5 gui
    'QFont', 'QColor', 'QPainter', 'QPixmap', 'QIcon', 'QPalette',
    'QBrush', 'QPen', 'QIntValidator', 'QDoubleValidator',
    
    # Project utilities (safe import function)
    'get_logger_safe',
    
    # Constants
    'DEFAULT_TIMEOUT', 'DEFAULT_RETRY_ATTEMPTS', 'DEFAULT_RETRY_DELAY'
]