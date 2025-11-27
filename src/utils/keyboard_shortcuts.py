"""Global keyboard shortcuts for stage movement and recording control."""

from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from src.controllers.stage_manager import StageManager
from src.utils.logger import get_logger

logger = get_logger("shortcuts")


class KeyboardShortcutManager:
    """Manages global keyboard shortcuts."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.stage_manager = None
        
        try:
            self.stage_manager = StageManager.get_instance()
        except Exception as e:
            logger.error(f"Stage manager init failed: {e}")
        
        self._setup_shortcuts()
    
    def _setup_shortcuts(self):
        """Setup all keyboard shortcuts."""
        shortcuts = {
            # XY Stage
            Qt.CTRL + Qt.Key_Up: lambda: self._move('move_up'),
            Qt.CTRL + Qt.Key_Down: lambda: self._move('move_down'),
            Qt.CTRL + Qt.Key_Left: lambda: self._move('move_left'),
            Qt.CTRL + Qt.Key_Right: lambda: self._move('move_right'),
            # Z Stage
            Qt.CTRL + Qt.Key_U: lambda: self._move('move_z_up'),
            Qt.CTRL + Qt.Key_D: lambda: self._move('move_z_down'),
            # Recording
            Qt.CTRL + Qt.Key_Space: self._toggle_recording
        }
        
        for key, handler in shortcuts.items():
            try:
                shortcut = QShortcut(QKeySequence(key), self.main_window)
                shortcut.activated.connect(handler)
            except Exception as e:
                logger.error(f"Shortcut setup failed: {e}")
    
    def _move(self, method_name):
        """Execute stage movement."""
        try:
            if not self.stage_manager:
                self.stage_manager = StageManager.get_instance()
            if self.stage_manager:
                getattr(self.stage_manager, method_name)()
        except Exception as e:
            logger.error(f"Movement error: {e}")
    
    def _toggle_recording(self):
        """Toggle recording on/off."""
        try:
            widget = self.main_window.acquisition_controls_widget
            if widget.is_recording:
                widget.stop_recording()
            else:
                widget.start_recording()
        except Exception as e:
            logger.error(f"Recording toggle error: {e}")