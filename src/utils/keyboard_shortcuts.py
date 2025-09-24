from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from src.controllers.stage_manager import StageManager
from src.utils.logger import get_logger

logger = get_logger("keyboard_shortcuts")


class KeyboardShortcutManager:
    """Global shortcuts for XY stage movement (Ctrl + arrow keys)."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.stage_manager = None
        self._setup_stage_shortcuts()

    def _setup_stage_shortcuts(self):
        try:
            self.stage_manager = StageManager.get_instance()
            # Ctrl+Up/Down/Left/Right as application-wide shortcuts
            self.up_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Up), self.main_window)
            self.up_shortcut.setContext(Qt.ApplicationShortcut)
            self.up_shortcut.activated.connect(self.move_stage_up)

            self.down_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Down), self.main_window)
            self.down_shortcut.setContext(Qt.ApplicationShortcut)
            self.down_shortcut.activated.connect(self.move_stage_down)

            self.left_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Left), self.main_window)
            self.left_shortcut.setContext(Qt.ApplicationShortcut)
            self.left_shortcut.activated.connect(self.move_stage_left)

            self.right_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Right), self.main_window)
            self.right_shortcut.setContext(Qt.ApplicationShortcut)
            self.right_shortcut.activated.connect(self.move_stage_right)
        except Exception as e:
            logger.error(f"Shortcut setup failed: {e}")
            self.stage_manager = None

    def move_stage_up(self):
        if self.stage_manager:
            self.stage_manager.move_up()

    def move_stage_down(self):
        if self.stage_manager:
            self.stage_manager.move_down()

    def move_stage_left(self):
        if self.stage_manager:
            self.stage_manager.move_left()

    def move_stage_right(self):
        if self.stage_manager:
            self.stage_manager.move_right()