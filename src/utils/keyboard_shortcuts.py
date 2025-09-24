from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QObject, QEvent

from src.controllers.stage_manager import StageManager
from src.utils.logger import get_logger

logger = get_logger("keyboard_shortcuts")


class KeyboardShortcutManager(QObject):
    """Global shortcuts for XY stage movement (Ctrl + arrow keys)."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.stage_manager = None
        
        # Install event filter at application level for global capture
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)
        
        self._setup_stage_shortcuts()

    def _setup_stage_shortcuts(self):
        try:
            self.stage_manager = StageManager.get_instance()
            # Stage movement shortcuts - using WindowShortcut context for higher priority
            self.up_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Up), self.main_window)
            self.up_shortcut.setContext(Qt.WindowShortcut)
            self.up_shortcut.activated.connect(self.move_stage_up)

            self.down_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Down), self.main_window)
            self.down_shortcut.setContext(Qt.WindowShortcut)
            self.down_shortcut.activated.connect(self.move_stage_down)

            self.left_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Left), self.main_window)
            self.left_shortcut.setContext(Qt.WindowShortcut)
            self.left_shortcut.activated.connect(self.move_stage_left)

            self.right_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Right), self.main_window)
            self.right_shortcut.setContext(Qt.WindowShortcut)
            self.right_shortcut.activated.connect(self.move_stage_right)
            
            logger.info("Stage keyboard shortcuts initialized: Ctrl+Arrow keys with WindowShortcut context")
        except Exception as e:
            logger.error(f"Shortcut setup failed: {e}")
            self.stage_manager = None

    def move_stage_up(self):
        try:
            if not self.stage_manager:
                self.stage_manager = StageManager.get_instance()
            if self.stage_manager:
                self.stage_manager.move_up()
        except Exception as e:
            logger.error(f"Error moving stage up: {e}")

    def move_stage_down(self):
        try:
            if not self.stage_manager:
                self.stage_manager = StageManager.get_instance()
            if self.stage_manager:
                self.stage_manager.move_down()
        except Exception as e:
            logger.error(f"Error moving stage down: {e}")

    def move_stage_left(self):
        try:
            if not self.stage_manager:
                self.stage_manager = StageManager.get_instance()
            if self.stage_manager:
                self.stage_manager.move_left()
        except Exception as e:
            logger.error(f"Error moving stage left: {e}")

    def move_stage_right(self):
        try:
            if not self.stage_manager:
                self.stage_manager = StageManager.get_instance()
            if self.stage_manager:
                self.stage_manager.move_right()
        except Exception as e:
            logger.error(f"Error moving stage right: {e}")

    def eventFilter(self, obj, event):
        """Event filter to intercept Ctrl+Arrow keys before they reach text controls."""
        if event.type() == QEvent.KeyPress:
            if event.modifiers() == Qt.ControlModifier:
                if event.key() == Qt.Key_Up:
                    self.move_stage_up()
                    return True  # Event handled, don't pass to other widgets
                elif event.key() == Qt.Key_Down:
                    self.move_stage_down()
                    return True
                elif event.key() == Qt.Key_Left:
                    self.move_stage_left()
                    return True
                elif event.key() == Qt.Key_Right:
                    self.move_stage_right()
                    return True
        
        # Pass other events to the default handler
        return super().eventFilter(obj, event)