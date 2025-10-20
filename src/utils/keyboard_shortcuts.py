from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QObject, QEvent
from typing import Dict, Callable, Optional

from src.controllers.stage_manager import StageManager
from src.utils.logger import get_logger

logger = get_logger("keyboard_shortcuts")


class KeyboardShortcutManager(QObject):
    """Global shortcuts for XY stage movement (Ctrl + arrow keys)."""
    
    # Movement mapping for DRY principle
    MOVEMENT_MAPPINGS = {
        Qt.Key_Up: 'move_up',
        Qt.Key_Down: 'move_down', 
        Qt.Key_Left: 'move_left',
        Qt.Key_Right: 'move_right'
    }

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.stage_manager: Optional[StageManager] = None
        
        # Install event filter at application level for global capture
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)
        
        self._setup_stage_shortcuts()

    def _setup_stage_shortcuts(self) -> None:
        """Set up stage movement shortcuts using a mapping-based approach."""
        try:
            self.stage_manager = StageManager.get_instance()
            
            # Create shortcuts for each movement direction
            for key, method_name in self.MOVEMENT_MAPPINGS.items():
                shortcut = QShortcut(QKeySequence(Qt.CTRL + key), self.main_window)
                shortcut.setContext(Qt.WindowShortcut)
                shortcut.activated.connect(lambda method=method_name: self._execute_stage_movement(method))
            
            logger.info("Stage keyboard shortcuts initialized: Ctrl+Arrow keys with WindowShortcut context")
        except Exception as e:
            logger.error(f"Shortcut setup failed: {e}")
            self.stage_manager = None

    def _execute_stage_movement(self, method_name: str) -> None:
        """Execute a stage movement method with error handling."""
        try:
            if not self.stage_manager:
                self.stage_manager = StageManager.get_instance()
            
            if self.stage_manager:
                method = getattr(self.stage_manager, method_name)
                method()
        except Exception as e:
            logger.error(f"Error executing {method_name}: {e}")

    def eventFilter(self, obj, event) -> bool:
        """Event filter to intercept Ctrl+Arrow keys before they reach text controls."""
        if (event.type() == QEvent.KeyPress and 
            event.modifiers() == Qt.ControlModifier and
            event.key() in self.MOVEMENT_MAPPINGS):
            
            method_name = self.MOVEMENT_MAPPINGS[event.key()]
            self._execute_stage_movement(method_name)
            return True  # Event handled, don't pass to other widgets
        
        # Pass other events to the default handler
        return super().eventFilter(obj, event)