from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QObject, QEvent
from typing import Dict, Callable, Optional

from src.controllers.stage_manager import StageManager
from src.utils.logger import get_logger

logger = get_logger("keyboard_shortcuts")


class KeyboardShortcutManager(QObject):
    """Global shortcuts for XY and Z stage movement and recording control."""
    
    # Movement mapping for DRY principle
    MOVEMENT_MAPPINGS = {
        Qt.Key_Up: 'move_up',
        Qt.Key_Down: 'move_down', 
        Qt.Key_Left: 'move_left',
        Qt.Key_Right: 'move_right'
    }
    
    # Z-stage movement mappings
    Z_MOVEMENT_MAPPINGS = {
        Qt.Key_U: 'move_z_up',
        Qt.Key_D: 'move_z_down'
    }

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.stage_manager: Optional[StageManager] = None
        
        # Install event filter at application level for global capture
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)
        
        self._setup_stage_shortcuts()
        self._setup_z_stage_shortcuts()
        self._setup_recording_shortcut()

    def _setup_stage_shortcuts(self) -> None:
        """Set up XY stage movement shortcuts using a mapping-based approach."""
        try:
            self.stage_manager = StageManager.get_instance()
            
            # Create shortcuts for each movement direction
            for key, method_name in self.MOVEMENT_MAPPINGS.items():
                shortcut = QShortcut(QKeySequence(Qt.CTRL + key), self.main_window)
                shortcut.setContext(Qt.WindowShortcut)
                shortcut.activated.connect(lambda method=method_name: self._execute_stage_movement(method))
            
        except Exception as e:
            logger.error(f"XY shortcut setup failed: {e}")
            self.stage_manager = None
    
    def _setup_z_stage_shortcuts(self) -> None:
        """Set up Z stage movement shortcuts (Ctrl+U for up, Ctrl+D for down)."""
        try:
            if self.stage_manager is None:
                self.stage_manager = StageManager.get_instance()
            
            # Create shortcuts for Z movement
            for key, method_name in self.Z_MOVEMENT_MAPPINGS.items():
                shortcut = QShortcut(QKeySequence(Qt.CTRL + key), self.main_window)
                shortcut.setContext(Qt.WindowShortcut)
                shortcut.activated.connect(lambda method=method_name: self._execute_stage_movement(method))
            
        except Exception as e:
            logger.error(f"Z shortcut setup failed: {e}")
    
    def _setup_recording_shortcut(self) -> None:
        """Set up Ctrl+Space shortcut for start/stop recording."""
        try:
            shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Space), self.main_window)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(self._toggle_recording)
        except Exception as e:
            logger.error(f"Recording shortcut setup failed: {e}")
    
    def _toggle_recording(self) -> None:
        """Toggle recording on/off with Ctrl+Space."""
        try:
            if hasattr(self.main_window, 'acquisition_controls_widget'):
                widget = self.main_window.acquisition_controls_widget
                if widget.is_recording:
                    # Stop recording
                    widget.stop_recording()
                else:
                    # Start recording
                    widget.start_recording()
        except Exception as e:
            logger.error(f"Error toggling recording: {e}")

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
        """Event filter to intercept Ctrl+Arrow keys, Ctrl+U/D, and Ctrl+Space before they reach text controls."""
        if event.type() == QEvent.KeyPress and event.modifiers() == Qt.ControlModifier:
            # Handle Ctrl+Arrow for XY stage movement
            if event.key() in self.MOVEMENT_MAPPINGS:
                method_name = self.MOVEMENT_MAPPINGS[event.key()]
                self._execute_stage_movement(method_name)
                return True  # Event handled, don't pass to other widgets
            
            # Handle Ctrl+U/D for Z stage movement
            elif event.key() in self.Z_MOVEMENT_MAPPINGS:
                method_name = self.Z_MOVEMENT_MAPPINGS[event.key()]
                self._execute_stage_movement(method_name)
                return True  # Event handled, don't pass to other widgets
            
            # Handle Ctrl+Space for recording toggle
            elif event.key() == Qt.Key_Space:
                self._toggle_recording()
                return True  # Event handled, don't pass to other widgets
        
        # Pass other events to the default handler
        return super().eventFilter(obj, event)