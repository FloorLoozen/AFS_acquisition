from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSizePolicy,
    QDialogButtonBox, QFrame, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QCloseEvent, QCursor

from src.utils.logger import get_logger
from src.controllers.stage_manager import StageManager

logger = get_logger("stages_gui")


class StagesWidget(QDialog):
    """
    Stages control popup dialog:
    - Arrow buttons for intuitive relative movement
    - Step size spinbox (syncs with global shortcuts)
    - Absolute position entry (Enter triggers move)
    - Live absolute position labels on top
    - Connect/Disconnect controls (uses StageManager singleton)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.Tool | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.manager = StageManager.get_instance()
        self.is_connected = False

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._build_ui()

        # Timer to refresh positions
        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self.update_position_display)
        self.position_timer.start(300)
        
        # Set dialog properties - with title
        self.setWindowTitle("XY Stage Control")  # Restore title
        self.setModal(False)  # Non-modal dialog so user can interact with main window
        self.resize(230, 320)  # More compact size for aligned elements
        
        # Auto-connect if possible
        self.connect_stage()

    # --- UI ---
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setSpacing(8)  # Add more spacing between elements
        
        # Section 1: Current Position
        # Add a section title
        current_pos_label = QLabel("<b>Current Position</b>")
        current_pos_label.setAlignment(Qt.AlignLeft)
        main.addWidget(current_pos_label)
        
        # Position display - Grid layout to align X and Y labels
        pos_grid = QGridLayout()
        pos_grid.setColumnMinimumWidth(0, 30)  # Fixed width for label column
        pos_grid.addWidget(QLabel("X:"), 0, 0, Qt.AlignLeft)
        self.x_pos_label = QLabel("0.000 mm")
        pos_grid.addWidget(self.x_pos_label, 0, 1, Qt.AlignLeft)
        
        pos_grid.addWidget(QLabel("Y:"), 1, 0, Qt.AlignLeft)
        self.y_pos_label = QLabel("0.000 mm")
        pos_grid.addWidget(self.y_pos_label, 1, 1, Qt.AlignLeft)
        
        main.addLayout(pos_grid)

        # Create connection buttons but don't add them to the layout or display
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_stage)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_stage)
        self.status_label = QLabel("")  # Empty status label, not displayed
        
        # Add first separator line
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        main.addWidget(line1)

        # Section 2: Relative Movement
        # Add a section title
        relative_move_label = QLabel("<b>Relative Movement</b>")
        relative_move_label.setAlignment(Qt.AlignLeft)
        main.addWidget(relative_move_label)
        
        # Movement arrows with intuitive layout
        grid = QGridLayout()
        self.up_btn = QPushButton("▲")
        self.up_btn.clicked.connect(self.move_up)
        self.up_btn.setMinimumSize(50, 50)
        
        self.left_btn = QPushButton("◄")
        self.left_btn.clicked.connect(self.move_left)
        self.left_btn.setMinimumSize(50, 50)
        
        self.right_btn = QPushButton("►")
        self.right_btn.clicked.connect(self.move_right)
        self.right_btn.setMinimumSize(50, 50)
        
        self.down_btn = QPushButton("▼")
        self.down_btn.clicked.connect(self.move_down)
        self.down_btn.setMinimumSize(50, 50)
        
        # Tooltips to hint shortcuts
        self.up_btn.setToolTip("Ctrl+Up (Move Up / +Y)")
        self.down_btn.setToolTip("Ctrl+Down (Move Down / -Y)")
        self.left_btn.setToolTip("Ctrl+Left (Move Left / -X)")
        self.right_btn.setToolTip("Ctrl+Right (Move Right / +X)")
        
        grid.addWidget(self.up_btn, 0, 1)
        grid.addWidget(self.left_btn, 1, 0)
        grid.addWidget(self.right_btn, 1, 2)
        grid.addWidget(self.down_btn, 2, 1)
        
        # Add a center point for reference
        center_label = QLabel("●")
        center_label.setAlignment(Qt.AlignCenter)
        grid.addWidget(center_label, 1, 1)
        
        main.addLayout(grid)

        # Step size with grid layout to match position controls
        step_grid = QGridLayout()
        step_grid.setColumnMinimumWidth(0, 30)  # Fixed width for label column
        step_grid.addWidget(QLabel("Step:"), 0, 0, Qt.AlignLeft)
        
        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(0.001, 10.0)
        self.step_size.setDecimals(3)
        self.step_size.setSingleStep(0.001)
        self.step_size.setSuffix(" mm")
        self.step_size.setFixedWidth(80)  # Make spinbox even narrower
        self.step_size.setValue(self.manager.default_step_size)
        self.step_size.valueChanged.connect(self._on_step_changed)
        step_grid.addWidget(self.step_size, 0, 1, Qt.AlignLeft)
        
        main.addLayout(step_grid)
        
        # Add second separator line
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        main.addWidget(line2)
        
        main.addSpacing(10)

        # Section 3: Absolute Position
        # Add a section title
        absolute_pos_label = QLabel("<b>Absolute Position</b>")
        absolute_pos_label.setAlignment(Qt.AlignLeft)
        main.addWidget(absolute_pos_label)
        
        # Absolute position + Go with aligned layout
        abs_group = QGridLayout()
        abs_group.setColumnMinimumWidth(0, 30)  # Fixed width for label column
        
        # X position
        abs_group.addWidget(QLabel("X:"), 0, 0, Qt.AlignLeft)
        self.x_abs = QDoubleSpinBox()
        self.x_abs.setRange(-100.0, 100.0)
        self.x_abs.setDecimals(3)
        self.x_abs.setSuffix(" mm")
        self.x_abs.setFixedWidth(80)  # Make spinbox narrower (same as step_size)
        abs_group.addWidget(self.x_abs, 0, 1, Qt.AlignLeft)
        
        # Y position
        abs_group.addWidget(QLabel("Y:"), 1, 0, Qt.AlignLeft)
        self.y_abs = QDoubleSpinBox()
        self.y_abs.setRange(-100.0, 100.0) 
        self.y_abs.setDecimals(3)
        self.y_abs.setSuffix(" mm")
        self.y_abs.setFixedWidth(80)  # Make spinbox narrower (same as step_size)
        abs_group.addWidget(self.y_abs, 1, 1, Qt.AlignLeft)
        
        # Go button (simplified text)
        self.go_btn = QPushButton("Go")
        self.go_btn.clicked.connect(self.go_to_position)
        abs_group.addWidget(self.go_btn, 2, 0, 1, 2, Qt.AlignCenter)
        
        main.addLayout(abs_group)
        
        # Enter-to-move behavior
        self.x_abs.editingFinished.connect(self.go_to_position)
        self.y_abs.editingFinished.connect(self.go_to_position)
        
        # No close button as requested
        
        self._set_controls_enabled(False)

    # --- Connection handling ---
    def connect_stage(self):
        """Connect to the XY stage hardware."""
        if self.is_connected:
            return
        self._update_status("Connecting...")
        try:
            # Ensure normal cursor during connection
            QApplication.restoreOverrideCursor()
            
            if self.manager.connect():
                self.is_connected = True
                self._set_controls_enabled(True)
                self._update_status("Connected")
                self.update_position_display()
            else:
                self._update_status("Connection failed")
        except Exception as e:
            logger.error(f"Error connecting: {e}")
            self._update_status("Connection error")

    def disconnect_stage(self):
        """Disconnect from the XY stage hardware."""
        if not self.is_connected:
            return
        try:
            self.manager.disconnect()
            self.is_connected = False
            self._set_controls_enabled(False)
            self._update_status("Disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
            
    def closeEvent(self, event):
        """Properly handle dialog close event"""
        # Stop the position update timer
        if self.position_timer.isActive():
            self.position_timer.stop()
        
        # No need to disconnect - we'll leave connection management to the StageManager
        # This allows other parts of the application to continue using the stage
        
        # Accept the close event
        event.accept()

    def _set_controls_enabled(self, enabled: bool):
        for w in [
            self.up_btn, self.down_btn, self.left_btn, self.right_btn,
            self.step_size, self.x_abs, self.y_abs, self.go_btn
        ]:
            w.setEnabled(enabled)
        
        # Connect/disconnect buttons are hidden, no need to update them

    def _update_status(self, text: str):
        # Status display is disabled, but keep method for compatibility
        pass

    # --- Position and movement ---
    def update_position_display(self):
        """Update the position display with current stage coordinates."""
        if not self.is_connected:
            return
        x, y = self.manager.get_position()
        if x is None or y is None:
            return
        self.x_pos_label.setText(f"{x:.3f} mm")
        self.y_pos_label.setText(f"{y:.3f} mm")
        # Only update spinboxes if the user isn't editing
        if not self.x_abs.hasFocus():
            self.x_abs.setValue(x)
        if not self.y_abs.hasFocus():
            self.y_abs.setValue(y)

    def _move_relative(self, dx: float = 0.0, dy: float = 0.0):
        if not self.is_connected:
            return
            
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        # Log the movement for debugging
        move_info = []
        if dx:
            direction = "RIGHT" if dx > 0 else "LEFT"
            move_info.append(f"X{'+' if dx > 0 else '-'}: {abs(dx):.3f} mm")
        if dy:
            direction = "UP" if dy > 0 else "DOWN"
            move_info.append(f"Y{'+' if dy > 0 else '-'}: {abs(dy):.3f} mm")
            
        if move_info:
            logger.debug(f"Moving: {', '.join(move_info)}")
            
        # Execute the move
        self.manager.move_relative(dx=dx if dx else None, dy=dy if dy else None)
        self.update_position_display()

    # These UI methods map button clicks to hardware directions
    # Since the camera view is rotated, we need to map the buttons
    # differently from the actual hardware directions
    
    def move_up(self):
        # UP arrow button pressed - need to move LEFT relative to camera view
        logger.debug("UP button pressed -> Moving LEFT relative to camera view")
        self._move_relative(dx=-self.step_size.value())  # Move LEFT (negative X)

    def move_down(self):
        # DOWN arrow button pressed - need to move RIGHT relative to camera view
        logger.debug("DOWN button pressed -> Moving RIGHT relative to camera view")
        self._move_relative(dx=self.step_size.value())  # Move RIGHT (positive X)

    def move_left(self):
        # LEFT arrow button pressed - need to move UP relative to camera view
        logger.debug("LEFT button pressed -> Moving UP relative to camera view")
        self._move_relative(dy=self.step_size.value())  # Move UP (positive Y)

    def move_right(self):
        # RIGHT arrow button pressed - need to move DOWN relative to camera view
        logger.debug("RIGHT button pressed -> Moving DOWN relative to camera view")
        self._move_relative(dy=-self.step_size.value())  # Move DOWN (negative Y)

    def go_to_position(self):
        if not self.is_connected:
            return
            
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        target_x = self.x_abs.value()
        target_y = self.y_abs.value()
        self.manager.move_to(x=target_x, y=target_y)
        self.update_position_display()

    def _on_step_changed(self, value: float):
        # Keep global shortcuts (Ctrl+Arrows) in sync with widget step
        self.manager.default_step_size = value
        
    def showEvent(self, event):
        """Override showEvent to ensure proper display and auto-connect when shown"""
        super().showEvent(event)
        
        # Always restore normal cursor when the dialog is shown
        QApplication.restoreOverrideCursor()
        
        # Ensure the dialog is properly sized and visible
        self.adjustSize()
        
        # If not connected, try to connect
        if not self.is_connected:
            QTimer.singleShot(100, self.connect_stage)