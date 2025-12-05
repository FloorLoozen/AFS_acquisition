from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSizePolicy,
    QDialogButtonBox, QFrame, QApplication, QLineEdit, QWidget, QGroupBox, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QCloseEvent, QCursor

from src.utils.logger import get_logger
from src.controllers.stage_manager import StageManager
from src.utils.status_display import StatusDisplay

logger = get_logger("stages_gui")


class StagesWidget(QDialog):
    """
    Stages control popup dialog:
    - XY stage controls with arrow buttons for intuitive relative movement
    - Z stage controls with up/down buttons for focus adjustment
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
        self.z_is_connected = False
        self._is_moving = False  # Track movement status for status indicator
        self._last_error_time = 0  # Track when last error occurred
        
        # Timers for status transitions
        self._moving_timer = QTimer(self)
        self._moving_timer.setSingleShot(True)
        self._moving_timer.timeout.connect(self._on_moving_complete)
        
        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(self._on_error_timeout)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._build_ui()

        # Timer to refresh positions
        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self.update_position_display)
        self.position_timer.start(300)
        
        # Set dialog properties - with title
        self.setWindowTitle("Stage Control")
        self.setModal(False)  # Non-modal dialog so user can interact with main window
        self.setMinimumWidth(450)  # Compact clean design
        
        # Auto-connect if possible
        self.connect_stage()
        self.connect_z_stage()

    # --- UI ---
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setSpacing(5)
        main.setContentsMargins(10, 10, 10, 10)
        
        # Status display at the top
        self.status_display = StatusDisplay()
        main.addWidget(self.status_display)
        
        # ===== XY AND Z STAGES IN SEPARATE GROUP BOXES =====
        # Using horizontal layout for side-by-side group boxes
        stages_layout = QHBoxLayout()
        stages_layout.setSpacing(10)
        
        spinbox_width = 85
        
        # XY Stage Group
        xy_group = QGroupBox("XY Stage")
        xy_grid = QGridLayout()
        xy_grid.setSpacing(8)
        xy_grid.setContentsMargins(15, 15, 15, 15)
        
        # Row 0: X position
        self.x_pos_label = QLabel("X: 0.000 mm")
        self.x_pos_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        xy_grid.addWidget(self.x_pos_label, 0, 0, 1, 3)
        
        # Row 1: Y position
        self.y_pos_label = QLabel("Y: 0.000 mm")
        self.y_pos_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        xy_grid.addWidget(self.y_pos_label, 1, 0, 1, 3)
        
        # Row 2: Up button only
        self.up_btn = QPushButton("↑")
        self.up_btn.clicked.connect(self.move_up)
        self.up_btn.setFixedSize(40, 40)
        self.up_btn.setToolTip("Ctrl+Up")
        xy_grid.addWidget(self.up_btn, 2, 1, Qt.AlignCenter)
        
        # Row 3: Left and Right buttons
        self.left_btn = QPushButton("←")
        self.left_btn.clicked.connect(self.move_left)
        self.left_btn.setFixedSize(40, 40)
        self.left_btn.setToolTip("Ctrl+Left")
        xy_grid.addWidget(self.left_btn, 3, 0, Qt.AlignRight)
        
        self.right_btn = QPushButton("→")
        self.right_btn.clicked.connect(self.move_right)
        self.right_btn.setFixedSize(40, 40)
        self.right_btn.setToolTip("Ctrl+Right")
        xy_grid.addWidget(self.right_btn, 3, 2, Qt.AlignLeft)
        
        # Row 4: Down button only
        self.down_btn = QPushButton("↓")
        self.down_btn.clicked.connect(self.move_down)
        self.down_btn.setFixedSize(40, 40)
        self.down_btn.setToolTip("Ctrl+Down")
        xy_grid.addWidget(self.down_btn, 4, 1, Qt.AlignCenter)
        
        # Row 5: Step
        step_label = QLabel("Step:")
        step_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        xy_grid.addWidget(step_label, 5, 0)
        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(0.001, 10.0)
        self.step_size.setValue(self.manager.default_step_size)
        self.step_size.setSuffix(" mm")
        self.step_size.setDecimals(3)
        self.step_size.setFixedWidth(spinbox_width)
        self.step_size.setAlignment(Qt.AlignRight)
        self.step_size.valueChanged.connect(self._on_step_changed_spinbox)
        xy_grid.addWidget(self.step_size, 5, 1, 1, 2, Qt.AlignRight)
        
        # Row 6: Empty spacer
        xy_grid.setRowMinimumHeight(6, 10)
        
        # Row 7: Go to X
        x_abs_label = QLabel("Go to X:")
        x_abs_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        xy_grid.addWidget(x_abs_label, 7, 0)
        self.x_abs = QDoubleSpinBox()
        self.x_abs.setRange(-100.0, 100.0)
        self.x_abs.setValue(0.0)
        self.x_abs.setSuffix(" mm")
        self.x_abs.setDecimals(3)
        self.x_abs.setFixedWidth(spinbox_width)
        self.x_abs.setAlignment(Qt.AlignRight)
        xy_grid.addWidget(self.x_abs, 7, 1, 1, 2, Qt.AlignRight)
        
        # Row 8: Go to Y
        y_abs_label = QLabel("Go to Y:")
        y_abs_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        xy_grid.addWidget(y_abs_label, 8, 0)
        self.y_abs = QDoubleSpinBox()
        self.y_abs.setRange(-100.0, 100.0)
        self.y_abs.setValue(0.0)
        self.y_abs.setSuffix(" mm")
        self.y_abs.setDecimals(3)
        self.y_abs.setFixedWidth(spinbox_width)
        self.y_abs.setAlignment(Qt.AlignRight)
        xy_grid.addWidget(self.y_abs, 8, 1, 1, 2, Qt.AlignRight)
        
        # Row 9: Go button
        self.go_btn = QPushButton("Go")
        self.go_btn.clicked.connect(self.go_to_position)
        xy_grid.addWidget(self.go_btn, 9, 0, 1, 3)
        
        xy_group.setLayout(xy_grid)
        
        # === Z Stage Group ===
        z_group = QGroupBox("Z Stage")
        z_grid = QGridLayout()
        z_grid.setSpacing(8)
        z_grid.setContentsMargins(15, 15, 15, 15)
        
        # Row 0: Z position (aligned with X position)
        self.z_pos_label = QLabel("Z: 0.000 µm")
        self.z_pos_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        z_grid.addWidget(self.z_pos_label, 0, 0, 1, 2)
        
        # Row 1: Empty - skip (no Y equivalent in Z)
        
        # Row 2: Z Up button (aligned with XY Up button)
        self.z_up_btn = QPushButton("↑")
        self.z_up_btn.clicked.connect(self.move_z_up)
        self.z_up_btn.setFixedSize(40, 40)
        self.z_up_btn.setToolTip("Ctrl+U")
        z_grid.addWidget(self.z_up_btn, 2, 0, 1, 2, Qt.AlignCenter)
        
        # Row 3: Empty (aligned with Left/Right buttons row height)
        empty_row3 = QLabel("")
        empty_row3.setFixedHeight(40)  # Match button height
        z_grid.addWidget(empty_row3, 3, 0, 1, 2)
        
        # Row 4: Z Down button (aligned with XY Down button)
        self.z_down_btn = QPushButton("↓")
        self.z_down_btn.clicked.connect(self.move_z_down)
        self.z_down_btn.setFixedSize(40, 40)
        self.z_down_btn.setToolTip("Ctrl+D")
        z_grid.addWidget(self.z_down_btn, 4, 0, 1, 2, Qt.AlignCenter)
        
        # Row 5: Step (aligned with XY Step)
        z_step_label = QLabel("Step:")
        z_step_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        z_grid.addWidget(z_step_label, 5, 0)
        self.z_step_size = QDoubleSpinBox()
        self.z_step_size.setRange(0.1, 100.0)
        self.z_step_size.setValue(self.manager.default_z_step_size)
        self.z_step_size.setSuffix(" µm")
        self.z_step_size.setDecimals(1)
        self.z_step_size.setFixedWidth(spinbox_width)
        self.z_step_size.setAlignment(Qt.AlignRight)
        self.z_step_size.valueChanged.connect(self._on_z_step_changed_spinbox)
        z_grid.addWidget(self.z_step_size, 5, 1, Qt.AlignRight)
        
        # Row 6: Empty spacer
        z_grid.setRowMinimumHeight(6, 10)
        
        # Row 7: Go to Z (aligned with Go to X)
        z_abs_label = QLabel("Go to Z:")
        z_abs_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        z_grid.addWidget(z_abs_label, 7, 0)
        self.z_abs = QDoubleSpinBox()
        self.z_abs.setRange(0.0, 100.0)
        self.z_abs.setValue(0.0)
        self.z_abs.setSuffix(" µm")
        self.z_abs.setDecimals(3)
        self.z_abs.setFixedWidth(spinbox_width)
        self.z_abs.setAlignment(Qt.AlignRight)
        z_grid.addWidget(self.z_abs, 7, 1, Qt.AlignRight)
        
        # Row 8: Empty (aligned with Go to Y)
        empty_row8 = QLabel("")
        z_grid.addWidget(empty_row8, 8, 0, 1, 2)
        
        # Row 9: Go button (aligned with XY Go button)
        self.z_go_btn = QPushButton("Go")
        self.z_go_btn.clicked.connect(self.go_to_z_position)
        z_grid.addWidget(self.z_go_btn, 9, 0, 1, 2)
        
        z_group.setLayout(z_grid)
        
        # Make both group boxes the same size
        group_width = 200
        xy_group.setMinimumWidth(group_width)
        z_group.setMinimumWidth(group_width)
        
        # Add both groups to stages layout
        stages_layout.addWidget(xy_group)
        stages_layout.addWidget(z_group)
        
        # Add stages layout to main
        main.addLayout(stages_layout)
        
        # Enter-to-move behavior
        self.x_abs.editingFinished.connect(self.go_to_position)
        self.y_abs.editingFinished.connect(self.go_to_position)
        
        # Enter-to-move behavior for Z
        self.z_abs.editingFinished.connect(self.go_to_z_position)
        
        self._set_controls_enabled(False)
        
        # Set initial status
        self._update_status("Initializing...")

    # --- Connection handling ---
    def connect_stage(self):
        """Connect to the XY stage hardware."""
        if self.is_connected:
            return
        self._update_status("Connecting XY stage...")
        try:
            # Ensure normal cursor during connection
            QApplication.restoreOverrideCursor()
            
            if self.manager.connect():
                self.is_connected = True
                self._set_controls_enabled(True)
                self._update_connection_status()
                self.update_position_display()
            else:
                self._update_status("XY connection failed")
        except Exception as e:
            logger.error(f"Error connecting XY stage: {e}")
            self._update_status("XY connection error")
    
    def connect_z_stage(self):
        """Connect to the Z stage hardware."""
        if self.z_is_connected:
            return
        self._update_status("Connecting Z stage...")
        try:
            # Ensure normal cursor during connection
            QApplication.restoreOverrideCursor()
            
            if self.manager.connect_z():
                self.z_is_connected = True
                self._set_z_controls_enabled(True)
                self._update_connection_status()
                self.update_position_display()
            else:
                self._update_status("Z connection failed")
        except Exception as e:
            logger.error(f"Error connecting Z stage: {e}")
            self._update_status("Z connection error")

    def disconnect_stage(self):
        """Disconnect from the XY stage hardware."""
        if not self.is_connected:
            return
        try:
            self.manager.disconnect()
            self.is_connected = False
            self._set_controls_enabled(False)
            self._update_connection_status()
        except Exception as e:
            logger.error(f"Error disconnecting XY stage: {e}")
    
    def disconnect_z_stage(self):
        """Disconnect from the Z stage hardware."""
        if not self.z_is_connected:
            return
        try:
            self.manager.disconnect_z()
            self.z_is_connected = False
            self._set_z_controls_enabled(False)
            self._update_connection_status()
        except Exception as e:
            logger.error(f"Error disconnecting Z stage: {e}")
    
    def _update_connection_status(self):
        """Update status display based on connection states."""
        # Only update if not currently showing Moving or Out of Range
        if self._is_moving or self._error_timer.isActive():
            return
            
        if self.is_connected and self.z_is_connected:
            self._update_status("Ready")
        elif self.is_connected:
            self._update_status("XY Ready")
        elif self.z_is_connected:
            self._update_status("Z Ready")
        else:
            self._update_status("Disconnected")
            
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
        """Enable/disable XY stage controls."""
        for w in [
            self.up_btn, self.down_btn, self.left_btn, self.right_btn,
            self.step_size, self.x_abs, self.y_abs, self.go_btn
        ]:
            w.setEnabled(enabled)
        
        # Connect/disconnect buttons are hidden, no need to update them
    
    def _set_z_controls_enabled(self, enabled: bool):
        """Enable/disable Z stage controls."""
        for w in [
            self.z_up_btn, self.z_down_btn, self.z_step_size,
            self.z_abs, self.z_go_btn
        ]:
            w.setEnabled(enabled)

    def _update_status(self, text: str):
        """Update status display with circle indicator."""
        self.status_display.set_status(text)
    
    def _on_moving_complete(self):
        """Called 1 second after movement starts to transition to Ready."""
        self._is_moving = False
        self._update_status("Ready")
    
    def _on_error_timeout(self):
        """Called 2 seconds after error to transition back to Connected/Ready."""
        if self.is_connected and self.z_is_connected:
            self._update_status("Ready")
        elif self.is_connected:
            self._update_status("XY Ready")
        elif self.z_is_connected:
            self._update_status("Z Ready")
    
    def _start_move(self):
        """Start a movement - show Moving status for 1 second, then Ready."""
        # Clear any pending error timeout
        if self._error_timer.isActive():
            self._error_timer.stop()
        
        # Set moving status
        self._is_moving = True
        self._update_status("Moving")
        
        # Schedule transition to Ready after 1 second
        self._moving_timer.start(1000)
    
    def _handle_error(self):
        """Handle movement error - show Out of Range for 2 seconds."""
        import time
        self._last_error_time = time.time()
        
        # Stop any active timers
        if self._moving_timer.isActive():
            self._moving_timer.stop()
        
        self._is_moving = False
        self._update_status("Out of Range")
        
        # Schedule transition back to Ready after 2 seconds
        self._error_timer.start(2000)

    # --- Position and movement ---
    def update_position_display(self):
        """Update the position display with current stage coordinates."""
        # Update XY position
        if self.is_connected:
            x, y = self.manager.get_position()
            if x is not None and y is not None:
                self.x_pos_label.setText(f"X: {x:.3f} mm")
                self.y_pos_label.setText(f"Y: {y:.3f} mm")
        
        # Update Z position
        if self.z_is_connected:
            z = self.manager.get_z_position()
            if z is not None:
                self.z_pos_label.setText(f"Z: {z:.3f} µm")

    def _move_relative(self, dx: float = 0.0, dy: float = 0.0):
        if not self.is_connected:
            return
            
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        # Start move (shows Moving status for 1 second)
        self._start_move()
        
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
        success = self.manager.move_relative(dx=dx if dx else None, dy=dy if dy else None)
        
        # Handle error if move failed
        if not success:
            self._handle_error()
        
        self.update_position_display()

    # These UI methods map button clicks to hardware directions
    # Since the camera view is rotated, we need to map the buttons
    # differently from the actual hardware directions
    
    def _get_step_size(self):
        """Get current step size from spinbox."""
        return self.step_size.value()

    def _on_step_changed_spinbox(self, value):
        """Handle step size change from spinbox."""
        # Keep global shortcuts (Ctrl+Arrows) in sync with widget step
        self.manager.default_step_size = value
    
    def _on_z_step_changed_spinbox(self, value):
        """Handle Z step size change from spinbox."""
        # Keep global shortcuts (Ctrl+U/D) in sync with widget step
        self.manager.default_z_step_size = value

    def move_up(self):
        # UP arrow button pressed -> Y decreases (camera rotated)
        logger.debug("UP button pressed -> Y-")
        self._move_relative(dy=-self._get_step_size())

    def move_down(self):
        # DOWN arrow button pressed -> Y increases (camera rotated)
        logger.debug("DOWN button pressed -> Y+")
        self._move_relative(dy=self._get_step_size())

    def move_left(self):
        # LEFT arrow button pressed -> X decreases
        logger.debug("LEFT button pressed -> X-")
        self._move_relative(dx=-self._get_step_size())

    def move_right(self):
        # RIGHT arrow button pressed -> X increases
        logger.debug("RIGHT button pressed -> X+")
        self._move_relative(dx=self._get_step_size())

    def go_to_position(self):
        if not self.is_connected:
            logger.warning("Cannot move to position: XY stage not connected")
            self._update_status("XY stage not connected")
            return
            
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        target_x = self.x_abs.value()
        target_y = self.y_abs.value()
        
        # Check if target is out of range
        xy_limit = 12.5
        if abs(target_x) > xy_limit or abs(target_y) > xy_limit:
            logger.error(f"Target position out of range: X={target_x:.3f}, Y={target_y:.3f}")
            self._handle_error()
            return
        
        logger.info(f"Moving to position: X={target_x:.3f} mm, Y={target_y:.3f} mm")
        self._start_move()
        
        success = self.manager.move_to(x=target_x, y=target_y)
        
        if not success:
            logger.error("Move failed")
            self._handle_error()
        else:
            logger.info("Move completed successfully")
        
        self.update_position_display()

    def _on_step_changed(self):
        """Deprecated - kept for compatibility."""
        pass
    
    def _on_z_step_changed(self):
        """Deprecated - kept for compatibility."""
        pass
    
    # --- Z-stage movement methods ---
    def move_z_up(self):
        """Move Z stage up."""
        if not self.z_is_connected:
            return
        
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        logger.debug("Z UP button pressed")
        self._start_move()
        
        success = self.manager.move_z_up()
        
        if not success:
            logger.warning("Z UP move rejected - likely out of range")
            self._handle_error()
        
        self.update_position_display()
    
    def move_z_down(self):
        """Move Z stage down."""
        if not self.z_is_connected:
            return
        
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        logger.debug("Z DOWN button pressed")
        self._start_move()
        
        success = self.manager.move_z_down()
        
        if not success:
            logger.warning("Z DOWN move rejected - likely out of range")
            self._handle_error()
        
        self.update_position_display()
    
    def go_to_z_position(self):
        """Move Z stage to absolute position."""
        if not self.z_is_connected:
            logger.warning("Cannot move Z to position: Z stage not connected")
            self._update_status("Z stage not connected")
            return
        
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        target_z = self.z_abs.value()
        
        # Check if target is out of range
        z_min = 0.0
        z_max = 200.0
        if target_z < z_min or target_z > z_max:
            logger.error(f"Target Z position out of range: {target_z:.3f} µm")
            self._handle_error()
            return
        
        logger.info(f"Moving Z to position: {target_z:.3f} µm")
        self._start_move()
        
        success = self.manager.move_z_to(target_z)
        
        if not success:
            logger.error("Z move failed - likely out of range")
            self._handle_error()
        else:
            logger.info("Z move completed successfully")
        
        self.update_position_display()
        
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
        if not self.z_is_connected:
            QTimer.singleShot(100, self.connect_z_stage)