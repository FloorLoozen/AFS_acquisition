from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSizePolicy,
    QDialogButtonBox, QFrame, QApplication, QLineEdit, QWidget
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

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._build_ui()

        # Timer to refresh positions
        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self.update_position_display)
        self.position_timer.start(300)
        
        # Set dialog properties - with title
        self.setWindowTitle("Stage Control")
        self.setModal(False)  # Non-modal dialog so user can interact with main window
        self.resize(420, 360)  # Compact clean design
        
        # Auto-connect if possible
        self.connect_stage()
        self.connect_z_stage()

    # --- UI ---
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setSpacing(15)
        main.setContentsMargins(15, 15, 15, 15)
        
        # Status display at the top
        self.status_display = StatusDisplay()
        main.addWidget(self.status_display)
        
        # ===== XY AND Z STAGES SIDE BY SIDE WITH GRID =====
        # Using grid layout for perfect alignment
        stages_grid = QGridLayout()
        stages_grid.setHorizontalSpacing(30)
        stages_grid.setVerticalSpacing(12)
        
        # Column headers
        xy_title = QLabel("XY Stage")
        xy_title.setStyleSheet("font-weight: bold; color: #666;")
        stages_grid.addWidget(xy_title, 0, 0)
        
        z_title = QLabel("Z Stage")
        z_title.setStyleSheet("font-weight: bold; color: #666;")
        stages_grid.addWidget(z_title, 0, 1)
        
        # Row 0.5: Underline for titles
        xy_title_separator = QFrame()
        xy_title_separator.setFrameShape(QFrame.HLine)
        xy_title_separator.setFrameShadow(QFrame.Sunken)
        stages_grid.addWidget(xy_title_separator, 1, 0)
        
        z_title_separator = QFrame()
        z_title_separator.setFrameShape(QFrame.HLine)
        z_title_separator.setFrameShadow(QFrame.Sunken)
        stages_grid.addWidget(z_title_separator, 1, 1)
        
        # Row 1: Position displays
        xy_pos_widget = QWidget()
        xy_pos_layout = QVBoxLayout(xy_pos_widget)
        xy_pos_layout.setContentsMargins(0, 0, 0, 0)
        xy_pos_layout.setSpacing(4)
        
        x_layout = QHBoxLayout()
        x_label = QLabel("X")
        x_label.setStyleSheet("color: #666;")
        self.x_pos_label = QLabel("0.000 mm")
        self.x_pos_label.setStyleSheet("color: #333;")
        x_layout.addWidget(x_label)
        x_layout.addWidget(self.x_pos_label)
        x_layout.addStretch()
        
        y_layout = QHBoxLayout()
        y_label = QLabel("Y")
        y_label.setStyleSheet("color: #666;")
        self.y_pos_label = QLabel("0.000 mm")
        self.y_pos_label.setStyleSheet("color: #333;")
        y_layout.addWidget(y_label)
        y_layout.addWidget(self.y_pos_label)
        y_layout.addStretch()
        
        xy_pos_layout.addLayout(x_layout)
        xy_pos_layout.addLayout(y_layout)
        stages_grid.addWidget(xy_pos_widget, 2, 0)
        
        z_pos_layout = QHBoxLayout()
        z_label = QLabel("Z")
        z_label.setStyleSheet("color: #666;")
        self.z_pos_label = QLabel("0.000 µm")
        self.z_pos_label.setStyleSheet("color: #333;")
        z_pos_layout.addWidget(z_label)
        z_pos_layout.addWidget(self.z_pos_label)
        z_pos_layout.addStretch()
        z_pos_widget = QWidget()
        z_pos_widget.setLayout(z_pos_layout)
        stages_grid.addWidget(z_pos_widget, 2, 1)
        
        # Row 2: Move labels
        xy_move_label = QLabel("Move")
        xy_move_label.setStyleSheet("font-weight: bold; color: #666;")
        stages_grid.addWidget(xy_move_label, 3, 0)
        
        z_move_label = QLabel("Move")
        z_move_label.setStyleSheet("font-weight: bold; color: #666;")
        stages_grid.addWidget(z_move_label, 3, 1)
        
        # Row 3: Movement controls
        grid = QGridLayout()
        grid.setSpacing(6)
        
        self.up_btn = QPushButton("↑")
        self.up_btn.clicked.connect(self.move_up)
        self.up_btn.setFixedSize(40, 40)
        self.up_btn.setStyleSheet("font-size: 14px;")
        
        self.left_btn = QPushButton("←")
        self.left_btn.clicked.connect(self.move_left)
        self.left_btn.setFixedSize(40, 40)
        self.left_btn.setStyleSheet("font-size: 14px;")
        
        self.right_btn = QPushButton("→")
        self.right_btn.clicked.connect(self.move_right)
        self.right_btn.setFixedSize(40, 40)
        self.right_btn.setStyleSheet("font-size: 14px;")
        
        self.down_btn = QPushButton("↓")
        self.down_btn.clicked.connect(self.move_down)
        self.down_btn.setFixedSize(40, 40)
        self.down_btn.setStyleSheet("font-size: 14px;")
        
        # Tooltips
        self.up_btn.setToolTip("Ctrl+Up")
        self.down_btn.setToolTip("Ctrl+Down")
        self.left_btn.setToolTip("Ctrl+Left")
        self.right_btn.setToolTip("Ctrl+Right")
        
        grid.addWidget(self.up_btn, 0, 1)
        grid.addWidget(self.left_btn, 1, 0)
        grid.addWidget(self.right_btn, 1, 2)
        grid.addWidget(self.down_btn, 2, 1)
        
        xy_arrows_widget = QWidget()
        xy_arrows_widget.setLayout(grid)
        stages_grid.addWidget(xy_arrows_widget, 4, 0)
        
        # Z movement controls
        z_controls_layout = QHBoxLayout()
        z_controls_layout.setSpacing(6)
        
        z_buttons_layout = QVBoxLayout()
        z_buttons_layout.setSpacing(6)
        
        self.z_up_btn = QPushButton("↑ Up")
        self.z_up_btn.clicked.connect(self.move_z_up)
        self.z_up_btn.setFixedSize(115, 35)
        self.z_up_btn.setToolTip("Ctrl+U")
        
        self.z_down_btn = QPushButton("↓ Down")
        self.z_down_btn.clicked.connect(self.move_z_down)
        self.z_down_btn.setFixedSize(115, 35)
        self.z_down_btn.setToolTip("Ctrl+D")
        
        z_buttons_layout.addWidget(self.z_up_btn)
        z_buttons_layout.addWidget(self.z_down_btn)
        
        z_controls_layout.addLayout(z_buttons_layout)
        z_controls_layout.addStretch()
        
        z_controls_widget = QWidget()
        z_controls_widget.setLayout(z_controls_layout)
        stages_grid.addWidget(z_controls_widget, 4, 1)
        
        # Row 5: Step inputs with inline labels
        xy_step_layout = QHBoxLayout()
        xy_step_label = QLabel("Step")
        xy_step_label.setStyleSheet("color: #666;")
        xy_step_label.setMinimumWidth(35)
        self.step_size = QLineEdit()
        self.step_size.setFixedWidth(70)
        self.step_size.setText(f"{self.manager.default_step_size:.3f}")
        self.step_size.setPlaceholderText("mm")
        self.step_size.editingFinished.connect(self._on_step_changed)
        xy_step_layout.addWidget(xy_step_label)
        xy_step_layout.addWidget(self.step_size)
        xy_step_layout.addStretch()
        xy_step_widget = QWidget()
        xy_step_widget.setLayout(xy_step_layout)
        stages_grid.addWidget(xy_step_widget, 5, 0)
        
        z_step_layout = QHBoxLayout()
        z_step_label = QLabel("Step")
        z_step_label.setStyleSheet("color: #666;")
        z_step_label.setMinimumWidth(35)
        self.z_step_size = QLineEdit()
        self.z_step_size.setFixedWidth(70)
        self.z_step_size.setText(f"{self.manager.default_z_step_size:.1f}")
        self.z_step_size.setPlaceholderText("µm")
        self.z_step_size.editingFinished.connect(self._on_z_step_changed)
        z_step_layout.addWidget(z_step_label)
        z_step_layout.addWidget(self.z_step_size)
        z_step_layout.addStretch()
        z_step_widget = QWidget()
        z_step_widget.setLayout(z_step_layout)
        stages_grid.addWidget(z_step_widget, 5, 1)
        
        # Row 6: Separators
        xy_separator = QFrame()
        xy_separator.setFrameShape(QFrame.HLine)
        xy_separator.setFrameShadow(QFrame.Sunken)
        stages_grid.addWidget(xy_separator, 6, 0)
        
        z_separator = QFrame()
        z_separator.setFrameShape(QFrame.HLine)
        z_separator.setFrameShadow(QFrame.Sunken)
        stages_grid.addWidget(z_separator, 6, 1)
        
        # Row 7: Go to labels
        xy_goto_label = QLabel("Go to")
        xy_goto_label.setStyleSheet("font-weight: bold; color: #666;")
        stages_grid.addWidget(xy_goto_label, 7, 0)
        
        z_goto_label = QLabel("Go to")
        z_goto_label.setStyleSheet("font-weight: bold; color: #666;")
        stages_grid.addWidget(z_goto_label, 7, 1)
        
        # Row 8: Absolute position inputs
        xy_abs_widget = QWidget()
        xy_abs_layout = QVBoxLayout(xy_abs_widget)
        xy_abs_layout.setContentsMargins(0, 0, 0, 0)
        xy_abs_layout.setSpacing(4)
        
        x_abs_layout = QHBoxLayout()
        x_abs_label = QLabel("X")
        x_abs_label.setStyleSheet("color: #666;")
        x_abs_label.setMinimumWidth(35)
        self.x_abs = QLineEdit()
        self.x_abs.setFixedWidth(70)
        self.x_abs.setText("0.000")
        self.x_abs.setPlaceholderText("mm")
        x_abs_layout.addWidget(x_abs_label)
        x_abs_layout.addWidget(self.x_abs)
        x_abs_layout.addStretch()
        
        y_abs_layout = QHBoxLayout()
        y_abs_label = QLabel("Y")
        y_abs_label.setStyleSheet("color: #666;")
        y_abs_label.setMinimumWidth(35)
        self.y_abs = QLineEdit()
        self.y_abs.setFixedWidth(70)
        self.y_abs.setText("0.000")
        self.y_abs.setPlaceholderText("mm")
        y_abs_layout.addWidget(y_abs_label)
        y_abs_layout.addWidget(self.y_abs)
        y_abs_layout.addStretch()
        
        xy_abs_layout.addLayout(x_abs_layout)
        xy_abs_layout.addLayout(y_abs_layout)
        stages_grid.addWidget(xy_abs_widget, 8, 0)
        
        z_input_layout = QHBoxLayout()
        z_abs_label_field = QLabel("Z")
        z_abs_label_field.setStyleSheet("color: #666;")
        z_abs_label_field.setMinimumWidth(35)
        self.z_abs = QLineEdit()
        self.z_abs.setFixedWidth(70)
        self.z_abs.setText("0.000")
        self.z_abs.setPlaceholderText("µm")
        z_input_layout.addWidget(z_abs_label_field)
        z_input_layout.addWidget(self.z_abs)
        z_input_layout.addStretch()
        z_input_widget = QWidget()
        z_input_widget.setLayout(z_input_layout)
        stages_grid.addWidget(z_input_widget, 8, 1)
        
        # Row 9: Go buttons
        self.go_btn = QPushButton("Go")
        self.go_btn.clicked.connect(self.go_to_position)
        self.go_btn.setFixedHeight(30)
        stages_grid.addWidget(self.go_btn, 9, 0)
        
        self.z_go_btn = QPushButton("Go")
        self.z_go_btn.clicked.connect(self.go_to_z_position)
        self.z_go_btn.setFixedHeight(30)
        stages_grid.addWidget(self.z_go_btn, 9, 1)
        
        # Add grid to main layout
        main.addLayout(stages_grid)
        
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
        if self.is_connected and self.z_is_connected:
            self._update_status("Connected")
        elif self.is_connected:
            self._update_status("XY Connected")
        elif self.z_is_connected:
            self._update_status("Z Connected")
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

    # --- Position and movement ---
    def update_position_display(self):
        """Update the position display with current stage coordinates."""
        # Update XY position
        if self.is_connected:
            x, y = self.manager.get_position()
            if x is not None and y is not None:
                self.x_pos_label.setText(f"{x:.3f} mm")
                self.y_pos_label.setText(f"{y:.3f} mm")
        
        # Update Z position
        if self.z_is_connected:
            z = self.manager.get_z_position()
            if z is not None:
                self.z_pos_label.setText(f"{z:.3f} µm")

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
    
    def _get_step_size(self):
        """Get current step size from text field, fallback to manager default."""
        try:
            return float(self.step_size.text())
        except ValueError:
            return self.manager.default_step_size

    def move_up(self):
        # UP arrow button pressed - need to move LEFT relative to camera view
        logger.debug("UP button pressed -> Moving LEFT relative to camera view")
        self._move_relative(dx=-self._get_step_size())  # Move LEFT (negative X)

    def move_down(self):
        # DOWN arrow button pressed - need to move RIGHT relative to camera view
        logger.debug("DOWN button pressed -> Moving RIGHT relative to camera view")
        self._move_relative(dx=self._get_step_size())  # Move RIGHT (positive X)

    def move_left(self):
        # LEFT arrow button pressed - need to move UP relative to camera view
        logger.debug("LEFT button pressed -> Moving UP relative to camera view")
        self._move_relative(dy=self._get_step_size())  # Move UP (positive Y)

    def move_right(self):
        # RIGHT arrow button pressed - need to move DOWN relative to camera view
        logger.debug("RIGHT button pressed -> Moving DOWN relative to camera view")
        self._move_relative(dy=-self._get_step_size())  # Move DOWN (negative Y)

    def go_to_position(self):
        if not self.is_connected:
            logger.warning("Cannot move to position: XY stage not connected")
            self._update_status("XY stage not connected")
            return
            
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        try:
            x_text = self.x_abs.text().strip()
            y_text = self.y_abs.text().strip()
            
            if not x_text or not y_text:
                logger.warning("Empty position values")
                return
                
            target_x = float(x_text)
            target_y = float(y_text)
        except ValueError as e:
            logger.warning(f"Invalid position values entered: {e}")
            self._update_status("Invalid position values")
            return
        
        logger.info(f"Moving to position: X={target_x:.3f} mm, Y={target_y:.3f} mm")
        self._update_status(f"Moving to X={target_x:.3f}, Y={target_y:.3f}...")
        
        success = self.manager.move_to(x=target_x, y=target_y)
        
        if success:
            logger.info("Move completed successfully")
            self._update_status("Move completed")
        else:
            logger.error("Move failed")
            self._update_status("Move failed")

    def _on_step_changed(self):
        """Handle step size change."""
        try:
            value = float(self.step_size.text())
            if 0.001 <= value <= 10.0:
                # Keep global shortcuts (Ctrl+Arrows) in sync with widget step
                self.manager.default_step_size = value
            else:
                # Reset to previous valid value
                self.step_size.setText(f"{self.manager.default_step_size:.3f}")
        except ValueError:
            # Reset to previous valid value
            self.step_size.setText(f"{self.manager.default_step_size:.3f}")
    
    def _on_z_step_changed(self):
        """Handle Z step size change."""
        try:
            value = float(self.z_step_size.text())
            if 0.1 <= value <= 100.0:
                # Keep global shortcuts (Ctrl+U/D) in sync with widget step
                self.manager.default_z_step_size = value
            else:
                # Reset to previous valid value
                self.z_step_size.setText(f"{self.manager.default_z_step_size:.1f}")
        except ValueError:
            # Reset to previous valid value
            self.z_step_size.setText(f"{self.manager.default_z_step_size:.1f}")
    
    # --- Z-stage movement methods ---
    def move_z_up(self):
        """Move Z stage up."""
        if not self.z_is_connected:
            return
        
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        logger.debug("Z UP button pressed")
        self.manager.move_z_up()
        self.update_position_display()
    
    def move_z_down(self):
        """Move Z stage down."""
        if not self.z_is_connected:
            return
        
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        logger.debug("Z DOWN button pressed")
        self.manager.move_z_down()
        self.update_position_display()
    
    def go_to_z_position(self):
        """Move Z stage to absolute position."""
        if not self.z_is_connected:
            logger.warning("Cannot move Z to position: Z stage not connected")
            self._update_status("Z stage not connected")
            return
        
        # Ensure normal cursor during movement
        QApplication.restoreOverrideCursor()
        
        try:
            z_text = self.z_abs.text().strip()
            
            if not z_text:
                logger.warning("Empty Z position value")
                return
                
            target_z = float(z_text)
        except ValueError as e:
            logger.warning(f"Invalid Z position value entered: {e}")
            self._update_status("Invalid Z position")
            return
        
        logger.info(f"Moving Z to position: {target_z:.3f} µm")
        self._update_status(f"Moving Z to {target_z:.3f} µm...")
        
        success = self.manager.move_z_to(target_z)
        
        if success:
            logger.info("Z move completed successfully")
            self._update_status("Z move completed")
        else:
            logger.error("Z move failed")
            self._update_status("Z move failed")
        
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