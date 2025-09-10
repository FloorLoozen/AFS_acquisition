from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QDoubleSpinBox, QSizePolicy
from PyQt5.QtCore import Qt, QTimer

from src.logger import get_logger
from src.controllers.xy_stage.xy_stage_controller import XYStageController

logger = get_logger("xy_stage_gui")


class XYStageWidget(QGroupBox):
    """Minimal XY stage control widget with arrows, step, absolute Go, and status."""

    def __init__(self, parent=None):
        super().__init__("XY Stage", parent)
        self.stage = None
        self.is_connected = False

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.init_ui()
        self.connect_stage()  # auto-connect

        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self.update_position_display)
        self.position_timer.start(500)

    def init_ui(self):
        main = QVBoxLayout(self)

        # Status row + connect controls
        status_row = QHBoxLayout()
        self.status_label = QLabel("Initializing...")
        status_row.addWidget(self.status_label, 1, Qt.AlignLeft)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_stage)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_stage)
        status_row.addWidget(self.connect_btn)
        status_row.addWidget(self.disconnect_btn)
        main.addLayout(status_row)

        # Movement arrows
        grid = QGridLayout()
        self.up_btn = QPushButton("▲");   self.up_btn.clicked.connect(self.move_up)
        self.left_btn = QPushButton("◄"); self.left_btn.clicked.connect(self.move_left)
        self.right_btn = QPushButton("►"); self.right_btn.clicked.connect(self.move_right)
        self.down_btn = QPushButton("▼");  self.down_btn.clicked.connect(self.move_down)
        grid.addWidget(self.up_btn, 0, 1)
        grid.addWidget(self.left_btn, 1, 0)
        grid.addWidget(self.right_btn, 1, 2)
        grid.addWidget(self.down_btn, 2, 1)
        main.addLayout(grid)

        # Step size
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step Size:"))
        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(0.001, 10.0)
        self.step_size.setSingleStep(0.001)
        self.step_size.setValue(0.01)
        self.step_size.setSuffix(" mm")
        step_row.addWidget(self.step_size)
        main.addLayout(step_row)

        # Absolute position + Go
        abs_row = QGridLayout()
        abs_row.addWidget(QLabel("X:"), 0, 0)
        self.x_abs = QDoubleSpinBox(); self.x_abs.setRange(-100.0, 100.0); self.x_abs.setDecimals(3); self.x_abs.setSuffix(" mm")
        abs_row.addWidget(self.x_abs, 0, 1)
        abs_row.addWidget(QLabel("Y:"), 1, 0)
        self.y_abs = QDoubleSpinBox(); self.y_abs.setRange(-100.0, 100.0); self.y_abs.setDecimals(3); self.y_abs.setSuffix(" mm")
        abs_row.addWidget(self.y_abs, 1, 1)
        self.go_btn = QPushButton("Go")
        self.go_btn.clicked.connect(self.go_to_position)
        abs_row.addWidget(self.go_btn, 2, 0, 1, 2, Qt.AlignCenter)
        main.addLayout(abs_row)

        # Hidden position labels to keep latest values
        self.x_pos_label = QLabel("X: 0.000 mm")
        self.y_pos_label = QLabel("Y: 0.000 mm")

        self.set_controls_enabled(False)

    # Connection handling
    def connect_stage(self):
        if self.is_connected:
            return
        self.update_status("Connecting...")
        try:
            self.stage = XYStageController()
            if self.stage.connect():
                self.is_connected = True
                self.update_status("Connected")
                self.set_controls_enabled(True)
                self.update_position_display()
            else:
                self.update_status("Connection Failed")
                self.stage = None
        except Exception as e:
            logger.error(f"Error connecting: {e}")
            self.update_status("Connection Error")
            self.stage = None

    def disconnect_stage(self):
        if not self.is_connected:
            return
        try:
            if self.stage:
                self.stage.disconnect()
            self.stage = None
            self.is_connected = False
            self.update_status("Disconnected")
            self.set_controls_enabled(False)
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

    def set_controls_enabled(self, enabled: bool):
        for w in [self.up_btn, self.down_btn, self.left_btn, self.right_btn, self.step_size, self.x_abs, self.y_abs, self.go_btn]:
            w.setEnabled(enabled)

    def update_status(self, text: str):
        self.status_label.setText(text)

    def update_position_display(self):
        if not self.is_connected or not self.stage:
            return
        x = self.stage.get_position(1) or 0.0
        y = self.stage.get_position(2) or 0.0
        self.x_pos_label.setText(f"X: {x:.3f} mm")
        self.y_pos_label.setText(f"Y: {y:.3f} mm")

    # Movement
    def _move(self, dx: float, dy: float):
        if not self.is_connected or not self.stage:
            return
        self.stage.move_xy(x_mm=dx if dx else None, y_mm=dy if dy else None)
        self.update_position_display()

    def move_up(self):
        self._move(0.0, self.step_size.value())

    def move_down(self):
        self._move(0.0, -self.step_size.value())

    def move_left(self):
        self._move(-self.step_size.value(), 0.0)

    def move_right(self):
        self._move(self.step_size.value(), 0.0)

    def go_to_position(self):
        if not self.is_connected or not self.stage:
            return
        cur_x = self.stage.get_position(1) or 0.0
        cur_y = self.stage.get_position(2) or 0.0
        dx = self.x_abs.value() - cur_x
        dy = self.y_abs.value() - cur_y
        self._move(dx, dy)