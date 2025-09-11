"""
XY Stage Manager singleton for the AFS Tracking System.
Provides global access to the XY stage controller.
"""

from typing import Optional, Tuple

from src.controllers.xy_stage.xy_stage_controller import XYStageController
from src.logger import get_logger

# Get logger for this module
logger = get_logger("stage_manager")


class StageManager:
    """
    Singleton manager for XY stage hardware.
    Provides a global access point to the XY stage controller with helpers
    for relative and absolute movement and shared defaults.
    """
    _instance = None

    def __init__(self):
        if StageManager._instance is not None:
            raise Exception("StageManager is a singleton! Use get_instance() instead.")
        # Hardware controller and state
        self._stage: Optional[XYStageController] = None
        self._is_connected: bool = False
        # Defaults
        self._default_step_size: float = 0.01  # mm (10 µm)
        self._default_speed: float = 0.5       # mm/s
        StageManager._instance = self

    # --- Singleton access ---
    @classmethod
    def get_instance(cls) -> "StageManager":
        if cls._instance is None:
            cls._instance = StageManager()
        return cls._instance

    # --- Connection management ---
    def connect(self) -> bool:
        if self._is_connected and self._stage is not None:
            return True
        try:
            logger.info("Connecting to XY stage via manager")
            self._stage = XYStageController()
            if self._stage.connect():
                self._is_connected = True
                logger.info("XY stage connected via manager")
                return True
            logger.error("Failed to connect to XY stage via manager")
            self._stage = None
            return False
        except Exception as e:
            logger.error(f"Error connecting to XY stage: {e}")
            self._stage = None
            return False

    def disconnect(self) -> None:
        if not self._is_connected or self._stage is None:
            return
        try:
            self._stage.disconnect()
            self._stage = None
            self._is_connected = False
            logger.info("XY stage disconnected via manager")
        except Exception as e:
            logger.error(f"Error disconnecting from XY stage: {e}")

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._stage is not None

    @property
    def stage(self) -> Optional[XYStageController]:
        return self._stage

    # --- Defaults ---
    @property
    def default_step_size(self) -> float:
        return self._default_step_size

    @default_step_size.setter
    def default_step_size(self, value: float) -> None:
        if value > 0:
            self._default_step_size = float(value)
            logger.debug(f"Default step size set to {value:.3f} mm")

    @property
    def default_speed(self) -> float:
        return self._default_speed

    @default_speed.setter
    def default_speed(self, value: float) -> None:
        if value > 0:
            self._default_speed = float(value)
            logger.debug(f"Default speed set to {value:.3f} mm/s")

    # --- Queries ---
    def get_position(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.is_connected:
            if not self.connect():
                return None, None
        try:
            x_pos = self._stage.get_position(1)
            y_pos = self._stage.get_position(2)
            return x_pos, y_pos
        except Exception as e:
            logger.error(f"Error getting stage position: {e}")
            return None, None

    # --- Relative movement helpers ---
    def move_relative(self, dx: Optional[float] = None, dy: Optional[float] = None) -> bool:
        """Move relatively in X and/or Y using manager default speed."""
        if not self.is_connected:
            if not self.connect():
                return False
        try:
            ok = True
            if dx is not None and dx != 0:
                ok = self._stage.move_axis(1, dx, self._default_speed) and ok
            if dy is not None and dy != 0:
                ok = self._stage.move_axis(2, dy, self._default_speed) and ok
            return ok
        except Exception as e:
            logger.error(f"Error in relative move: {e}")
            return False

    # --- Absolute movement helper ---
    def move_to(self, x: Optional[float] = None, y: Optional[float] = None) -> bool:
        """Move to absolute X/Y in mm (based on internally tracked positions)."""
        cur_x, cur_y = self.get_position()
        if cur_x is None or cur_y is None:
            return False
        dx = (x - cur_x) if x is not None else None
        dy = (y - cur_y) if y is not None else None
        return self.move_relative(dx, dy)

    # --- Directional helpers using default step ---
    # Adjusted for camera rotation:
    # UP key = LEFT on stage (negative X)
    # DOWN key = RIGHT on stage (positive X)
    # LEFT key = UP on stage (positive Y)
    # RIGHT key = DOWN on stage (negative Y)
    
    def move_up(self) -> bool:
        # UP key should move LEFT (negative X)
        return self.move_relative(dx=-self._default_step_size)

    def move_down(self) -> bool:
        # DOWN key should move RIGHT (positive X)
        return self.move_relative(dx=self._default_step_size)

    def move_left(self) -> bool:
        # LEFT key should move UP (positive Y)
        return self.move_relative(dy=self._default_step_size)

    def move_right(self) -> bool:
        # RIGHT key should move DOWN (negative Y)
        return self.move_relative(dy=-self._default_step_size)