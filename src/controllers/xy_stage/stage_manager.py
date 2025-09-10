"""
XY Stage Manager singleton for the AFS Tracking System.
Provides global access to the XY stage controller.
"""

from src.controllers.xy_stage.xy_stage_controller import XYStageController
from src.logger import get_logger

# Get logger for this module
logger = get_logger("stage_manager")


class StageManager:
    """
    Singleton manager for XY stage hardware.
    Provides a global access point to the XY stage controller.
    """
    _instance = None
    _stage = None
    _is_connected = False
    _default_step_size = 0.01  # Default step size in mm (10µm)
    _default_speed = 0.5  # Default speed in mm/s

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the StageManager."""
        if cls._instance is None:
            cls._instance = StageManager()
        return cls._instance

    def __init__(self):
        """Initialize the stage manager."""
        # Only allow one instance
        if StageManager._instance is not None:
            raise Exception("StageManager is a singleton! Use get_instance() instead.")
        StageManager._instance = self

    def connect(self):
        """Connect to the XY stage if not already connected."""
        if self._is_connected and self._stage is not None:
            return True

        try:
            logger.info("Connecting to XY stage via manager")
            self._stage = XYStageController()
            if self._stage.connect():
                self._is_connected = True
                logger.info("XY stage connected via manager")
                return True
            else:
                logger.error("Failed to connect to XY stage via manager")
                self._stage = None
                return False
        except Exception as e:
            logger.error(f"Error connecting to XY stage: {str(e)}")
            self._stage = None
            return False

    def disconnect(self):
        """Disconnect from the XY stage."""
        if not self._is_connected or self._stage is None:
            return

        try:
            self._stage.disconnect()
            self._stage = None
            self._is_connected = False
            logger.info("XY stage disconnected via manager")
        except Exception as e:
            logger.error(f"Error disconnecting from XY stage: {str(e)}")

    @property
    def is_connected(self):
        """Get the connection status."""
        return self._is_connected and self._stage is not None

    @property
    def stage(self):
        """Get the XY stage controller."""
        return self._stage

    @property
    def default_step_size(self):
        """Get the default step size."""
        return self._default_step_size

    @default_step_size.setter
    def default_step_size(self, value):
        """Set the default step size."""
        if value > 0:
            self._default_step_size = value
            logger.debug(f"Default step size set to {value} mm")

    @property
    def default_speed(self):
        """Get the default movement speed."""
        return self._default_speed

    @default_speed.setter
    def default_speed(self, value):
        """Set the default movement speed."""
        if value > 0:
            self._default_speed = value
            logger.debug(f"Default speed set to {value} mm/s")

    def move_up(self):
        """Move the stage up by the default step size."""
        if not self.is_connected:
            if not self.connect():
                return False
        try:
            success = self._stage.move_axis(2, self._default_step_size, self._default_speed)
            if success:
                logger.debug(f"Moved up by {self._default_step_size} mm")
            return success
        except Exception as e:
            logger.error(f"Error moving stage up: {str(e)}")
            return False

    def move_down(self):
        """Move the stage down by the default step size."""
        if not self.is_connected:
            if not self.connect():
                return False
        try:
            success = self._stage.move_axis(2, -self._default_step_size, self._default_speed)
            if success:
                logger.debug(f"Moved down by {self._default_step_size} mm")
            return success
        except Exception as e:
            logger.error(f"Error moving stage down: {str(e)}")
            return False

    def move_left(self):
        """
        Move the stage left by the default step size.
        Auto-connects to stage if not already connected.
        Uses positive X axis value (axis 1) to move left.
        """
        if not self.is_connected:
            if not self.connect():
                return False
        try:
            # Using positive value to move left (depends on stage orientation)
            success = self._stage.move_axis(1, self._default_step_size, self._default_speed)
            if success:
                logger.debug(f"Moved left by {self._default_step_size} mm")
            return success
        except Exception as e:
            logger.error(f"Error moving stage left: {str(e)}")
            return False

    def move_right(self):
        """
        Move the stage right by the default step size.
        Auto-connects to stage if not already connected.
        Uses negative X axis value (axis 1) to move right.
        """
        if not self.is_connected:
            if not self.connect():
                return False
        try:
            # Using negative value to move right (depends on stage orientation)
            success = self._stage.move_axis(1, -self._default_step_size, self._default_speed)
            if success:
                logger.debug(f"Moved right by {self._default_step_size} mm")
            return success
        except Exception as e:
            logger.error(f"Error moving stage right: {str(e)}")
            return False

    def get_position(self):
        """Get the current position of the stage."""
        if not self.is_connected:
            if not self.connect():
                return None, None
        try:
            x_pos = self._stage.get_position(1)
            y_pos = self._stage.get_position(2)
            return x_pos, y_pos
        except Exception as e:
            logger.error(f"Error getting stage position: {str(e)}")
            return None, None
