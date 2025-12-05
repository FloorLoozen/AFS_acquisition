"""
XY and Z Stage Manager singleton for AFS Acquisition.
Provides global access to the XY and Z stage controllers.
"""

from typing import Optional, Tuple, Dict, Any

from src.controllers.xy_stage_controller import XYStageController
from src.controllers.z_stage_controller import ZStageController
from src.utils.logger import get_logger

# Get logger for this module
logger = get_logger("stage_manager")


class StageManager:
    """
    Singleton manager for XY and Z stage hardware.
    Provides a global access point to the XY and Z stage controllers with helpers
    for relative and absolute movement and shared defaults.
    """
    _instance = None

    def __init__(self):
        if StageManager._instance is not None:
            raise Exception("StageManager is a singleton! Use get_instance() instead.")
        # Hardware controllers and state
        self._stage: Optional[XYStageController] = None
        self._z_stage: Optional[ZStageController] = None
        self._is_connected: bool = False
        self._z_is_connected: bool = False
        # Defaults
        self._default_step_size: float = 0.01  # mm (10 µm)
        self._default_speed: float = 0.5       # mm/s
        self._default_z_step_size: float = 1.0  # µm (1 micrometer)
        StageManager._instance = self

    # --- Singleton access ---
    @classmethod
    def get_instance(cls) -> "StageManager":
        if cls._instance is None:
            cls._instance = StageManager()
        return cls._instance

    # --- Settings access ---
    def get_stage_settings(self) -> Dict[str, Any]:
        """Get current stage settings for metadata storage.
        
        Returns:
            Dictionary containing stage configuration and state
        """
        from datetime import datetime
        
        settings = {
            'timestamp': datetime.now().isoformat(),
            'xy_is_connected': self._is_connected,
            'z_is_connected': self._z_is_connected,
            'default_step_size_mm': self._default_step_size,
            'default_speed_mm_per_s': self._default_speed,
            'default_z_step_size_um': self._default_z_step_size,
        }
        
        # XY stage settings
        if self._is_connected and self._stage:
            try:
                # Get current position
                x_pos, y_pos = self.get_position()
                settings.update({
                    'current_x_mm': x_pos,
                    'current_y_mm': y_pos,
                    'xy_stage_type': 'MCL MicroDrive',
                    'xy_controller_connected': True,
                })
                
                # Get hardware-specific settings if available
                if hasattr(self._stage, 'get_settings'):
                    hw_settings = self._stage.get_settings()
                    settings.update(hw_settings)
                    
            except Exception as e:
                logger.warning(f"Error getting XY stage settings: {e}")
                settings['xy_settings_error'] = str(e)
        else:
            settings.update({
                'xy_controller_connected': False,
                'xy_stage_type': 'Disconnected',
            })
        
        # Z stage settings
        if self._z_is_connected and self._z_stage:
            try:
                # Get current Z position
                z_pos = self.get_z_position()
                settings.update({
                    'current_z_um': z_pos,
                    'z_stage_type': 'MCL NanoDrive',
                    'z_controller_connected': True,
                })
                
                # Get hardware-specific settings if available
                if hasattr(self._z_stage, 'get_settings'):
                    z_hw_settings = self._z_stage.get_settings()
                    settings['z_stage_settings'] = z_hw_settings
                    
            except Exception as e:
                logger.warning(f"Error getting Z stage settings: {e}")
                settings['z_settings_error'] = str(e)
        else:
            settings.update({
                'z_controller_connected': False,
                'z_stage_type': 'Disconnected',
            })
        
        return settings

    # --- Connection management ---
    def connect(self) -> bool:
        """Connect to XY stage."""
        if self._is_connected and self._stage is not None:
            return True
        try:
            self._stage = XYStageController()
            if self._stage.connect():
                self._is_connected = True
                return True
            logger.error("Failed to connect to XY stage via manager")
            self._stage = None
            return False
        except Exception as e:
            logger.error(f"Error connecting to XY stage: {e}")
            self._stage = None
            return False

    def disconnect(self) -> None:
        """Disconnect from XY stage."""
        if not self._is_connected or self._stage is None:
            return
        try:
            self._stage.disconnect()
            self._stage = None
            self._is_connected = False
        except Exception as e:
            logger.error(f"Error disconnecting from XY stage: {e}")
    
    def connect_z(self) -> bool:
        """Connect to Z stage."""
        if self._z_is_connected and self._z_stage is not None:
            return True
        try:
            self._z_stage = ZStageController()
            if self._z_stage.connect():
                self._z_is_connected = True
                return True
            logger.error("Failed to connect to Z stage via manager")
            self._z_stage = None
            return False
        except Exception as e:
            logger.error(f"Error connecting to Z stage: {e}")
            self._z_stage = None
            return False

    def disconnect_z(self) -> None:
        """Disconnect from Z stage."""
        if not self._z_is_connected or self._z_stage is None:
            return
        try:
            self._z_stage.disconnect()
            self._z_stage = None
            self._z_is_connected = False
        except Exception as e:
            logger.error(f"Error disconnecting from Z stage: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if XY stage is connected."""
        return self._is_connected and self._stage is not None
    
    @property
    def z_is_connected(self) -> bool:
        """Check if Z stage is connected."""
        return self._z_is_connected and self._z_stage is not None

    @property
    def stage(self) -> Optional[XYStageController]:
        """Get XY stage controller."""
        return self._stage
    
    @property
    def z_stage(self) -> Optional[ZStageController]:
        """Get Z stage controller."""
        return self._z_stage

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
    
    @property
    def default_z_step_size(self) -> float:
        return self._default_z_step_size

    @default_z_step_size.setter
    def default_z_step_size(self, value: float) -> None:
        if value > 0:
            self._default_z_step_size = float(value)
            logger.debug(f"Default Z step size set to {value:.3f} µm")

    # --- Queries ---
    def get_position(self) -> Tuple[Optional[float], Optional[float]]:
        """Get XY stage position in mm."""
        if not self.is_connected:
            if not self.connect():
                return None, None
        try:
            return self._stage.get_position()
        except Exception as e:
            logger.error(f"Error getting stage position: {e}")
            return None, None
    
    def get_z_position(self) -> Optional[float]:
        """Get Z stage position in µm."""
        if not self.z_is_connected:
            if not self.connect_z():
                return None
        try:
            return self._z_stage.get_position()
        except Exception as e:
            logger.error(f"Error getting Z position: {e}")
            return None

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
        # UP key should move UP (negative Y, camera rotated)
        if not self.is_connected and not self.connect():
            logger.debug("Cannot move up: XY stage not connected")
            return False
        return self.move_relative(dy=-self._default_step_size)

    def move_down(self) -> bool:
        # DOWN key should move DOWN (positive Y, camera rotated)
        if not self.is_connected and not self.connect():
            logger.debug("Cannot move down: XY stage not connected")
            return False
        return self.move_relative(dy=self._default_step_size)

    def move_left(self) -> bool:
        # LEFT key should move LEFT (negative X)
        if not self.is_connected and not self.connect():
            logger.debug("Cannot move left: XY stage not connected")
            return False
        return self.move_relative(dx=-self._default_step_size)

    def move_right(self) -> bool:
        # RIGHT key should move RIGHT (positive X)
        if not self.is_connected and not self.connect():
            logger.debug("Cannot move right: XY stage not connected")
            return False
        return self.move_relative(dx=self._default_step_size)
    
    # --- Z-stage movement helpers ---
    def move_z_up(self) -> bool:
        """Move Z stage up (positive direction) by default step size."""
        if not self.z_is_connected and not self.connect_z():
            logger.debug("Cannot move Z up: Z stage not connected")
            return False
        try:
            return self._z_stage.move_relative(self._default_z_step_size)
        except Exception as e:
            logger.error(f"Error moving Z up: {e}")
            return False
    
    def move_z_down(self) -> bool:
        """Move Z stage down (negative direction) by default step size."""
        if not self.z_is_connected and not self.connect_z():
            logger.debug("Cannot move Z down: Z stage not connected")
            return False
        try:
            return self._z_stage.move_relative(-self._default_z_step_size)
        except Exception as e:
            logger.error(f"Error moving Z down: {e}")
            return False
    
    def move_z_to(self, position_um: float) -> bool:
        """Move Z stage to absolute position in µm."""
        if not self.z_is_connected and not self.connect_z():
            logger.debug("Cannot move Z to position: Z stage not connected")
            return False
        try:
            return self._z_stage.move_to(position_um)
        except Exception as e:
            logger.error(f"Error moving Z to position: {e}")
            return False