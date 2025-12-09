"""
Z Stage controller for AFS Acquisition.
This module provides an interface to the Z stage hardware using MCL's NanoDrive DLL.
"""

import ctypes
import os
import time
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exceptions import StageError
from src.utils.validation import validate_positive_number

# Get logger for this module
logger = get_logger("z_stage")


class ZStageController:
    """
    Controller for Z stage hardware using MCL's NanoDrive DLL.
    This class provides an interface to the Z stage and handles initialization,
    movement, and cleanup.
    """
    
    # Error code constants
    ERROR_CODES = {
        -1: "General Error",
        -2: "Device does not exist",
        -3: "Device is not attached",
        -4: "Usage error"
    }
    
    def __init__(self, dll_path: str = r"C:\Program Files\Mad City Labs\NanoDrive\Labview Executable Examples\Madlib.dll"):
        """Initialize the Z stage controller with the path to the NanoDrive DLL."""
        self.dll_path = dll_path
        self.nano = None
        self.handle = None
        self._is_disconnected = False
        self._last_error_code = 0
        
        # Z-axis range (typical for NanoDrive, in micrometers)
        self.z_min = 0.0
        self.z_max = 200.0  # 200 µm typical range
    
    def connect(self) -> bool:
        """Connect to the Z stage."""
        try:
            if not os.path.exists(self.dll_path):
                logger.error(f"DLL not found at {self.dll_path}")
                return False
            
            # Load DLL
            self.nano = ctypes.WinDLL(self.dll_path)
            self._set_function_signatures()
            
            # Initialize handle
            self.handle = self.nano.MCL_InitHandle()
            
            if self.handle < 0:
                error_msg = self._get_error_description(self.handle)
                logger.error(f"Failed to initialize NanoDrive handle: {error_msg}")
                return False
            
            self._is_disconnected = False
            logger.info(f"Connected to Z stage (handle={self.handle})")
            
            # Read initial position
            try:
                z_pos = self.get_position()
                logger.info(f"Initial Z position: {z_pos:.3f} µm")
            except Exception as pos_err:
                logger.warning(f"Could not read initial Z position (non-critical): {pos_err}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Z stage: {e}")
            return False
    
    def _set_function_signatures(self) -> None:
        """Set the function signatures for the NanoDrive DLL."""
        # MCL_InitHandle
        self.nano.MCL_InitHandle.restype = ctypes.c_int
        self.nano.MCL_InitHandle.argtypes = []
        
        # MCL_SingleWriteZ
        self.nano.MCL_SingleWriteZ.argtypes = [ctypes.c_double, ctypes.c_int]
        self.nano.MCL_SingleWriteZ.restype = ctypes.c_int
        
        # MCL_SingleReadZ
        self.nano.MCL_SingleReadZ.argtypes = [ctypes.c_int]
        self.nano.MCL_SingleReadZ.restype = ctypes.c_double
        
        # MCL_ReleaseHandle (for proper cleanup)
        if hasattr(self.nano, 'MCL_ReleaseHandle'):
            self.nano.MCL_ReleaseHandle.argtypes = [ctypes.c_int]
            self.nano.MCL_ReleaseHandle.restype = None
    
    @property
    def is_connected(self) -> bool:
        """Check if stage is connected."""
        return self.handle is not None and not self._is_disconnected
    
    def disconnect(self) -> None:
        """Disconnect from the Z stage."""
        if self.handle is not None and not self._is_disconnected:
            try:
                if hasattr(self.nano, 'MCL_ReleaseHandle'):
                    self.nano.MCL_ReleaseHandle(self.handle)
            except Exception as e:
                logger.warning(f"Error releasing handle: {e}")
            finally:
                self.handle = None
                self._is_disconnected = True
                logger.info("Disconnected from Z stage")
    
    def get_position(self) -> float:
        """
        Get the current Z position.
        
        Returns:
            The Z position in micrometers (µm)
        """
        if not self.is_connected:
            raise StageError("Z stage not connected")
        
        try:
            z_pos = self.nano.MCL_SingleReadZ(self.handle)
            return z_pos
        except Exception as e:
            logger.error(f"Failed to read Z position: {e}")
            raise StageError(f"Failed to read Z position: {e}")
    
    def move_to(self, position_um: float) -> bool:
        """
        Move to an absolute Z position.
        
        Args:
            position_um (float): The target position in micrometers (µm)
            
        Returns:
            bool: True if the movement was successful, False otherwise
        """
        if not self.is_connected:
            logger.error("Z stage not connected")
            return False
        
        # Validate position is within range
        if position_um < self.z_min or position_um > self.z_max:
            logger.error(f"Position {position_um:.3f} µm is out of range [{self.z_min:.3f}, {self.z_max:.3f}]")
            return False
        
        try:
            ret = self.nano.MCL_SingleWriteZ(ctypes.c_double(position_um), self.handle)
            
            if ret != 0:
                error_msg = self._get_error_description(ret)
                logger.error(f"Z move failed, code {ret} - {error_msg}")
                return False
            
            # Allow stage to settle (hardware stabilization time)
            time.sleep(0.05)
            return True
            
        except Exception as e:
            logger.error(f"Failed to move Z stage: {e}")
            return False
    
    def move_relative(self, distance_um: float) -> bool:
        """
        Move by a relative distance from current position.
        
        Args:
            distance_um (float): The distance to move in micrometers (µm)
            
        Returns:
            bool: True if the movement was successful, False otherwise
        """
        try:
            current_pos = self.get_position()
            target_pos = current_pos + distance_um
            return self.move_to(target_pos)
        except Exception as e:
            logger.error(f"Failed to move Z stage relatively: {e}")
            return False
    
    def _get_error_description(self, error_code: int) -> str:
        """Convert NanoDrive error codes to human-readable descriptions."""
        self._last_error_code = error_code
        return self.ERROR_CODES.get(error_code, "Unknown error")
    
    def get_settings(self) -> dict:
        """Get Z stage settings for metadata storage."""
        settings = {
            # dll_path removed - system-specific path not portable or scientifically relevant
            'z_range_um': [self.z_min, self.z_max],
        }
        
        if self.is_connected:
            try:
                settings['current_z_um'] = self.get_position()
            except Exception as pos_err:
                # Position read may fail if stage is moving or temporarily unavailable
                logger.debug(f"Could not read current Z position for metadata: {pos_err}")
        
        return settings


# Example usage if this file is run directly
if __name__ == "__main__":
    # Initialize Z stage controller
    stage = ZStageController()
    if not stage.connect():
        exit(1)
    
    try:
        # Read current Z
        z0 = stage.get_position()
        logger.info(f"Current Z = {z0:.3f} µm")
        
        # Move +10 µm
        target = z0 + 10
        logger.info(f"Moving to {target:.3f} µm...")
        stage.move_to(target)
        
        time.sleep(0.2)
        
        # Read again
        z1 = stage.get_position()
        logger.info(f"New Z = {z1:.3f} µm")
        
        # Move back
        logger.info(f"Moving back to {z0:.3f} µm...")
        stage.move_to(z0)
        
    except Exception as e:
        logger.error(f"{str(e)}")
    
    finally:
        # Always disconnect when done
        stage.disconnect()
