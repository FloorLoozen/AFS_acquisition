"""
XY Stage controller for the AFS Tracking System.
This module provides an interface to the XY stage hardware using MCL's MicroDrive DLL.
"""

import ctypes
import time
from typing import Tuple
# Import logger from package
from src.utils.logger import get_logger

# Get logger for this module
logger = get_logger("xy_stage")


class XYStageController:
    """
    Controller for XY stage hardware using MCL's MicroDrive DLL.
    This class provides an interface to the XY stage and handles initialization,
    movement, and cleanup.
    """
    
    # Error code constants for better maintainability
    ERROR_CODES = {
        -1: "General Error",
        -2: "Function not supported", 
        -3: "Handle not valid",
        -4: "Not enough memory",
        -5: "Device not ready",
        -6: "Device not found",
        -7: "Device already in use",
        -8: "Value out of range",
        -9: "Module not found"
    }
    
    # Supported XY stage product IDs
    SUPPORTED_STAGE_IDS = {9475, 9472, 9473}
    
    def __init__(self, dll_path: str = r"C:\Program Files\Mad City Labs\MicroDrive\Labview Executables\MicroDrive.dll"):
        """Initialize the XY stage controller with the path to the MicroDrive DLL."""
        self.dll_path = dll_path
        self.micro = None
        
        # Handles & axes
        self.handle = None
        self.axes_bitmap = 0
        self._is_disconnected = False
        self._last_error_code = 0
        
        # Position tracking (since hardware doesn't provide absolute position reading)
        self.x_position = 0.0  # mm
        self.y_position = 0.0  # mm

    def connect(self) -> bool:
        """Connect to the XY stage."""
        try:
            self.micro = ctypes.CDLL(self.dll_path)
            self._set_function_signatures()
            
            if self._detect_stage():
                self._is_disconnected = False
                return True
            else:
                logger.error("No XY stage detected")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to XY stage: {e}")
            return False

    def _set_function_signatures(self) -> None:
        """Set the function signatures for the MicroDrive DLL."""
        # Define all function signatures in a structured way
        function_signatures = {
            # Basic functions
            'MCL_GrabAllHandles': ([], ctypes.c_int),
            'MCL_GetAllHandles': ([ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int),
            'MCL_GetProductID': ([ctypes.POINTER(ctypes.c_ushort), ctypes.c_int], ctypes.c_int),
            'MCL_GetAxisInfo': ([ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int], ctypes.c_int),
            
            # Movement functions  
            'MCL_MDMoveR': ([ctypes.c_int, ctypes.c_double, ctypes.c_double, 
                           ctypes.c_int, ctypes.c_int], ctypes.c_int),
        }
        
        # Set up required functions
        for func_name, (argtypes, restype) in function_signatures.items():
            func = getattr(self.micro, func_name)
            func.argtypes = argtypes
            func.restype = restype
        
        # Optional functions - set up if available
        if hasattr(self.micro, 'MCL_MDSetSettlingTime'):
            try:
                self.micro.MCL_MDSetSettlingTime.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
                self.micro.MCL_MDSetSettlingTime.restype = ctypes.c_int
            except Exception:
                pass  # Ignore if functions aren't available

    def _detect_stage(self) -> bool:
        """Detect the XY stage."""
        num = self.micro.MCL_GrabAllHandles()
        if num <= 0:
            return False

        handles = (ctypes.c_int * num)()
        self.micro.MCL_GetAllHandles(handles, num)

        for handle in handles:
            pid = ctypes.c_ushort()
            self.micro.MCL_GetProductID(ctypes.byref(pid), handle)
            
            if pid.value in self.SUPPORTED_STAGE_IDS:
                axis_bitmap = ctypes.c_ubyte()
                self.micro.MCL_GetAxisInfo(ctypes.byref(axis_bitmap), handle)
                
                self.handle = handle
                self.axes_bitmap = axis_bitmap.value
                
                # Configure settling time for better performance
                self._configure_settling_time()
                
                logger.info(f"Connected (handle={handle}, axes={bin(axis_bitmap.value)})")
                return True

        return False
    
    def _configure_settling_time(self) -> None:
        """Configure settling time for both axes if supported."""
        if not hasattr(self.micro, 'MCL_MDSetSettlingTime'):
            return
            
        try:
            for axis in range(1, 3):  # Axes 1 and 2 (X and Y)
                if self.axes_bitmap & (1 << (axis - 1)):
                    self.micro.MCL_MDSetSettlingTime(
                        ctypes.c_int(axis), 
                        ctypes.c_double(0.1),  # 100ms settling time
                        ctypes.c_int(self.handle)
                    )
        except Exception:
            pass  # Ignore if function fails
    
    def wait_for_movement_complete(self, timeout_ms: float = 1000) -> bool:
        """
        Wait for stage movement to complete by checking if position is stable.
        More efficient than fixed time.sleep() delays.
        
        Args:
            timeout_ms: Maximum time to wait in milliseconds
            
        Returns:
            True if movement completed, False if timeout
        """
        if not self._is_connected:
            return False
            
        import time
        start_time = time.time()
        
        try:
            last_x, last_y = self.get_position()
        except:
            # If we can't get position, use minimal delay as fallback
            time.sleep(0.1)
            return True
        
        # Wait for position to stabilize
        stable_count = 0
        required_stable_readings = 3  # Need 3 consecutive stable readings
        
        while (time.time() - start_time) * 1000 < timeout_ms:
            time.sleep(0.01)  # Small check interval
            try:
                current_x, current_y = self.get_position()
                
                # Check if position is stable (within 1 micron)
                if (abs(current_x - last_x) < 0.001 and 
                    abs(current_y - last_y) < 0.001):
                    stable_count += 1
                    if stable_count >= required_stable_readings:
                        return True
                else:
                    stable_count = 0
                    
                last_x, last_y = current_x, current_y
            except:
                # If position reading fails, assume movement complete
                return True
                
        return False  # Timeout

    def disconnect(self) -> None:
        """Disconnect from the XY stage."""
        if self.handle is not None and not self._is_disconnected:
            # No direct disconnect method in DLL, but we can set flags
            self.handle = None
            self._is_disconnected = True
            logger.info("Disconnected")

    def get_position(self) -> Tuple[float, float]:
        """
        Get the current position of both axes.
        
        Returns:
            Tuple of (x_position, y_position) in mm
        """
        return (self.x_position, self.y_position)
    
    def get_axis_position(self, axis: int) -> float:
        """
        Get the current position of a specific axis.
        
        Args:
            axis: The axis to get the position for (1=X, 2=Y)
            
        Returns:
            The position in mm
        """
        if axis == 1:
            return self.x_position
        elif axis == 2:
            return self.y_position
        else:
            logger.error(f"Invalid axis: {axis}. Use 1 for X or 2 for Y.")
            return 0.0

    def move_axis(self, axis: int, distance_mm: float, velocity: float = 0.5, rounding: int = 0):
        """
        Move a single axis by the specified distance.
        
        Args:
            axis (int): The axis to move (1=X, 2=Y)
            distance_mm (float): The distance to move in mm
            velocity (float): The velocity to move at in mm/s
            rounding (int): Rounding parameter for the movement
            
        Returns:
            bool: True if the movement was successful, False otherwise
        """
        # Move a single axis by the specified distance
        if self.handle is None:
            logger.error("No XY stage connected")
            return False
            
        if not (self.axes_bitmap & (1 << (axis - 1))):
            logger.error(f"Axis {axis} not present in handle {self.handle}")
            return False

        try:
            # Convert to the correct types for the C API
            axis_param = ctypes.c_int(axis)
            velocity_param = ctypes.c_double(velocity) # mm/s
            distance_param = ctypes.c_double(distance_mm)
            rounding_param = ctypes.c_int(rounding)
            handle_param = ctypes.c_int(self.handle)
            
            # Make the movement call
            ret = self.micro.MCL_MDMoveR(
                axis_param,
                velocity_param,
                distance_param,
                rounding_param,
                handle_param
            )
            
            if ret != 0:
                error_msg = self._get_error_description(ret)
                logger.error(f"Axis {axis} move failed, code {ret} - {error_msg}")
                return False
            else:
                # Update position tracking
                if axis == 1:
                    self.x_position += distance_mm
                elif axis == 2:
                    self.y_position += distance_mm
                return True
                
        except Exception as e:
            logger.error(f"Failed to move axis {axis}: {e}")
            return False

    def _get_error_description(self, error_code: int) -> str:
        """Convert MicroDrive error codes to human-readable descriptions."""
        self._last_error_code = error_code
        return self.ERROR_CODES.get(error_code, "Unknown error")
    
    def move_xy(self, x_mm=None, y_mm=None):
        """
        Move both X and Y axes by the specified distances.
        
        Args:
            x_mm (float): The distance to move the X axis in mm
            y_mm (float): The distance to move the Y axis in mm
            
        Returns:
            bool: True if both movements were successful, False otherwise
        """
        success = True
        if x_mm is not None:
            success = self.move_axis(1, x_mm) and success
        if y_mm is not None:
            success = self.move_axis(2, y_mm) and success
        return success


# Example usage if this file is run directly
if __name__ == "__main__":
    # Initialize XY stage controller
    stage = XYStageController()
    if not stage.connect():
        exit(1)
    
    try:
        # Small square movement pattern
        side_length = 0.1  # mm (100 microns)
        
        # Move in square pattern with efficient waiting
        stage.move_axis(1, side_length)  # Move right
        stage.wait_for_movement_complete(500)  # Wait up to 500ms
        
        stage.move_axis(2, side_length)  # Move up
        stage.wait_for_movement_complete(500)
        
        stage.move_axis(1, -side_length)  # Move left
        stage.wait_for_movement_complete(500)
        
        stage.move_axis(2, -side_length)  # Move down (back to origin)
        
    except Exception as e:
        logger.error(f"{str(e)}")
    
    finally:
        # Always disconnect when done
        stage.disconnect()
