"""
XY Stage controller for the AFS Tracking System.
This module provides an interface to the XY stage hardware using MCL's MicroDrive DLL.
"""

import ctypes
import time
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
    
    def __init__(self, dll_path=r"C:\Program Files\Mad City Labs\MicroDrive\Labview Executables\MicroDrive.dll"):
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

    def connect(self):
        """Connect to the XY stage."""
        try:
            self.micro = ctypes.CDLL(self.dll_path)
            self._set_function_signatures()
            result = self._detect_stage()
            
            if result:
                self._is_disconnected = False
                return True
            else:
                logger.error("No XY stage detected")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to XY stage: {e}")
            return False

    def _set_function_signatures(self):
        """Set the function signatures for the MicroDrive DLL."""
        # Basic functions
        self.micro.MCL_GrabAllHandles.restype = ctypes.c_int
        self.micro.MCL_GetAllHandles.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.micro.MCL_GetAllHandles.restype = ctypes.c_int
        self.micro.MCL_GetProductID.argtypes = [ctypes.POINTER(ctypes.c_ushort), ctypes.c_int]
        self.micro.MCL_GetProductID.restype = ctypes.c_int
        self.micro.MCL_GetAxisInfo.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
        self.micro.MCL_GetAxisInfo.restype = ctypes.c_int
        
        # Movement functions
        self.micro.MCL_MDMoveR.argtypes = [
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int
        ]
        self.micro.MCL_MDMoveR.restype = ctypes.c_int
        
        # Optional functions - check if they exist
        try:
            # Try to set up additional functions if available
            if hasattr(self.micro, 'MCL_MDSetSettlingTime'):
                self.micro.MCL_MDSetSettlingTime.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
                self.micro.MCL_MDSetSettlingTime.restype = ctypes.c_int
        except Exception:
            # Ignore if functions aren't available
            pass

    def _detect_stage(self):
        """Detect the XY stage."""
        num = self.micro.MCL_GrabAllHandles()
        if num <= 0:
            return False

        handles = (ctypes.c_int * num)()
        self.micro.MCL_GetAllHandles(handles, num)

        for h in handles:
            pid = ctypes.c_ushort()
            self.micro.MCL_GetProductID(ctypes.byref(pid), h)
            axis_bitmap = ctypes.c_ubyte()
            self.micro.MCL_GetAxisInfo(ctypes.byref(axis_bitmap), h)

            if pid.value in [9475, 9472, 9473]:  # XY stage IDs
                self.handle = h
                self.axes_bitmap = axis_bitmap.value
                
                # Try to configure settling time if the function exists
                try:
                    if hasattr(self.micro, 'MCL_MDSetSettlingTime'):
                        for axis in range(1, 3):  # Axes 1 and 2 (X and Y)
                            if self.axes_bitmap & (1 << (axis - 1)):
                                self.micro.MCL_MDSetSettlingTime(
                                    ctypes.c_int(axis), 
                                    ctypes.c_double(0.1),  # 100ms settling time
                                    ctypes.c_int(self.handle)
                                )
                except Exception:
                    # Ignore if function isn't available
                    pass
                
                logger.info(f"Connected (handle={h}, axes={bin(axis_bitmap.value)})")
                return True

        return False

    def disconnect(self):
        """Disconnect from the XY stage."""
        if self.handle is not None and not self._is_disconnected:
            # No direct disconnect method in DLL, but we can set flags
            self.handle = None
            self._is_disconnected = True
            logger.info("Disconnected")

    def get_position(self, axis):
        """
        Get the current position of the specified axis.
        
        Args:
            axis (int): The axis to get the position for (1=X, 2=Y)
            
        Returns:
            float: The position in mm
        """
        # Get current position for specified axis (1=X, 2=Y)
        if axis == 1:
            return self.x_position
        elif axis == 2:
            return self.y_position
        else:
            logger.error(f"Invalid axis: {axis}. Use 1 for X or 2 for Y.")
            return None

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

    def _get_error_description(self, error_code):
        """Convert MicroDrive error codes to human-readable descriptions."""
        # Store the last error code for diagnostic purposes
        self._last_error_code = error_code
        
        error_codes = {
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
        return error_codes.get(error_code, "Unknown error")
    
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
        
        # Move in square pattern
        stage.move_axis(1, side_length)  # Move right
        time.sleep(0.5)
        
        stage.move_axis(2, side_length)  # Move up
        time.sleep(0.5)
        
        stage.move_axis(1, -side_length)  # Move left
        time.sleep(0.5)
        
        stage.move_axis(2, -side_length)  # Move down (back to origin)
        
    except Exception as e:
        logger.error(f"{str(e)}")
    
    finally:
        # Always disconnect when done
        stage.disconnect()
