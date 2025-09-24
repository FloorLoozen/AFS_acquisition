"""
Camera controller module for the AFS Tracking System.
Provides access to the IDS camera using pyueye.
"""

from pyueye import ueye
import numpy as np

# Fix imports to work with both direct run and module run

# Import our custom logger
from src.utils.logger import get_logger

# Get logger for this module
logger = get_logger("camera")


class CameraController:
    """
    Controller for IDS cameras using the uEye API.
    Handles camera initialization, frame capture, and cleanup.
    """
    def __init__(self, camera_id=0):
        """
        Initialize a camera controller.
        
        Args:
            camera_id (int): ID of the camera to connect to
        """
        self.h_cam = ueye.HIDS(camera_id)
        self.mem_ptr = ueye.c_mem_p()
        self.mem_id = ueye.int()
        self.width = 0
        self.height = 0
        self.bits_per_pixel = 24  # BGR8 = 3 bytes
        self._is_open = False
        self._is_disconnected = False
        self.camera_id = camera_id
        
    @property
    def is_open(self):
        """
        Check if the camera is currently open and ready for use.
        
        Returns:
            bool: True if camera is open, False otherwise
        """
        return self._is_open and not self._is_disconnected

    def initialize(self):
        """
        Initialize the camera connection and setup memory for frame capture.
        
        Performs these steps:
        1. Initialize physical camera connection
        2. Get sensor info and dimensions
        3. Set color mode to BGR8 (8-bit per channel)
        4. Allocate memory for image data
        5. Set active memory region
        6. Start video capture
        
        Returns:
            bool: True if all steps succeeded, False otherwise
        """
# Camera initialization starting
        
        try:
            # Reset state if we're reinitializing
            if self._is_disconnected:
                self._is_disconnected = False
                
            # Initialize camera
            ret = ueye.is_InitCamera(self.h_cam, None)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"InitCamera failed: {ret}")
                return False
    
            # Get sensor info
            sensor_info = ueye.SENSORINFO()
            ret = ueye.is_GetSensorInfo(self.h_cam, sensor_info)
            if ret != ueye.IS_SUCCESS:
                logger.error("GetSensorInfo failed")
                self.close()  # Clean up
                return False
    
            self.width = sensor_info.nMaxWidth
            self.height = sensor_info.nMaxHeight
            
            # Set color mode
            ret = ueye.is_SetColorMode(self.h_cam, ueye.IS_CM_BGR8_PACKED)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"SetColorMode failed: {ret}")
                self.close()  # Clean up
                return False
    
            # Allocate memory for image data
            ret = ueye.is_AllocImageMem(
                self.h_cam, self.width, self.height,
                self.bits_per_pixel, self.mem_ptr, self.mem_id
            )
            if ret != ueye.IS_SUCCESS:
                logger.error(f"AllocImageMem failed: {ret}")
                self.close()  # Clean up
                return False
    
            # Set active memory
            ret = ueye.is_SetImageMem(self.h_cam, self.mem_ptr, self.mem_id)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"SetImageMem failed: {ret}")
                self.close()  # Clean up
                return False
    
            # Start capturing video
            ret = ueye.is_CaptureVideo(self.h_cam, ueye.IS_WAIT)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"CaptureVideo failed: {ret}")
                self.close()  # Clean up
                return False
    
# Connection logged by camera widget
            self._is_open = True
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            self.close()  # Clean up on any exception
            return False

    def get_frame(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray: BGR image array or None if capture failed
        """
        # Quick check for camera state before attempting capture
        if not self.is_open:
            # Don't log warnings for closed cameras, handled by the widget
            return None
            
        try:
            # Capture a single frame from the camera
            ret = ueye.is_FreezeVideo(self.h_cam, ueye.IS_WAIT)
            if ret != ueye.IS_SUCCESS:
                # No need to log every frame failure, it's too verbose
                # Only mark camera as closed if we get a critical error
                if ret in [ueye.IS_INVALID_CAMERA_HANDLE, ueye.IS_NO_SUCCESS]:
                    self._is_open = False
                return None
    
            height = int(self.height)
            width = int(self.width)
            channels = int(self.bits_per_pixel / 8)
    
            array = ueye.get_data(
                self.mem_ptr, width, height,
                self.bits_per_pixel, width * channels, copy=True  # Use copy=True for safer memory handling
            )
    
            frame = np.frombuffer(array, dtype=np.uint8)
            return frame.reshape((height, width, channels))
            
        except Exception as e:
            # Mark camera as closed if we get an exception during capture
            self._is_open = False
            return None

    def disconnect(self):
        """
        Disconnect from the camera.
        
        Returns:
            None
        """
        # Be silent if not open to avoid 'already closed' noise
        if not self._is_open or self._is_disconnected:
            return None
            
        ret = ueye.is_ExitCamera(self.h_cam)
        if ret == ueye.IS_SUCCESS:
            self._is_disconnected = True
# Disconnection logged by camera widget
        elif ret != 1:  # Only log if it's a real failure; code 1 means already closed
            logger.error(f"Disconnect failed (ID: {self.h_cam.value}, code: {ret})")
            
        self._is_open = False
        return None

    def close(self):
        """
        Close the camera connection and clean up resources.
        This is a more thorough cleanup than disconnect().
        """
        # Best-effort teardown; suppress extra logs
        if self._is_disconnected:
            return
            
        try:
            ueye.is_StopLiveVideo(self.h_cam, ueye.IS_FORCE_VIDEO_STOP)
        except Exception as e:
            logger.debug(f"Error stopping live video: {e}")
            
        try:
            ueye.is_FreeImageMem(self.h_cam, self.mem_ptr, self.mem_id)
        except Exception as e:
            logger.debug(f"Error freeing image memory: {e}")
            
        try:
            ueye.is_ExitCamera(self.h_cam)
        except Exception as e:
            logger.debug(f"Error exiting camera: {e}")
            
        self._is_open = False
        self._is_disconnected = True
# Resources cleaned up


# Example usage if this file is run directly
if __name__ == "__main__":
    import cv2
    
    # Initialize camera
    camera = CameraController()
    if not camera.initialize():
        logger.error("Failed to initialize camera")
        exit(1)
    
    try:
        # Capture frames until 'q' is pressed
        logger.info("Press 'q' to exit")
        while True:
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow("Camera Feed", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        logger.error(f"Camera error: {str(e)}")
    
    finally:
        # Always disconnect when done
        camera.close()
        cv2.destroyAllWindows()
