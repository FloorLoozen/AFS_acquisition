"""
High-performance camera controller module with multi-threaded video acquisition.
Provides background frame capture with queuing to reduce lag and increase frame rates.
"""

import threading
import time
import queue
import numpy as np
from typing import Optional, Tuple, Any, Dict, List
from dataclasses import dataclass
from datetime import datetime
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

from src.utils.logger import get_logger

# Try to import pyueye, fall back gracefully
try:
    from pyueye import ueye
    import ctypes
    PYUEYE_AVAILABLE = True
except ImportError:
    PYUEYE_AVAILABLE = False
    ueye = None
    ctypes = None

logger = get_logger("camera")


@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame: np.ndarray
    timestamp: float
    frame_number: int
    camera_id: int


class FramePool:
    """Memory pool for reusing frame buffers to reduce GC pressure."""
    
    def __init__(self, frame_shape: Tuple[int, int, int], pool_size: int = 5):
        """Initialize frame pool.
        
        Args:
            frame_shape: (height, width, channels) of frames
            pool_size: Number of frames to pre-allocate
        """
        self.frame_shape = frame_shape
        self.pool_size = pool_size
        self._available_frames: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        
        # Pre-allocate frame buffers
        for _ in range(pool_size):
            frame = np.zeros(frame_shape, dtype=np.uint8)
            self._available_frames.put(frame)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get a frame buffer from the pool."""
        try:
            return self._available_frames.get_nowait()
        except queue.Empty:
            # Pool exhausted, create new frame (will be GC'd)
            return np.zeros(self.frame_shape, dtype=np.uint8)
    
    def return_frame(self, frame: np.ndarray) -> None:
        """Return a frame buffer to the pool."""
        if frame.shape == self.frame_shape and self._available_frames.qsize() < self.pool_size:
            try:
                self._available_frames.put_nowait(frame)
            except queue.Full:
                pass  # Pool full, let frame be garbage collected


class CameraController:
    """
    High-performance camera controller with background frame capture.
    
    Features:
    - Background thread for continuous frame capture
    - Thread-safe queue for frame buffering
    - Configurable frame dropping to maintain real-time performance
    - Statistics tracking (FPS, dropped frames, etc.)
    - Test pattern mode when no camera hardware is available
    """
    
    def __init__(self, camera_id: int = 0, max_queue_size: int = 10):
        """
        Initialize the threaded camera controller.
        
        Args:
            camera_id: ID of the camera to connect to
            max_queue_size: Maximum number of frames to buffer in queue
        """
        self.camera_id = camera_id
        self.max_queue_size = max_queue_size
        
        # Camera hardware interface
        self.h_cam = None
        self.mem_ptr = None
        self.mem_id = None
        self.width = 0
        self.height = 0
        self.bits_per_pixel = 24  # BGR8
        
        # Thread control
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False
        self.thread_lock = threading.Lock()
        
        # Frame queue (thread-safe)
        self.frame_queue: queue.Queue[FrameData] = queue.Queue(maxsize=max_queue_size)
        
        # Statistics
        self.stats_lock = threading.Lock()
        self.frame_count = 0
        self.dropped_frames = 0  # Frames dropped due to full queue during capture
        self.discarded_frames = 0  # Frames discarded during get_latest_frame (normal for real-time)
        self.last_fps_time = time.time()
        self.last_fps_count = 0
        self.current_fps = 0.0
        
        # State flags
        self.is_initialized = False
        self.use_test_pattern = False  # Always try hardware first
        self.test_frame_counter = 0
        
        # Error tracking
        self.last_error = None
        self.consecutive_errors = 0
        
        # Frame pool for memory optimization (initialized later when we know frame size)
        self.frame_pool: Optional[FramePool] = None
        
        # Thread pool for parallel operations
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="camera_worker")
        
        logger.debug(f"CameraController created (ID: {camera_id}, queue_size: {max_queue_size})")
    
    @property
    def is_running(self) -> bool:
        """Check if capture thread is running."""
        with self.thread_lock:
            return self.running
    
    def get_camera_settings(self) -> Dict[str, Any]:
        """
        Extract comprehensive camera settings for metadata storage.
        
        Returns:
            Dictionary containing all camera parameters and settings
        """
        settings = self._get_basic_camera_info()
        
        if not self.use_test_pattern and PYUEYE_AVAILABLE and self.h_cam is not None:
            try:
                settings.update(self._get_hardware_camera_settings())
            except Exception as e:
                logger.warning(f"Error extracting camera settings: {e}")
                settings['settings_extraction_error'] = str(e)
        else:
            settings.update(self._get_test_pattern_settings())
        
        return settings
    
    def _get_basic_camera_info(self) -> Dict[str, Any]:
        """Get basic camera information."""
        return {
            'camera_id': self.camera_id,
            'use_test_pattern': self.use_test_pattern,
            'timestamp': datetime.now().isoformat(),
            'width': self.width,
            'height': self.height,
            'bits_per_pixel': self.bits_per_pixel,
        }
    
    def _get_hardware_camera_settings(self) -> Dict[str, Any]:
        """Get hardware camera settings from actual camera."""
        settings = {}
        
        # Get sensor information
        settings.update(self._get_sensor_info())
        
        # Get timing and exposure settings
        settings.update(self._get_timing_settings())
        
        # Get image processing settings
        settings.update(self._get_image_processing_settings())
        
        # Get camera identification
        settings.update(self._get_camera_identification())
        
        # Get optional temperature (if supported)
        settings.update(self._get_optional_settings())
        
        return settings
    
    def _get_sensor_info(self) -> Dict[str, Any]:
        """Get sensor hardware information."""
        settings = {}
        try:
            sensor_info = ueye.SENSORINFO()
            ret = ueye.is_GetSensorInfo(self.h_cam, sensor_info)
            if ret == ueye.IS_SUCCESS:
                settings.update({
                    'sensor_name': sensor_info.strSensorName.decode('utf-8').strip(),
                    'sensor_id': sensor_info.SensorID,
                    'sensor_max_width': sensor_info.nMaxWidth,
                    'sensor_max_height': sensor_info.nMaxHeight,
                    'sensor_color_mode': sensor_info.nColorMode,
                })
        except Exception as e:
            logger.debug(f"Could not get sensor info: {e}")
        return settings
    
    def _get_timing_settings(self) -> Dict[str, Any]:
        """Get timing-related camera settings."""
        settings = {}
        
        # Pixel clock
        try:
            pixel_clock = ueye.c_uint()
            ret = ueye.is_PixelClock(self.h_cam, ueye.IS_PIXELCLOCK_CMD_GET, pixel_clock, 4)
            if ret == ueye.IS_SUCCESS:
                settings['pixel_clock_mhz'] = pixel_clock.value
        except Exception as e:
            logger.debug(f"Could not get pixel clock: {e}")
        
        # Frame rate
        try:
            fps_ptr = ueye.DOUBLE()
            ret = ueye.is_SetFrameRate(self.h_cam, ueye.IS_GET_FRAMERATE, fps_ptr)
            if ret == ueye.IS_SUCCESS:
                settings['frame_rate_fps'] = fps_ptr.value
        except Exception as e:
            logger.debug(f"Could not get frame rate: {e}")
        
        # Exposure time
        try:
            exposure = ueye.DOUBLE()
            ret = ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exposure, 8)
            if ret == ueye.IS_SUCCESS:
                settings['exposure_ms'] = exposure.value
        except Exception as e:
            logger.debug(f"Could not get exposure: {e}")
        
        return settings
    
    def _get_image_processing_settings(self) -> Dict[str, Any]:
        """Get image processing related settings."""
        settings = {}
        
        # Gain settings
        try:
            gain_master = ueye.c_int()
            ret = ueye.is_SetHardwareGain(self.h_cam, ueye.IS_GET_MASTER_GAIN, gain_master, 
                                        ueye.c_int(), ueye.c_int())
            if ret == ueye.IS_SUCCESS:
                settings['gain_master'] = gain_master.value
            else:
                settings['gain_master'] = 'unavailable'
        except Exception as e:
            logger.debug(f"Could not get gain settings: {e}")
            settings['gain_master'] = 'unavailable'
        
        # Color mode (known value since we configure it)
        settings['color_mode'] = 'IS_CM_BGR8_PACKED'
        
        # AOI (Area of Interest)
        try:
            rect_aoi = ueye.IS_RECT()
            ret = ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
            if ret == ueye.IS_SUCCESS:
                settings.update({
                    'aoi_x': rect_aoi.s32X,
                    'aoi_y': rect_aoi.s32Y,
                    'aoi_width': rect_aoi.s32Width,
                    'aoi_height': rect_aoi.s32Height,
                })
        except Exception as e:
            logger.debug(f"Could not get AOI: {e}")
        
        return settings
    
    def _get_camera_identification(self) -> Dict[str, Any]:
        """Get camera identification information."""
        settings = {}
        try:
            cam_info = ueye.CAMINFO()
            ret = ueye.is_GetCameraInfo(self.h_cam, cam_info)
            if ret == ueye.IS_SUCCESS:
                settings.update({
                    'camera_serial': cam_info.SerNo.decode('utf-8').strip(),
                    'camera_version': cam_info.Version.decode('utf-8').strip(),
                    'camera_date': cam_info.Date.decode('utf-8').strip(),
                })
        except Exception as e:
            logger.debug(f"Could not get camera info: {e}")
        return settings
    
    def _get_optional_settings(self) -> Dict[str, Any]:
        """Get optional camera settings that may not be supported."""
        settings = {}
        
        # Temperature (if supported)
        try:
            temperature = ueye.c_int()
            ret = ueye.is_SetTemperature(self.h_cam, ueye.IS_GET_TEMPERATURE, temperature)
            if ret == ueye.IS_SUCCESS:
                settings['temperature_celsius'] = temperature.value
        except Exception:
            pass  # Temperature not supported on all cameras
        
        return settings
    
    def _get_test_pattern_settings(self) -> Dict[str, Any]:
        """Get settings for test pattern mode."""
        return {
            'sensor_name': 'Test Pattern Generator',
            'frame_rate_fps': 30.0,
            'exposure_ms': 33.33,
        }
    
    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Get the frame dimensions (height, width, channels)."""
        if self.use_test_pattern:
            return (480, 640, 3)
        return (int(self.height), int(self.width), 3)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current capture statistics."""
        with self.stats_lock:
            return {
                'fps': self.current_fps,
                'total_frames': self.frame_count,
                'dropped_frames': self.dropped_frames,  # Capture drops (bad)
                'discarded_frames': self.discarded_frames,  # Display drops (normal for real-time)
                'queue_size': self.frame_queue.qsize(),
                'max_queue_size': self.max_queue_size,
                'capture_drop_rate': self.dropped_frames / max(1, self.frame_count) * 100,
                'display_drop_rate': self.discarded_frames / max(1, self.frame_count) * 100,
                'use_test_pattern': self.use_test_pattern,
                'consecutive_errors': self.consecutive_errors
            }
    
    def initialize(self) -> bool:
        """
        Initialize camera hardware or test pattern mode.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            logger.warning("Camera already initialized")
            return True
        
        # Always try hardware first if pyueye is available
        if PYUEYE_AVAILABLE:
            if self._initialize_hardware():
                self.use_test_pattern = False
                self.is_initialized = True
                return True
            else:
                logger.warning("Hardware initialization failed, falling back to test pattern")
                self.use_test_pattern = True
        else:
            logger.info("pyueye not available, using test pattern mode")
            self.use_test_pattern = True
        
        # Fall back to test pattern mode
        self.is_initialized = True
        
        # Initialize frame pool for test pattern (standard resolution)
        if self.frame_pool is None:
            self.frame_pool = FramePool((480, 640, 3), pool_size=3)
        
        logger.info("Using test pattern mode")
        return True
    
    def _initialize_hardware(self) -> bool:
        """Initialize camera hardware. Returns True if successful."""
        if not PYUEYE_AVAILABLE:
            return False
        
        try:
            # Create camera handle
            self.h_cam = ueye.HIDS(self.camera_id)
            self.mem_ptr = ueye.c_mem_p()
            self.mem_id = ueye.int()
            
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
                self._cleanup_hardware()
                return False
            
            # Convert to regular integers to avoid ctypes issues
            self.width = int(sensor_info.nMaxWidth)
            self.height = int(sensor_info.nMaxHeight)
            
            # Set color mode
            ret = ueye.is_SetColorMode(self.h_cam, ueye.IS_CM_BGR8_PACKED)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"SetColorMode failed: {ret}")
                self._cleanup_hardware()
                return False
            
            # Allocate memory
            ret = ueye.is_AllocImageMem(
                self.h_cam, self.width, self.height,
                self.bits_per_pixel, self.mem_ptr, self.mem_id
            )
            if ret != ueye.IS_SUCCESS:
                logger.error(f"AllocImageMem failed: {ret}")
                self._cleanup_hardware()
                return False
            
            # Set active memory
            ret = ueye.is_SetImageMem(self.h_cam, self.mem_ptr, self.mem_id)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"SetImageMem failed: {ret}")
                self._cleanup_hardware()
                return False
            
            # Optimize camera settings for maximum frame rate
            try:
                # Set pixel clock to maximum for best performance
                pixel_clock_range = (ueye.c_uint * 3)()
                ret = ueye.is_PixelClock(self.h_cam, ueye.IS_PIXELCLOCK_CMD_GET_RANGE, pixel_clock_range, 12)
                if ret == ueye.IS_SUCCESS:
                    max_pixel_clock = pixel_clock_range[1]  # Maximum value
                    ret = ueye.is_PixelClock(self.h_cam, ueye.IS_PIXELCLOCK_CMD_SET, max_pixel_clock, 4)
                    if ret == ueye.IS_SUCCESS:
                        logger.debug(f"Set pixel clock to maximum: {max_pixel_clock} MHz")
                
                # Try to get and set maximum frame rate
                fps_ptr = ueye.DOUBLE()
                ret = ueye.is_SetFrameRate(self.h_cam, ueye.IS_GET_FRAMERATE, fps_ptr)
                if ret == ueye.IS_SUCCESS:
                    logger.debug(f"Current camera frame rate: {fps_ptr.value:.1f} FPS")
            except Exception as e:
                logger.debug(f"Frame rate optimization failed: {e}")
            
            # Start continuous capture
            ret = ueye.is_CaptureVideo(self.h_cam, ueye.IS_DONT_WAIT)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"CaptureVideo failed: {ret}")
                self._cleanup_hardware()
                return False
            
            # Initialize frame pool with actual camera dimensions
            if self.frame_pool is None:
                self.frame_pool = FramePool((self.height, self.width, 3), pool_size=3)
            
            return True
            
        except Exception as e:
            logger.error(f"Hardware initialization error: {e}")
            self._cleanup_hardware()
            return False
    
    def start_capture(self) -> bool:
        """
        Start background frame capture thread.
        
        Returns:
            True if capture started successfully, False otherwise
        """
        if not self.is_initialized:
            logger.error("Camera not initialized")
            return False
        
        with self.thread_lock:
            if self.running:
                logger.warning("Capture already running")
                return True
            
            # Clear any old frames from queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Reset statistics
            current_time = time.time()
            with self.stats_lock:
                self.frame_count = 0
                self.dropped_frames = 0
                self.discarded_frames = 0
                self.last_fps_time = current_time
                self.last_fps_count = 0
                self.current_fps = 0.0
                self.consecutive_errors = 0
            
            # Start capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info("Background capture started")
            return True
    
    def stop_capture(self) -> None:
        """Stop background frame capture thread."""
        with self.thread_lock:
            if not self.running:
                return
            
            self.running = False
        
        # Wait for thread to finish (with timeout)
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not stop cleanly")
        
        logger.info("Background capture stopped")
    
    def get_latest_frame(self, timeout: float = 0.1) -> Optional[FrameData]:
        """
        Get the most recent frame from the queue with frame pooling optimization.
        
        Args:
            timeout: Maximum time to wait for a frame (seconds)
            
        Returns:
            FrameData object or None if no frame available
        """
        try:
            # Get the most recent frame (may discard older frames)
            latest_frame = None
            frames_discarded = 0
            discarded_frames_list = []
            
            # Keep getting frames until queue is empty, keeping only the latest
            while True:
                try:
                    frame_data = self.frame_queue.get(timeout=timeout if latest_frame is None else 0.0)
                    if latest_frame is not None:
                        frames_discarded += 1
                        # Keep track of discarded frames for pool return
                        discarded_frames_list.append(latest_frame)
                    latest_frame = frame_data
                except queue.Empty:
                    break
            
            # Return discarded frames to pool to reduce memory allocation
            if self.frame_pool and discarded_frames_list:
                for discarded_frame in discarded_frames_list:
                    if hasattr(discarded_frame, 'frame'):
                        self.frame_pool.return_frame(discarded_frame.frame)
            
            # Update statistics for discarded frames (this is normal for real-time display)
            if frames_discarded > 0:
                with self.stats_lock:
                    self.discarded_frames += frames_discarded
            
            return latest_frame
            
        except queue.Empty:
            return None
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get the latest frame as a numpy array (for compatibility).
        
        Args:
            timeout: Maximum time to wait for a frame (seconds)
            
        Returns:
            Frame as numpy array or None if no frame available
        """
        frame_data = self.get_latest_frame(timeout)
        return frame_data.frame if frame_data else None
    
    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        logger.debug(f"Capture loop started (test_pattern: {self.use_test_pattern})")
        
        # MAXIMUM SPEED MODE - no frame rate limiting
        # target_fps = 30.0 if self.use_test_pattern else 120.0  # Higher FPS for real cameras
        # target_interval = 1.0 / target_fps
        last_capture_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # NO RATE LIMITING - generate frames as fast as possible
                # if self.use_test_pattern:
                #     time_since_last = current_time - last_capture_time
                #     if time_since_last < target_interval:
                #         time.sleep(target_interval - time_since_last)
                #         current_time = time.time()
                
                # Capture frame
                frame = self._capture_single_frame()
                if frame is not None:
                    # Create frame data
                    with self.stats_lock:
                        frame_data = FrameData(
                            frame=frame,
                            timestamp=current_time,
                            frame_number=self.frame_count,
                            camera_id=self.camera_id
                        )
                        current_frame_number = self.frame_count
                        self.frame_count += 1
                        self.consecutive_errors = 0
                    
                    # Try to add to queue (non-blocking)
                    try:
                        self.frame_queue.put(frame_data, block=False)
                        
                        # Update FPS statistics
                        with self.stats_lock:
                            # Calculate FPS every second
                            if current_time - self.last_fps_time >= 1.0:
                                frames_this_second = self.frame_count - self.last_fps_count
                                elapsed_time = current_time - self.last_fps_time
                                self.current_fps = frames_this_second / elapsed_time
                                self.last_fps_time = current_time
                                self.last_fps_count = self.frame_count
                        
                    except queue.Full:
                        # Queue is full, drop this frame (true capture drop)
                        with self.stats_lock:
                            self.dropped_frames += 1
                
                else:
                    # Frame capture failed - don't count as captured or dropped
                    with self.stats_lock:
                        self.consecutive_errors += 1
                    
                    # REMOVED delay that was slowing down frame generation
                    # if self.consecutive_errors > 10:
                    #     time.sleep(0.01)  # 10ms delay
                
                last_capture_time = current_time
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                with self.stats_lock:
                    self.consecutive_errors += 1
                # time.sleep(0.01)  # REMOVED delay that was slowing down frame generation
        
        logger.debug("Capture loop ended")
    
    def _capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera or generate test pattern."""
        if self.use_test_pattern:
            return self._generate_test_pattern()
        else:
            return self._capture_hardware_frame()
    
    def _capture_hardware_frame(self) -> Optional[np.ndarray]:
        """Capture frame from camera hardware."""
        if not PYUEYE_AVAILABLE or not self.h_cam:
            return None
        
        try:
            # Use blocking capture with very short timeout for better frame rates
            ret = ueye.is_FreezeVideo(self.h_cam, ueye.IS_WAIT)
            if ret != ueye.IS_SUCCESS:
                if ret == ueye.IS_TIMED_OUT:
                    # Timeout is normal, just return None
                    return None
                elif ret in [ueye.IS_INVALID_CAMERA_HANDLE, ueye.IS_NO_SUCCESS]:
                    # Camera disconnected, switch to test pattern
                    if not self.use_test_pattern:
                        logger.warning("Camera disconnected, switching to test pattern")
                        self.use_test_pattern = True
                return None
            
            # Get frame data
            height = int(self.height)
            width = int(self.width)
            channels = int(self.bits_per_pixel / 8)
            
            array = ueye.get_data(
                self.mem_ptr, width, height,
                self.bits_per_pixel, width * channels, copy=True
            )
            
            frame = np.frombuffer(array, dtype=np.uint8)
            return frame.reshape((height, width, channels))
            
        except Exception as e:
            return None
    
    def _generate_test_pattern(self) -> np.ndarray:
        """Generate test pattern frame with frame pooling for better performance."""
        import cv2  # Import here to avoid dependency when not needed
        
        width, height = 640, 480
        
        # Try to get frame from pool for memory efficiency
        if self.frame_pool:
            frame = self.frame_pool.get_frame()
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                # Clear the frame for reuse
                frame.fill(0)
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Grid pattern
        for i in range(0, height, 50):
            cv2.line(frame, (0, i), (width, i), (50, 50, 50), 1)
        for i in range(0, width, 50):
            cv2.line(frame, (i, 0), (i, height), (50, 50, 50), 1)
        
        # Animated elements
        t = self.test_frame_counter / 60.0  # Time in seconds at 60 FPS
        
        # Moving crosshair
        center_x = int(width // 2 + 100 * np.sin(t))
        center_y = int(height // 2 + 50 * np.cos(t * 1.3))
        cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), 2)
        cv2.line(frame, (0, center_y), (width, center_y), (0, 255, 0), 2)
        
        # Pulsing circle
        radius = int(30 + 20 * np.sin(t * 3))
        cv2.circle(frame, (width // 2, height // 2), radius, (0, 0, 255), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.test_frame_counter}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS indicator
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        self.test_frame_counter += 1
        return frame
    
    def _cleanup_hardware(self) -> None:
        """Clean up camera hardware resources."""
        if not PYUEYE_AVAILABLE:
            return
        
        try:
            if self.h_cam:
                ueye.is_StopLiveVideo(self.h_cam, ueye.IS_FORCE_VIDEO_STOP)
                ueye.is_FreeImageMem(self.h_cam, self.mem_ptr, self.mem_id)
                ueye.is_ExitCamera(self.h_cam)
        except Exception as e:
            logger.debug(f"Error during hardware cleanup: {e}")
        
        self.h_cam = None
        self.mem_ptr = None
        self.mem_id = None
    
    def _set_camera_parameter(self, param_name: str, value: Any, 
                             setter_func, success_msg: str, error_msg: str) -> bool:
        """Generic parameter setter with consistent error handling."""
        if not self.h_cam or not PYUEYE_AVAILABLE:
            logger.warning(f"Cannot set {param_name}: camera not connected")
            return False
        
        try:
            ret = setter_func()
            if ret == ueye.IS_SUCCESS:
                logger.info(success_msg)
                return True
            else:
                logger.error(f"{error_msg}: error code {ret}")
                return False
        except Exception as e:
            logger.error(f"Error setting {param_name}: {e}")
            return False
    
    def set_exposure(self, exposure_ms: float) -> bool:
        """Set camera exposure time in milliseconds."""
        exposure = ueye.DOUBLE(exposure_ms)
        return self._set_camera_parameter(
            "exposure", exposure_ms,
            lambda: ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure, 8),
            f"Exposure set to {exposure_ms:.2f} ms",
            "Failed to set exposure"
        )
    
    def set_gain(self, gain: int) -> bool:
        """Set camera master gain (0-100)."""
        return self._set_camera_parameter(
            "gain", gain,
            lambda: ueye.is_SetHardwareGain(self.h_cam, gain, ueye.IS_IGNORE_PARAMETER, 
                                          ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER),
            f"Gain set to {gain}",
            "Failed to set gain"
        )
    
    def set_framerate(self, fps: float) -> bool:
        """Set camera frame rate in frames per second."""
        new_fps = ueye.DOUBLE(fps)
        return self._set_camera_parameter(
            "framerate", fps,
            lambda: ueye.is_SetFrameRate(self.h_cam, new_fps, None),
            f"Frame rate set to {fps:.1f} fps",
            "Failed to set frame rate"
        )
    
    def apply_settings(self, settings: dict) -> dict:
        """Apply multiple camera settings at once.
        
        Args:
            settings: Dictionary with keys like 'exposure_ms', 'gain_master', 'fps', etc.
            
        Returns:
            Dictionary with success/failure status for each setting
        """
        results = {}
        
        if 'exposure_ms' in settings:
            results['exposure_ms'] = self.set_exposure(settings['exposure_ms'])
        
        if 'gain_master' in settings:
            results['gain_master'] = self.set_gain(settings['gain_master'])
        
        if 'fps' in settings:
            results['fps'] = self.set_framerate(settings['fps'])
        
        return results

    def close(self) -> None:
        """Close camera and clean up all resources efficiently."""
        logger.debug("Closing camera controller")
        
        # Stop capture thread
        self.stop_capture()
        
        # Clean up hardware
        self._cleanup_hardware()
        
        # Clear queue and return frames to pool
        frames_returned = 0
        while not self.frame_queue.empty():
            try:
                frame_data = self.frame_queue.get_nowait()
                if self.frame_pool and hasattr(frame_data, 'frame'):
                    self.frame_pool.return_frame(frame_data.frame)
                    frames_returned += 1
            except queue.Empty:
                break
        
        if frames_returned > 0:
            logger.debug(f"Returned {frames_returned} frames to pool during cleanup")
        
        # Shutdown thread pool
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)
        
        # Clear frame pool
        self.frame_pool = None
        
        # Force garbage collection to free memory
        gc.collect()
        
        self.is_initialized = False
        logger.info("Camera controller closed and resources freed")


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Create camera controller
    camera = CameraController(camera_id=0, max_queue_size=5)
    
    if not camera.initialize():
        logger.error("Failed to initialize camera")
        exit(1)
    
    if not camera.start_capture():
        logger.error("Failed to start capture")
        exit(1)
    
    try:
        logger.info("Press 'q' to exit")
        frame_count = 0
        
        while True:
            # Get latest frame
            frame_data = camera.get_latest_frame(timeout=0.1)
            
            if frame_data:
                frame_count += 1
                
                # Display frame
                cv2.imshow("Camera Feed", frame_data.frame)
                
                # Print statistics every 60 frames
                if frame_count % 60 == 0:
                    stats = camera.get_statistics()
                    logger.info(f"FPS: {stats['fps']:.1f}, Dropped: {stats['dropped_frames']}, "
                               f"Queue: {stats['queue_size']}/{stats['max_queue_size']}")
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        camera.close()
        cv2.destroyAllWindows()