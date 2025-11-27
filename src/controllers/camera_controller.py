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
    
    def __init__(self, camera_id: int = 0, max_queue_size: int = 1):
        """
        Initialize the threaded camera controller.
        
        Args:
            camera_id: ID of the camera to connect to
            max_queue_size: Maximum number of frames to buffer in queue (default 1 for absolute minimum latency)
        """
        self.camera_id = camera_id
        self.max_queue_size = max_queue_size
        
        # Camera hardware interface
        self.h_cam = None
        self.mem_ptr = None
        self.mem_id = None
        self.width = 0
        self.height = 0
        
        # ALWAYS BLACK & WHITE (MONO8) - 3x faster than color
        self.bits_per_pixel = 8  # Grayscale: 1 byte per pixel
        self.color_mode = ueye.IS_CM_MONO8
        self.channels = 1
        
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
        
        # FPS control
        self.target_fps = None  # Target FPS for capture limiting (None = no limit)
        
        # Error tracking
        self.last_error = None
        self.consecutive_errors = 0
        
        # Frame pool for memory optimization (initialized later when we know frame size)
        self.frame_pool: Optional[FramePool] = None

        # Optional callback to notify when a new frame is available (called from capture thread)
        self._frame_callback = None
        
        # Thread pool for parallel operations (4 workers for 20-core CPU)
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="camera_worker")
        
    
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
            pass  # Sensor info error
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
            pass  # Pixel clock error
        
        # Frame rate
        try:
            fps_ptr = ueye.DOUBLE()
            ret = ueye.is_SetFrameRate(self.h_cam, ueye.IS_GET_FRAMERATE, fps_ptr)
            if ret == ueye.IS_SUCCESS:
                settings['frame_rate_fps'] = fps_ptr.value
        except Exception as e:
            pass  # Frame rate error
        
        # Exposure time
        try:
            exposure = ueye.DOUBLE()
            ret = ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exposure, 8)
            if ret == ueye.IS_SUCCESS:
                settings['exposure_ms'] = exposure.value
        except Exception as e:
            pass  # Exposure error
        
        return settings
    
    def _get_image_processing_settings(self) -> Dict[str, Any]:
        """Get image processing related settings."""
        settings = {}
        
        # Gain settings
        try:
            gain_master = ueye.c_int()
            ret = ueye.is_SetHardwareGain(self.h_cam, ueye.IS_GET_MASTER_GAIN, gain_master, 
                                        ueye.c_int(), ueye.c_int())
            # For IS_GET_MASTER_GAIN, the return value IS the gain value, not a success code
            if ret >= 0:  # Valid gain values are >= 0
                settings['gain_master'] = ret  # Use the return value directly
            else:
                settings['gain_master'] = 'unavailable'
        except Exception as e:
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
            pass  # AOI error
        
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
            pass  # Camera info error
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
            return (480, 640, 1)  # Test pattern is also MONO8
        return (int(self.height), int(self.width), self.channels)  # Use actual channel count (1 for MONO8)
    
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
            
            # Set MONO8 mode (black & white, better for analysis)
            ret = ueye.is_SetColorMode(self.h_cam, ueye.IS_CM_MONO8)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"SetColorMode MONO8 failed: {ret}")
                self._cleanup_hardware()
                return False
            
            logger.info("Camera mode: MONO8 (black & white, better compression & file sizes)")
            # Note: Actual FPS will be logged after frame rate configuration below
            
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
                # 1. Set MINIMUM exposure for maximum FPS (3ms allows ~333 FPS theoretical)
                min_exposure = ueye.DOUBLE(3.0)  # 3ms minimum exposure
                ret = ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, min_exposure, 8)
                if ret == ueye.IS_SUCCESS:
                    logger.info(f"Exposure set to minimum: {min_exposure.value:.2f} ms for maximum FPS")
                else:
                    logger.warning(f"Failed to set minimum exposure: {ret}")
                
                # 2. Set pixel clock to maximum for best performance
                pixel_clock_range = (ueye.c_uint * 3)()
                ret = ueye.is_PixelClock(self.h_cam, ueye.IS_PIXELCLOCK_CMD_GET_RANGE, pixel_clock_range, 12)
                if ret == ueye.IS_SUCCESS:
                    max_pixel_clock = pixel_clock_range[1]  # Maximum value
                    ret = ueye.is_PixelClock(self.h_cam, ueye.IS_PIXELCLOCK_CMD_SET, max_pixel_clock, 4)
                    if ret == ueye.IS_SUCCESS:
                        logger.info(f"Pixel clock set to maximum: {max_pixel_clock.value} MHz")
                
                # 3. Get current achievable frame rate after exposure/pixel clock changes
                fps_ptr = ueye.DOUBLE()
                ret = ueye.is_SetFrameRate(self.h_cam, ueye.IS_GET_FRAMERATE, fps_ptr)
                if ret == ueye.IS_SUCCESS:
                    logger.info(f"Camera achievable FPS (after optimization): {fps_ptr.value:.1f} FPS")
                
                # 4. Request 60 FPS target (or maximum if less)
                target_fps = ueye.DOUBLE(60.0)  # Target 60 FPS for smooth recording
                actual_fps = ueye.DOUBLE()
                ret = ueye.is_SetFrameRate(self.h_cam, target_fps, actual_fps)
                if ret == ueye.IS_SUCCESS:
                    logger.info(f"Camera configured for: {actual_fps.value:.1f} FPS (requested {target_fps.value:.0f})")
                else:
                    logger.warning(f"Failed to set frame rate: {ret}")
            except Exception as e:
                logger.warning(f"Frame rate optimization error: {e}")
            
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
            
            # CRITICAL: Enable CONTINUOUS CAPTURE MODE for high FPS (was using slow FreezeVideo)
            # CaptureVideo with IS_DONT_WAIT enables free-running capture at full camera speed
            ret = ueye.is_CaptureVideo(self.h_cam, ueye.IS_DONT_WAIT)
            if ret == ueye.IS_SUCCESS:
                logger.info("Enabled continuous capture mode (CaptureVideo) for maximum FPS - camera now free-running!")
            else:
                logger.warning(f"Failed to enable continuous capture: {ret}, will use slower single-frame mode")
            
            # Initialize frame pool with actual camera dimensions (grayscale)
            if self.frame_pool is None:
                frame_shape = (self.height, self.width, 1)  # Always grayscale
                self.frame_pool = FramePool(frame_shape, pool_size=10)  # Larger pool for 32GB RAM
            
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
            
            return True

    def register_frame_callback(self, callback):
        """Register a callable to be invoked whenever a new frame is queued.

        The callback is invoked from the capture thread; callers should ensure
        thread-safety (e.g. schedule UI updates via QTimer.singleShot).
        """
        self._frame_callback = callback
    
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
        
    
    def get_latest_frame(self, timeout: float = 0.1) -> Optional[FrameData]:
        """
        Get the most recent frame from the queue with optimized O(1) draining.
        
        Args:
            timeout: Maximum time to wait for a frame (seconds)
            
        Returns:
            FrameData object or None if no frame available
        """
        try:
            # Get first frame (with timeout)
            latest_frame = self.frame_queue.get(timeout=timeout)
            frames_discarded = 0
            
            # Fast drain: get all remaining frames without timeout
            while True:
                try:
                    old_frame = latest_frame
                    latest_frame = self.frame_queue.get_nowait()
                    frames_discarded += 1
                    
                    # Return old frame to pool immediately
                    if self.frame_pool and hasattr(old_frame, 'frame'):
                        self.frame_pool.return_frame(old_frame.frame)
                except queue.Empty:
                    break
            
            # Batch update statistics (more efficient than per-frame update)
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
        
        last_capture_time = 0
        target_frame_interval = 1.0 / 30.0  # 30 FPS for test pattern mode
        
        while self.running:
            try:
                current_time = time.time()
                
                # Frame rate limiting for test pattern mode only
                # Hardware mode: camera's is_FreezeVideo already blocks at configured FPS
                if self.use_test_pattern:
                    time_since_last_capture = current_time - last_capture_time
                    if time_since_last_capture < target_frame_interval:
                        # Sleep for remaining time to maintain 30 FPS
                        sleep_time = target_frame_interval - time_since_last_capture
                        time.sleep(sleep_time)
                        current_time = time.time()  # Update time after sleep
                
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
                        # ALWAYS KEEP NEWEST FRAME: If queue is full, remove old frame first
                        if self.frame_queue.full():
                            try:
                                old_frame = self.frame_queue.get_nowait()
                                # Return old frame to pool
                                if self.frame_pool and hasattr(old_frame, 'frame'):
                                    self.frame_pool.return_frame(old_frame.frame)
                            except queue.Empty:
                                pass
                        
                        # Now add the fresh frame
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
                        # Notify UI/clients that a new frame is available (non-blocking)
                        try:
                            if self._frame_callback:
                                try:
                                    self._frame_callback()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except queue.Full:
                        # Queue is full, drop this frame (true capture drop)
                        with self.stats_lock:
                            self.dropped_frames += 1
                
                else:
                    # Frame capture failed - don't count as captured or dropped
                    with self.stats_lock:
                        self.consecutive_errors += 1
                
                # FreezeVideo naturally rate-limits to camera FPS, no sleep needed
                last_capture_time = current_time
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                with self.stats_lock:
                    self.consecutive_errors += 1
        
    
    def _capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera or generate test pattern."""
        if self.use_test_pattern:
            return self._generate_test_pattern()
        else:
            return self._capture_hardware_frame()
    
    def _capture_hardware_frame(self) -> Optional[np.ndarray]:
        """Capture fresh frame using FreezeVideo - minimal lag, captures on-demand."""
        if not PYUEYE_AVAILABLE or not self.h_cam:
            return None
        
        try:
            # In CaptureVideo mode, just copy the current buffer (camera continuously updates it)
            # This is MUCH faster than FreezeVideo which waits for each frame
            # IMPORTANT: Add tiny delay to ensure buffer is updated (camera needs ~17ms at 57 FPS)
            import time
            time.sleep(0.001)  # 1ms safety margin for buffer update
            
            height = int(self.height)
            width = int(self.width)
            
            array = ueye.get_data(
                self.mem_ptr, width, height,
                8, width, copy=True  # 8 bits per pixel, MUST copy since buffer updates continuously
            )
            
            frame = np.frombuffer(array, dtype=np.uint8)
            # Grayscale: (height, width) -> (height, width, 1) for consistency
            frame_reshaped = frame.reshape((height, width, 1))
            
            # Validate frame is not all black (camera issue detection)
            if frame_reshaped.max() == 0:
                logger.warning("Camera returned all-black frame - check lens cap or exposure settings")
            
            return frame_reshaped
            
        except Exception as e:
            return None
    
    def _generate_test_pattern(self) -> np.ndarray:
        """Generate grayscale test pattern frame."""
        import cv2
        
        width, height = 640, 480
        
        # Create grayscale frame
        if self.frame_pool:
            frame = self.frame_pool.get_frame()
            if frame is None:
                frame = np.zeros((height, width, 1), dtype=np.uint8)
            else:
                frame.fill(0)
        else:
            frame = np.zeros((height, width, 1), dtype=np.uint8)
        
        # Grid pattern (grayscale)
        for i in range(0, height, 50):
            cv2.line(frame, (0, i), (width, i), 50, 1)
        for i in range(0, width, 50):
            cv2.line(frame, (i, 0), (i, height), 50, 1)
        
        # Animated elements
        t = self.test_frame_counter / 60.0
        
        # Moving crosshair
        center_x = int(width // 2 + 100 * np.sin(t))
        center_y = int(height // 2 + 50 * np.cos(t * 1.3))
        cv2.line(frame, (center_x, 0), (center_x, height), 200, 2)
        cv2.line(frame, (0, center_y), (width, center_y), 200, 2)
        
        # Pulsing circle
        radius = int(30 + 20 * np.sin(t * 3))
        cv2.circle(frame, (width // 2, height // 2), radius, 150, 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.test_frame_counter}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        
        # FPS indicator
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 180, 2)
        
        self.test_frame_counter += 1
        return frame
    
    def _cleanup_hardware(self) -> None:
        """Clean up camera hardware resources."""
        if not PYUEYE_AVAILABLE:
            return
        
        try:
            if self.h_cam:
                ueye.is_StopLiveVideo(self.h_cam, ueye.IS_FORCE_VIDEO_STOP)
                
                # Free buffers
                if self.mem_ptr and self.mem_id:
                    ueye.is_FreeImageMem(self.h_cam, self.mem_ptr, self.mem_id)
                
                ueye.is_ExitCamera(self.h_cam)
        except Exception as e:
            pass  # Hardware cleanup error
        
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
        """Set camera exposure time in milliseconds.
        
        Args:
            exposure_ms: Exposure time in milliseconds (must be > 0)
            
        Returns:
            True if successful, False otherwise
        """
        if exposure_ms <= 0:
            logger.error(f"Invalid exposure time: {exposure_ms} ms (must be > 0)")
            return False
            
        exposure = ueye.DOUBLE(exposure_ms)
        return self._set_camera_parameter(
            "exposure", exposure_ms,
            lambda: ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure, 8),
            f"Exposure set to {exposure_ms:.2f} ms",
            "Failed to set exposure"
        )
    
    def set_gain(self, gain: int) -> bool:
        """Set camera master gain (0-100).
        
        Args:
            gain: Master gain value (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        if not (0 <= gain <= 100):
            logger.error(f"Invalid gain: {gain} (must be 0-100)")
            return False
            
        return self._set_camera_parameter(
            "gain", gain,
            lambda: ueye.is_SetHardwareGain(self.h_cam, gain, ueye.IS_IGNORE_PARAMETER, 
                                          ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER),
            f"Gain set to {gain}",
            "Failed to set gain"
        )
    
    def set_framerate(self, fps: float) -> bool:
        """Set camera frame rate in frames per second.
        
        Args:
            fps: Frame rate in FPS (must be > 0)
            
        Returns:
            True if successful, False otherwise
        """
        if fps <= 0:
            logger.error(f"Invalid frame rate: {fps} fps (must be > 0)")
            return False
        
        if not self.h_cam or not PYUEYE_AVAILABLE:
            logger.warning(f"Cannot set framerate: camera not connected")
            return False
        
        try:
            new_fps = ueye.DOUBLE(fps)
            actual_fps = ueye.DOUBLE()
            ret = ueye.is_SetFrameRate(self.h_cam, new_fps, actual_fps)
            
            if ret == ueye.IS_SUCCESS:
                self.target_fps = fps  # Store requested target
                logger.info(f"Frame rate set: requested {fps:.1f} fps, actual {actual_fps.value:.1f} fps")
                return True
            else:
                logger.error(f"Failed to set frame rate: error code {ret}")
                return False
        except Exception as e:
            logger.error(f"Error setting framerate: {e}")
            return False
    
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
            pass  # Frames returned to pool
        
        # Shutdown thread pool
        if hasattr(self, '_thread_pool'):
            try:
                # Block until worker threads finish to ensure clean shutdown
                self._thread_pool.shutdown(wait=True)
            except Exception:
                try:
                    self._thread_pool.shutdown(wait=False)
                except Exception:
                    pass
        
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