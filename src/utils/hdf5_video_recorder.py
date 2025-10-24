"""HDF5 Video Recorder for AFS Acquisition.

Provides high-performance video recording with frame-level access, compression,
and metadata storage. Optimized for real-time recording with LZF compression.
"""

import h5py
import numpy as np
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import gc

from src.utils.logger import get_logger
from src.utils.validation import validate_positive_number, validate_frame_shape, validate_file_path

logger = get_logger("hdf5_recorder")


class HDF5VideoRecorder:
    """
    HDF5-based video recorder that stores frames as a 4D dataset with compression and metadata.
    
    Features:
    - 4D dataset structure: (n_frames, height, width, channels)
    - LZF compression for fast random access
    - Chunking optimized for frame-level access
    - Comprehensive metadata storage
    - Frame-by-frame recording with dynamic dataset growth
    """
    
    def __init__(self, file_path: Union[str, Path], frame_shape: Tuple[int, int, int], 
                 fps: float = 60.0, min_fps: float = 20.0, compression_level: int = 4, downscale_factor: int = 1) -> None:
        """
        Initialize the HDF5 video recorder with configurable compression.
        
        Args:
            file_path: Path to save the HDF5 file (str or Path object)
            frame_shape: Shape of each frame (height, width, channels)
            fps: Target frames per second for metadata (must be > 0)
            min_fps: CRITICAL minimum FPS - recording will log errors if below this (must be > 0)
            compression_level: Compression level (0=none, 1-3=fast/LZF, 4-9=best/GZIP)
            downscale_factor: Factor to downscale frames (1=no downscale, 2=half size, 4=quarter size)
            
        Raises:
            ValueError: If frame_shape is invalid or fps <= 0
            OSError: If the parent directory doesn't exist or isn't writable
        """
        # Input validation using validation utilities
        self.original_frame_shape = validate_frame_shape(frame_shape, "frame_shape")
        self.fps = validate_positive_number(fps, "fps")
        self.min_fps = validate_positive_number(min_fps, "min_fps")
        
        # CRITICAL: Enforce minimum FPS requirement
        if self.min_fps > self.fps:
            logger.warning(f"min_fps ({self.min_fps}) > fps ({self.fps}), adjusting fps to match min_fps")
            self.fps = self.min_fps
        
        file_path_obj = validate_file_path(file_path, must_exist=False, create_parent=True, field_name="file_path")
        
        # Compression settings
        self.compression_level = max(0, min(9, compression_level))  # Clamp 0-9
        self.downscale_factor = max(1, min(4, downscale_factor))  # Clamp 1-4
        
        # Calculate actual frame shape after downscaling
        if self.downscale_factor > 1:
            h, w, c = self.original_frame_shape
            self.frame_shape = (h // self.downscale_factor, w // self.downscale_factor, c)
        else:
            self.frame_shape = self.original_frame_shape
        
        self.file_path = str(file_path_obj)
        
        # Recording state with proper type annotations
        self.h5_file: Optional[h5py.File] = None
        self.video_dataset: Optional[h5py.Dataset] = None
        self.is_recording: bool = False
        self.frame_count: int = 0
        self.start_time: Optional[float] = None
        
        # Dataset parameters - optimized for performance (use downscaled frame shape for chunks)
        self.chunk_size = self._calculate_optimal_chunk_size(self.frame_shape)
        self.initial_size = 2000  # Larger initial allocation to reduce resizing
        self.growth_factor = 2.0  # Exponential growth for better amortized performance
        
        # Performance tracking
        self.write_errors = 0
        self.max_write_errors = 10  # Stop recording after too many errors
        self.last_flush_time = 0
        self.flush_interval = 5.0  # Flush every 5 seconds for data safety
        
        # Recording state
        self._closed = False
        
        # Metadata storage
        self.camera_settings_saved = False
        self.stage_settings_saved = False
        
        # Function generator timeline logging
        self.fg_timeline_buffer = []
        self.fg_timeline_buffer_size = 1000  # Buffer entries before writing to disk
        
        # Execution data logging (for force path execution, etc.)
        self.execution_data_group = None
        
        # High-performance asynchronous writing - MAXIMUM BUFFER for constant FPS
        self._write_queue = queue.Queue(maxsize=1000)  # Very large buffer for constant FPS (was 500)
        self._write_thread = None
        self._stop_writing = threading.Event()
        self._write_lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Optimized frame batching for constant performance
        self._frame_batch = []
        self._batch_indices = []
        self._batch_size = 50  # Larger batches for consistent performance (was 25)
        self._batch_lock = threading.Lock()
        
        # Performance counters
        self._frames_queued = 0
        self._frames_written = 0
        self._batch_writes = 0
        
        # FPS enforcement and tracking (CRITICAL)
        self._frame_timestamps = []
        self._last_fps_check = 0
        self._fps_check_interval = 2.0  # Check FPS every 2 seconds
        self._fps_violations = 0
        self._max_fps_violations = 5  # Stop recording after 5 consecutive violations
        
    def _calculate_optimal_chunk_size(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, ...]:
        """
        Calculate optimal chunk size for the dataset.
        
        Optimized chunking for compression and I/O performance:
        - Balance between compression efficiency and random access
        - Target chunk size around 1-4 MB for optimal HDF5 performance
        
        Args:
            frame_shape: Shape of each frame (height, width, channels)
            
        Returns:
            Optimal chunk size tuple
        """
        frame_bytes = np.prod(frame_shape)
        target_chunk_bytes = 2 * 1024 * 1024  # 2MB target chunk size
        
        # Calculate frames per chunk for target size
        frames_per_chunk = max(1, min(16, target_chunk_bytes // frame_bytes))
        
        return (frames_per_chunk, *frame_shape)
    
    def start_recording(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start recording to HDF5 file.
        
        Args:
            metadata: Optional dictionary of metadata to store
            
        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False
        
        try:
            # Pre-recording checks and setup
            if not self._prepare_recording():
                return False
            
            # Create and configure HDF5 file
            if not self._create_hdf5_file():
                return False
            
            # Set up datasets and metadata
            if not self._setup_datasets_and_metadata(metadata):
                return False
            
            # Initialize recording state and start async processing
            self._initialize_recording_state()
            
            logger.info(f"Recording started: {self.file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HDF5 recording: {e}")
            self._cleanup_failed_start()
            return False
    
    def _prepare_recording(self) -> bool:
        """Prepare for recording - check disk space and create directories."""
        # Ensure directory exists
        dir_path = os.path.dirname(self.file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Check available disk space before starting
        if not self._check_disk_space():
            return self._handle_insufficient_disk_space()
        
        return True
    
    def _handle_insufficient_disk_space(self) -> bool:
        """Handle insufficient disk space by attempting cleanup."""
        try:
            logger.warning("Attempting emergency cleanup to free disk space...")
        except OSError:
            pass
        
        if self._emergency_cleanup():
            # Recheck after cleanup
            if self._check_disk_space():
                try:
                    logger.info("Emergency cleanup successful - proceeding with recording")
                except OSError:
                    pass
                return True
            else:
                try:
                    logger.error("Insufficient disk space even after cleanup")
                except OSError:
                    pass
                return False
        else:
            try:
                logger.error("Insufficient disk space and cleanup failed")
            except OSError:
                pass
            return False
    
    def _create_hdf5_file(self) -> bool:
        """Create and open HDF5 file with optimized settings."""
        try:
            self.h5_file = h5py.File(
                self.file_path, 
                'w',
                libver='latest'  # Use latest HDF5 format for best performance
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create HDF5 file: {e}")
            return False
    
    def _setup_datasets_and_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set up video dataset and metadata."""
        try:
            # Create video dataset
            self._create_video_dataset()
            
            # Add metadata
            self._add_dataset_metadata()
            
            if metadata:
                self._add_user_metadata_to_dataset(metadata)
            
            # Create function generator timeline dataset
            self._create_fg_timeline_dataset()
            
            # Create comprehensive groups structure
            self._create_metadata_group()
            
            # Create execution data group for force path execution, etc.
            self._create_execution_data_group()
            
            return True
        except Exception as e:
            logger.error(f"Failed to setup datasets: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _create_video_dataset(self):
        """Create the main video dataset with configurable compression under /raw_data/main_video."""
        # Create /raw_data group if it doesn't exist (RENAMED from 'data' to 'raw_data')
        if 'raw_data' not in self.h5_file:
            data_group = self.h5_file.create_group('raw_data')
            data_group.attrs['description'] = b'All raw measurement data including video and timelines'
        else:
            data_group = self.h5_file['raw_data']
        
        # Shape: (n_frames, height, width, channels)
        initial_shape = (self.initial_size, *self.frame_shape)
        max_shape = (None, *self.frame_shape)  # Unlimited frames
        
        # Base dataset parameters
        dataset_kwargs = {
            'shape': initial_shape,
            'maxshape': max_shape,
            'dtype': np.uint8,
            'chunks': self.chunk_size,
            'shuffle': True,  # Enable shuffle filter for better compression
            'fillvalue': 0,
            'track_times': False,
        }
        
        # Configure compression based on level (default level 9 = GZIP level 6)
        if self.compression_level == 0:
            # No compression - fastest, largest files
            dataset_kwargs['compression'] = None
        elif self.compression_level <= 3:
            # LZF compression - fast, good for real-time
            dataset_kwargs['compression'] = 'lzf'
            dataset_kwargs['fletcher32'] = True
        else:
            # GZIP compression - slower but best compression (for offline analysis)
            gzip_level = min(self.compression_level - 3, 9)  # Map 4-9 to 1-6 (capped at 9 for extreme compression)
            dataset_kwargs['compression'] = 'gzip'
            dataset_kwargs['compression_opts'] = gzip_level
            dataset_kwargs['fletcher32'] = True
        
        self.video_dataset = data_group.create_dataset('main_video', **dataset_kwargs)
        self.video_dataset.attrs['description'] = b'Main camera video with efficient compression'
    
    def _initialize_recording_state(self):
        """Initialize recording state and start async processing."""
        self.is_recording = True
        self.frame_count = 0
        self.start_time = datetime.now()
        
        # Start async write thread for better performance
        self._start_async_writer()
    
    def _cleanup_failed_start(self):
        """Clean up resources after a failed start attempt."""
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None
    
    def _check_disk_space(self, min_gb: float = 0.05) -> bool:
        """
        Check disk space with emergency protection against complete disk fill.
        
        Args:
            min_gb: Minimum required space in GB (default 50MB for emergency recording)
            
        Returns:
            True if sufficient space available
        """
        try:
            import shutil
            total, used, free = shutil.disk_usage(os.path.dirname(self.file_path) or '.')
            free_gb = free / (1024**3)
            free_mb = free / (1024**2)
            
            # Emergency protection - never allow less than 20MB free space
            if free_mb < 20:
                try:
                    logger.critical(f"EMERGENCY: Only {free_mb:.1f}MB free space - preventing disk full")
                except OSError:
                    pass  # Can't even log
                return False
            
            # Critical warning - less than 50MB
            if free_mb < 50:
                try:
                    logger.error(f"CRITICAL: Only {free_mb:.1f}MB free space - recording will stop soon")
                except OSError:
                    pass
                # Still allow recording but warn it will fail soon
                return free_gb >= min_gb
            
            # Adaptive minimum based on available space
            adaptive_min = min_gb if free_gb > 0.5 else 0.05
            
            if free_gb < adaptive_min:
                try:
                    logger.warning(f"Low disk space: {free_gb:.2f}GB available, {adaptive_min:.2f}GB required")
                except OSError:
                    pass
                return False
            
            if free_gb < 0.2:
                try:
                    logger.warning(f"Very low disk space: {free_gb:.2f}GB available - recording may fail")
                except OSError:
                    pass
                
            return True
            
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Allow recording if check fails
    
    def _add_dataset_metadata(self):
        """Add technical metadata to the video dataset."""
        if not self.video_dataset:
            return
            
        # Recording parameters
        self.video_dataset.attrs['fps'] = self.fps
        self.video_dataset.attrs['min_fps'] = self.min_fps
        self.video_dataset.attrs['frame_shape'] = self.frame_shape
        self.video_dataset.attrs['original_frame_shape'] = self.original_frame_shape
        self.video_dataset.attrs['downscale_factor'] = self.downscale_factor
        self.video_dataset.attrs['compression_level'] = self.compression_level
        
        # Compression details
        if self.compression_level == 0:
            self.video_dataset.attrs['compression'] = 'none'
        elif self.compression_level <= 3:
            self.video_dataset.attrs['compression'] = 'lzf'
        else:
            self.video_dataset.attrs['compression'] = 'gzip'
            self.video_dataset.attrs['gzip_level'] = min(self.compression_level - 3, 9)  # Match _create_video_dataset
        
        self.video_dataset.attrs['chunk_size'] = self.chunk_size
        
        # Timestamp information
        self.video_dataset.attrs['created_at'] = datetime.now().isoformat()
        
        # Format information
        self.video_dataset.attrs['format_version'] = '1.0'
        self.video_dataset.attrs['color_format'] = 'BGR'  # OpenCV default
        self.video_dataset.attrs['data_type'] = 'uint8'
        
        # AFS-specific metadata
        self.video_dataset.attrs['system'] = 'AFS_tracking'
        
    def _add_user_metadata_to_dataset(self, metadata: Dict[str, Any]):
        """Add user-provided metadata directly to the video dataset attributes."""
        if not self.video_dataset:
            return
            
        for key, value in metadata.items():
            if value is not None and value != "":
                try:
                    # Handle different data types appropriately
                    attr_key = f"user_{key}"  # Prefix to distinguish user metadata
                    if isinstance(value, str):
                        self.video_dataset.attrs[attr_key] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        self.video_dataset.attrs[attr_key] = value
                    elif isinstance(value, datetime):
                        self.video_dataset.attrs[attr_key] = value.isoformat()
                    else:
                        # Convert to string as fallback
                        self.video_dataset.attrs[attr_key] = str(value).encode('utf-8')
                        
                except Exception as e:
                    logger.warning(f"Could not save metadata '{key}': {e}")
    

    
    def record_frame(self, frame: np.ndarray, use_async: bool = True) -> bool:
        """
        Record a single frame with optional downscaling and compression.
        CRITICAL: Enforces minimum FPS requirements with harsh monitoring.
        
        Args:
            frame: Frame data as numpy array (height, width, channels)
            use_async: Use asynchronous writing for better performance
            
        Returns:
            True if frame queued/recorded successfully
        """
        if not self.is_recording or not self.video_dataset or self._closed:
            return False
        
        # CRITICAL FPS ENFORCEMENT: Track frame timing
        current_time = time.time()
        self._frame_timestamps.append(current_time)
        
        # Periodic FPS check (every 2 seconds)
        if current_time - self._last_fps_check >= self._fps_check_interval:
            self._check_fps_performance(current_time)
            self._last_fps_check = current_time
        
        # Check for too many write errors
        if self.write_errors >= self.max_write_errors:
            logger.error(f"Too many write errors ({self.write_errors}), stopping recording")
            self.stop_recording()
            return False
        
        # Apply downscaling if enabled
        if self.downscale_factor > 1:
            frame = self._downscale_frame(frame)
        
        # Fast async path for high performance
        if use_async:
            return self._record_frame_async(frame)
        
        # Synchronous fallback for compatibility
        return self._record_frame_sync(frame)
    
    def _downscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Downscale frame using fast area interpolation."""
        try:
            import cv2
            new_height = self.frame_shape[0]
            new_width = self.frame_shape[1]
            
            # Handle grayscale frames (H, W, 1) - squeeze before resize, then reshape
            is_grayscale = len(frame.shape) == 3 and frame.shape[2] == 1
            if is_grayscale:
                frame_2d = frame.squeeze()  # (H, W, 1) -> (H, W)
                downscaled_2d = cv2.resize(frame_2d, (new_width, new_height), interpolation=cv2.INTER_AREA)
                downscaled = downscaled_2d.reshape((new_height, new_width, 1))  # -> (H, W, 1)
            else:
                # Color frames work normally
                downscaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return downscaled
        except Exception as e:
            logger.error(f"Error downscaling frame: {e}")
            return frame  # Return original on error
    
    def _check_fps_performance(self, current_time: float):
        """
        CRITICAL: Check if recording is maintaining minimum FPS requirements.
        This is harsh - will log errors and potentially stop recording if FPS is too low.
        """
        if len(self._frame_timestamps) < 10:
            return  # Need at least 10 frames to calculate FPS
        
        # Calculate FPS over last 2 seconds
        recent_timestamps = [t for t in self._frame_timestamps if current_time - t <= self._fps_check_interval]
        
        if len(recent_timestamps) < 2:
            return
        
        time_span = recent_timestamps[-1] - recent_timestamps[0]
        if time_span <= 0:
            return
        
        actual_fps = (len(recent_timestamps) - 1) / time_span
        
        # HARSH FPS ENFORCEMENT
        if actual_fps < self.min_fps:
            self._fps_violations += 1
            logger.error(
                f"CRITICAL FPS VIOLATION #{self._fps_violations}: "
                f"Recording at {actual_fps:.1f} FPS (minimum required: {self.min_fps:.1f} FPS)"
            )
            
            if self._fps_violations >= self._max_fps_violations:
                logger.error(
                    f"STOPPING RECORDING: Too many FPS violations "
                    f"({self._fps_violations} consecutive violations). "
                    f"System cannot maintain minimum {self.min_fps} FPS."
                )
                self._closed = True
                # Don't call stop_recording here to avoid recursion
        else:
            # Reset violation counter if FPS is good
            if self._fps_violations > 0:
                logger.info(f"FPS recovered: {actual_fps:.1f} FPS (violations reset)")
            self._fps_violations = 0
        
        # Trim old timestamps to prevent memory growth
        cutoff_time = current_time - 10.0  # Keep last 10 seconds
        self._frame_timestamps = [t for t in self._frame_timestamps if t > cutoff_time]
    
    def _record_frame_async(self, frame: np.ndarray) -> bool:
        """Memory-efficient asynchronous frame recording."""
        if self._closed:
            return False
            
        try:
            # Quick validation with early exit
            if frame.shape != self.frame_shape:
                logger.warning(f"Frame shape mismatch: expected {self.frame_shape}, got {frame.shape}")
                self.write_errors += 1
                return False
            
            # Ensure uint8 type efficiently
            if frame.dtype != np.uint8:
                # In-place conversion when possible to save memory
                if frame.flags.writeable:
                    frame = frame.astype(np.uint8, copy=False)
                else:
                    frame = frame.astype(np.uint8)
            
            # Emergency disk space check during recording
            try:
                import shutil
                free_space_mb = shutil.disk_usage(os.path.dirname(self.file_path)).free / (1024**2)
                if free_space_mb < 50:  # Less than 50MB - emergency stop
                    logger.error(f"EMERGENCY: Only {free_space_mb:.1f}MB left - stopping recording to prevent system failure")
                    self._closed = True
                    return False
            except Exception:
                pass  # Don't fail if disk check fails
            
            # Check queue capacity before copying frame (memory optimization)
            if self._write_queue.qsize() >= self._write_queue.maxsize * 0.9:
                # Use silent return instead of logging to prevent disk space issues
                return False
            
            # Queue frame for async writing with minimal memory copy
            try:
                frame_index = self.frame_count
                # Use view when possible to reduce memory usage
                frame_copy = np.copy(frame) if not frame.flags.owndata else frame
                
                self._write_queue.put((frame_copy, frame_index), timeout=0.001)
                self.frame_count += 1
                self._frames_queued += 1
                
                # Periodic memory management
                if self._frames_queued % 100 == 0:
                    gc.collect()
                
                return True
                
            except queue.Full:
                logger.warning("Write queue full, frame dropped")
                return False
                
        except Exception as e:
            logger.error(f"Error in async frame recording: {e}")
            self.write_errors += 1
            return False
    
    def _record_frame_sync(self, frame: np.ndarray) -> bool:
        """Synchronous frame recording with full validation."""
        try:
            # Full validation
            if frame.shape != self.frame_shape:
                logger.warning(f"Frame shape mismatch: expected {self.frame_shape}, got {frame.shape}")
                self.write_errors += 1
                return False
            
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Thread-safe writing
            with self._write_lock:
                # Check if we need to resize the dataset
                if self.frame_count >= self.video_dataset.shape[0]:
                    self._grow_dataset()
                
                # Write frame to dataset
                self.video_dataset[self.frame_count] = frame
                self.frame_count += 1
                self._frames_written += 1
            
            # Reset error count on successful write
            self.write_errors = 0
            
            # Smart flushing
            current_time = time.time()
            if current_time - self.last_flush_time >= self.flush_interval:
                try:
                    self.h5_file.flush()
                    self.last_flush_time = current_time
                except Exception as flush_error:
                    logger.warning(f"Flush failed: {flush_error}")
            
            return True
            
        except OSError as e:
            # Handle disk space and I/O errors specifically
            if "No space left" in str(e) or "errno 28" in str(e):
                logger.error(f"Disk full! Stopping recording: {e}")
                self.stop_recording()
                return False
            else:
                logger.error(f"I/O error recording frame {self.frame_count}: {e}")
                self.write_errors += 1
                return False
                
        except Exception as e:
            logger.error(f"Error recording frame {self.frame_count}: {e}")
            self.write_errors += 1
            return False
    
    def add_camera_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Add comprehensive camera settings to /meta_data/hardware_settings/camera_settings.
        
        Args:
            settings: Dictionary of camera settings
            
        Returns:
            True if settings saved successfully
        """
        if not self.is_recording or not hasattr(self, 'metadata_group') or not self.metadata_group or self.camera_settings_saved:
            return False
            
        try:
            # Get hardware_settings group
            if 'hardware_settings' not in self.metadata_group:
                hardware_group = self.metadata_group.create_group('hardware_settings')
            else:
                hardware_group = self.metadata_group['hardware_settings']
            
            # Create camera settings subgroup
            camera_group = hardware_group.create_group('camera_settings')
            camera_group.attrs['description'] = b'All camera configuration parameters'
            camera_group.attrs['saved_at'] = datetime.now().isoformat().encode('utf-8')
            
            settings_count = 0
            # Add camera settings as group attributes
            for key, value in settings.items():
                if value is not None:
                    try:
                        # Handle different data types appropriately
                        if isinstance(value, str):
                            camera_group.attrs[key] = value.encode('utf-8')
                        elif isinstance(value, bytes):
                            camera_group.attrs[key] = value.decode('utf-8', errors='ignore').encode('utf-8')
                        elif isinstance(value, (int, float, bool)):
                            camera_group.attrs[key] = value
                        elif value is None:
                            camera_group.attrs[key] = 'None'.encode('utf-8')
                        else:
                            # Convert other types to string
                            camera_group.attrs[key] = str(value).encode('utf-8')
                        
                        settings_count += 1
                    except Exception as e:
                        # Log individual setting errors but continue
                        logger.warning(f"Failed to save camera setting '{key}': {e}")
                        # Save error info
                        error_attr_name = f"{key}_error"
                        camera_group.attrs[error_attr_name] = str(e).encode('utf-8')
            
            # Add summary metadata
            camera_group.attrs['total_parameters'] = settings_count
            
            self.camera_settings_saved = True
            return True
            
        except Exception as e:
            logger.error(f"Error saving camera settings: {e}")
            return False
    
    def add_stage_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Add XY stage settings to /meta_data/hardware_settings/stage_settings.
        
        Args:
            settings: Dictionary of stage settings
            
        Returns:
            True if settings saved successfully
        """
        if not self.is_recording or not hasattr(self, 'metadata_group') or not self.metadata_group or self.stage_settings_saved:
            return False
            
        try:
            # Get hardware_settings group
            if 'hardware_settings' not in self.metadata_group:
                hardware_group = self.metadata_group.create_group('hardware_settings')
            else:
                hardware_group = self.metadata_group['hardware_settings']
            
            # Create stage settings subgroup
            stage_group = hardware_group.create_group('stage_settings')
            stage_group.attrs['description'] = b'XY stage configuration and position'
            stage_group.attrs['saved_at'] = datetime.now().isoformat().encode('utf-8')
            
            # Add stage settings as group attributes
            for key, value in settings.items():
                if value is not None:
                    if isinstance(value, str):
                        stage_group.attrs[key] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        stage_group.attrs[key] = value
                    else:
                        stage_group.attrs[key] = str(value).encode('utf-8')
            
            stage_group.attrs['total_parameters'] = len(settings)
            
            self.stage_settings_saved = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to save XY stage settings: {e}")
            return False

    def add_recording_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Add user recording metadata to /meta_data/recording_info.
        
        Args:
            metadata: Dictionary of recording metadata
            
        Returns:
            True if metadata saved successfully
        """
        if not self.is_recording or not hasattr(self, 'metadata_group') or not self.metadata_group:
            return False
            
        try:
            # Get recording_info group (already created in _create_metadata_group)
            if 'recording_info' not in self.metadata_group:
                recording_group = self.metadata_group.create_group('recording_info')
                recording_group.attrs['description'] = b'Recording timestamps and session information'
                recording_group.attrs['created_at'] = datetime.now().isoformat().encode('utf-8')
            else:
                recording_group = self.metadata_group['recording_info']
            
            # Add user metadata
            for key, value in metadata.items():
                if value is not None:
                    if isinstance(value, str):
                        recording_group.attrs[key] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        recording_group.attrs[key] = value
                    else:
                        recording_group.attrs[key] = str(value).encode('utf-8')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save recording metadata: {e}")
            return False
    
    def add_force_path_execution(self, path_points: List, execution_start: str = None, execution_end: str = None) -> bool:
        """
        Add force path execution table to /meta_data/force_path_execution.
        Saves the exact force path table from the designer widget.
        
        Args:
            path_points: List of PathPoint objects from force path designer
                Each point has: time, frequency, amplitude, transition
            execution_start: ISO format timestamp of when execution started
            execution_end: ISO format timestamp of when execution ended
            
        Returns:
            True if data saved successfully
        """
        if not self.is_recording or not hasattr(self, 'metadata_group') or not self.metadata_group:
            logger.warning("Cannot add force path execution: not recording or no metadata group")
            return False
            
        try:
            # Create force_path_execution group if it doesn't exist
            if 'force_path_execution' not in self.metadata_group:
                fp_group = self.metadata_group.create_group('force_path_execution')
                fp_group.attrs['description'] = b'Force path execution table from designer'
                logger.info("Created /meta_data/force_path_execution group")
            else:
                fp_group = self.metadata_group['force_path_execution']
            
            # Add execution metadata
            fp_group.attrs['total_points'] = len(path_points)
            if execution_start:
                fp_group.attrs['execution_start'] = execution_start.encode('utf-8')
            if execution_end:
                fp_group.attrs['execution_end'] = execution_end.encode('utf-8')
            fp_group.attrs['created_at'] = datetime.now().isoformat().encode('utf-8')
            
            # Extract data from PathPoint objects
            times = []
            frequencies = []
            amplitudes = []
            transitions = []
            
            for point in path_points:
                times.append(point.time)
                frequencies.append(point.frequency)
                amplitudes.append(point.amplitude)
                transitions.append(point.transition.value if hasattr(point.transition, 'value') else str(point.transition))
            
            # Store as compressed datasets (more efficient than attributes for tables)
            if 'time' in fp_group:
                del fp_group['time']
            fp_group.create_dataset('time', data=np.array(times), compression='gzip', compression_opts=9)
            fp_group['time'].attrs['unit'] = b'seconds'
            fp_group['time'].attrs['description'] = b'Time for each point in the force path'
            
            if 'frequency' in fp_group:
                del fp_group['frequency']
            fp_group.create_dataset('frequency', data=np.array(frequencies), compression='gzip', compression_opts=9)
            fp_group['frequency'].attrs['unit'] = b'MHz'
            fp_group['frequency'].attrs['description'] = b'Frequency at each point'
            
            if 'amplitude' in fp_group:
                del fp_group['amplitude']
            fp_group.create_dataset('amplitude', data=np.array(amplitudes), compression='gzip', compression_opts=9)
            fp_group['amplitude'].attrs['unit'] = b'Vpp'
            fp_group['amplitude'].attrs['description'] = b'Amplitude at each point'
            
            if 'transition' in fp_group:
                del fp_group['transition']
            # Store transitions as strings
            transition_bytes = [t.encode('utf-8') for t in transitions]
            fp_group.create_dataset('transition', data=np.array(transition_bytes, dtype='S10'), compression='gzip', compression_opts=9)
            fp_group['transition'].attrs['description'] = b'Transition type (Hold/Linear) at each point'
            
            logger.info(f"Force path execution table saved: {len(path_points)} points")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save force path execution table: {e}", exc_info=True)
            return False
    
    def log_function_generator_event(self, frequency_mhz: float, amplitude_vpp: float, 
                                   output_enabled: bool = True, 
                                   event_type: str = 'parameter_change') -> bool:
        """Log a function generator timeline event.
        
        Args:
            frequency_mhz: Frequency in MHz (13-15 MHz range)
            amplitude_vpp: Amplitude in volts peak-to-peak
            output_enabled: Whether output is enabled
            event_type: Type of event ('parameter_change', 'output_on', 'output_off', etc.)
            
        Returns:
            True if event logged successfully
        """
        if not self.is_recording or not hasattr(self, 'fg_timeline_dataset') or not self.fg_timeline_dataset:
            return False
            
        try:
            current_time = time.time()
            # Calculate relative time from recording start
            start_timestamp = self.start_time.timestamp() if isinstance(self.start_time, datetime) else self.start_time
            relative_time = current_time - start_timestamp
            
            # Create timeline entry
            timeline_entry = np.array([
                (relative_time, frequency_mhz, amplitude_vpp, 
                 output_enabled, event_type.encode('utf-8')[:20])
            ], dtype=self.fg_timeline_dataset.dtype)
            
            # Add to buffer
            self.fg_timeline_buffer.extend(timeline_entry)
            
            # Flush buffer if it's getting full
            if len(self.fg_timeline_buffer) >= self.fg_timeline_buffer_size:
                self._flush_fg_timeline_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging FG timeline event: {e}")
            return False
    
    def _create_fg_timeline_dataset(self):
        """Create the function generator timeline dataset under /raw_data/function_generator_timeline."""
        if not self.h5_file:
            return
            
        try:
            # Get or create /raw_data group
            if 'raw_data' not in self.h5_file:
                data_group = self.h5_file.create_group('raw_data')
            else:
                data_group = self.h5_file['raw_data']
            
            # Define compound datatype for timeline entries
            timeline_dtype = np.dtype([
                ('timestamp', 'f8'),        # Relative time from recording start (seconds)
                ('frequency_mhz', 'f4'),     # Frequency in MHz (13-15 MHz range)
                ('amplitude_vpp', 'f4'),     # Amplitude in Vpp
                ('output_enabled', '?'),     # Boolean: output on/off
                ('event_type', 'S20')        # Event type: 'parameter_change', 'output_on', 'output_off', etc.
            ])
            
            # Create extensible dataset under /raw_data
            self.fg_timeline_dataset = data_group.create_dataset(
                'function_generator_timeline',
                shape=(0,),
                maxshape=(None,),
                dtype=timeline_dtype,
                chunks=True,
                compression='gzip',
                compression_opts=9
            )
            
            # Add metadata about the timeline dataset
            self.fg_timeline_dataset.attrs['description'] = b'Function generator parameter timeline'
            self.fg_timeline_dataset.attrs['timestamp_reference'] = b'Relative to recording start'
            self.fg_timeline_dataset.attrs['frequency_units'] = b'MHz'
            self.fg_timeline_dataset.attrs['amplitude_units'] = b'Volts peak-to-peak'
            
            
        except Exception as e:
            logger.error(f"Error creating FG timeline dataset: {e}")
            self.fg_timeline_dataset = None
    
    def _flush_fg_timeline_buffer(self):
        """Flush the function generator timeline buffer to disk."""
        if not self.fg_timeline_buffer or not hasattr(self, 'fg_timeline_dataset') or not self.fg_timeline_dataset:
            return
            
        try:
            # Current dataset size
            current_size = self.fg_timeline_dataset.shape[0]
            new_entries = len(self.fg_timeline_buffer)
            
            # Resize dataset to accommodate new entries
            self.fg_timeline_dataset.resize((current_size + new_entries,))
            
            # Write buffer to dataset
            self.fg_timeline_dataset[current_size:] = self.fg_timeline_buffer
            
            # Clear buffer
            self.fg_timeline_buffer.clear()
            
            
        except Exception as e:
            logger.error(f"Error flushing FG timeline buffer: {e}")
    
    def _create_metadata_group(self):
        """Create /meta_data group structure for hardware settings and recording info."""
        if not self.h5_file:
            return
            
        try:
            # Create main metadata group
            self.metadata_group = self.h5_file.create_group('meta_data')
            self.metadata_group.attrs['description'] = b'Hardware settings, recording info, and configuration'
            self.metadata_group.attrs['created_at'] = datetime.now().isoformat().encode('utf-8')
            
            # Create hardware_settings subgroup
            hardware_group = self.metadata_group.create_group('hardware_settings')
            hardware_group.attrs['description'] = b'Camera and stage hardware configuration'
            
            # Create recording_info subgroup (always create it)
            recording_group = self.metadata_group.create_group('recording_info')
            recording_group.attrs['description'] = b'Recording session metadata and timestamps'
            recording_group.attrs['recording_started'] = datetime.now().isoformat().encode('utf-8')
            recording_group.attrs['compression_level'] = self.compression_level
            recording_group.attrs['downscale_factor'] = self.downscale_factor
            recording_group.attrs['target_fps'] = self.fps
            recording_group.attrs['min_fps'] = self.min_fps
            recording_group.attrs['frame_shape'] = str(self.frame_shape).encode('utf-8')
            recording_group.attrs['original_frame_shape'] = str(self.original_frame_shape).encode('utf-8')
            
            # Store reference for later use
            self.recording_info_group = recording_group
            
            
        except Exception as e:
            logger.error(f"Error creating metadata group: {e}", exc_info=True)
            self.metadata_group = None

    def _create_execution_data_group(self):
        """Create /raw_data/LUT group structure (to be implemented - placeholder for now)."""
        if not self.h5_file:
            return
            
        try:
            # Get or create /raw_data group
            if 'raw_data' not in self.h5_file:
                data_group = self.h5_file.create_group('raw_data')
            else:
                data_group = self.h5_file['raw_data']
            
            # Create LUT subgroup (placeholder for future implementation)
            lut_group = data_group.create_group('LUT')
            lut_group.attrs['description'] = b'Look-up tables and z-stage data (to be implemented)'
            lut_group.attrs['status'] = b'placeholder'
            
            # Placeholder subgroups
            # lut_table_group = lut_group.create_group('LUT_table')
            # lut_table_group.attrs['description'] = b'LUT table data (to be implemented)'
            
            # z_stage_group = lut_group.create_group('z_stage_heights')
            # z_stage_group.attrs['description'] = b'Z-stage height data (to be implemented)'
            
            
            # Keep reference for compatibility
            self.execution_data_group = lut_group
            
        except Exception as e:
            logger.error(f"Error creating LUT group: {e}")
            self.execution_data_group = None
    
    def log_execution_data(self, execution_type: str, data: dict) -> bool:
        """
        Log execution data to the HDF5 file (replaces session file logging).
        
        Args:
            execution_type: Type of execution ('force_path_design', 'force_path_execution', etc.)
            data: Dictionary of execution data
            
        Returns:
            True if logged successfully
        """
        if not self.is_recording or not self.execution_data_group:
            return False
            
        try:
            import time
            
            # Create timestamped execution entry
            timestamp = int(time.time() * 1000)  # millisecond precision
            exec_name = f"{execution_type}_{timestamp}"
            
            execution_entry = self.execution_data_group.create_group(exec_name)
            execution_entry.attrs['execution_type'] = execution_type.encode('utf-8')
            execution_entry.attrs['timestamp'] = datetime.now().isoformat().encode('utf-8')
            
            # Calculate relative time from recording start
            if hasattr(self, 'start_time') and self.start_time:
                start_timestamp = self.start_time.timestamp() if isinstance(self.start_time, datetime) else self.start_time
                relative_time = time.time() - start_timestamp
                execution_entry.attrs['relative_time_s'] = relative_time
            
            # Store execution data as attributes
            for key, value in data.items():
                try:
                    if isinstance(value, str):
                        execution_entry.attrs[key] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        execution_entry.attrs[key] = value
                    elif value is None:
                        execution_entry.attrs[key] = 'None'.encode('utf-8')
                    else:
                        execution_entry.attrs[key] = str(value).encode('utf-8')
                except Exception as e:
                    logger.warning(f"Failed to store execution data '{key}': {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging execution data: {e}")
            return False
    

    

    

    
    def _start_async_writer(self):
        """Start the asynchronous frame writer thread."""
        if self._write_thread is None or not self._write_thread.is_alive():
            self._stop_writing.clear()
            self._write_thread = threading.Thread(
                target=self._async_write_worker, 
                name="HDF5Writer",
                daemon=True
            )
            self._write_thread.start()
    
    def _async_write_worker(self):
        """Optimized background worker for high-performance batch writing."""
        batch_frames = []
        batch_indices = []
        last_batch_time = time.time()
        batch_timeout = 0.05  # 50ms max batch delay for responsiveness
        
        while not self._stop_writing.is_set() or not self._write_queue.empty():
            try:
                # Collect frames for batch processing
                frames_collected = 0
                start_time = time.time()
                
                # Fill batch efficiently
                while (frames_collected < self._batch_size and 
                       time.time() - start_time < batch_timeout and
                       not self._stop_writing.is_set()):
                    
                    try:
                        frame_data = self._write_queue.get(timeout=0.01)
                        if frame_data is None:  # Shutdown signal
                            break
                        
                        frame, index = frame_data
                        batch_frames.append(frame)
                        batch_indices.append(index)
                        frames_collected += 1
                        
                    except queue.Empty:
                        break
                
                # Write batch if we have frames or timeout reached
                current_time = time.time()
                if (batch_frames and 
                    (len(batch_frames) >= self._batch_size or
                     current_time - last_batch_time >= batch_timeout or
                     self._write_queue.empty())):
                    
                    self._write_frame_batch_optimized(batch_frames, batch_indices)
                    self._batch_writes += 1
                    batch_frames.clear()
                    batch_indices.clear()
                    last_batch_time = current_time
                    
                # Small sleep to prevent busy waiting
                if not batch_frames:
                    time.sleep(0.001)  # 1ms sleep
                    
            except Exception as e:
                logger.error(f"Error in optimized async write worker: {e}")
                # Clear problematic batch and continue
                batch_frames.clear()
                batch_indices.clear()
                
        # Final batch write
        if batch_frames:
            self._write_frame_batch_optimized(batch_frames, batch_indices)
            
        logger.info(f"Async writer stopped: {self._frames_written} frames written in {self._batch_writes} batches")
    
    def _write_frame_batch_optimized(self, frames: List[np.ndarray], indices: List[int]):
        """Memory-optimized batch writing with robust error handling."""
        if not frames or not self.video_dataset:
            return
            
        # Memory optimization: process in smaller sub-batches if batch is large
        batch_size = len(frames)
        max_sub_batch = 10  # Process max 10 frames at once to manage memory
        
        try:
            with self._write_lock:
                # Ensure dataset is large enough
                max_index = max(indices)
                if max_index >= self.video_dataset.shape[0]:
                    self._grow_dataset_to_size(max_index + 1)
                
                # Process in sub-batches for memory efficiency
                for i in range(0, batch_size, max_sub_batch):
                    end_idx = min(i + max_sub_batch, batch_size)
                    sub_frames = frames[i:end_idx]
                    sub_indices = indices[i:end_idx]
                    
                    # Sort for sequential access (improves I/O performance)
                    if len(sub_frames) > 2:
                        sorted_pairs = sorted(zip(sub_indices, sub_frames), key=lambda x: x[0])
                        sub_indices, sub_frames = zip(*sorted_pairs)
                    
                    # Write sub-batch
                    for frame, index in zip(sub_frames, sub_indices):
                        try:
                            self.video_dataset[index] = frame
                            self._frames_written += 1
                        except Exception as write_error:
                            logger.error(f"Failed to write frame {index}: {write_error}")
                            self.write_errors += 1
                            if self.write_errors >= self.max_write_errors:
                                logger.error("Too many write errors, stopping recording")
                                self._closed = True
                                return
                    
                    # Trigger garbage collection every few sub-batches
                    if (i // max_sub_batch) % 5 == 0:
                        gc.collect()
                
                # Periodic flush for data safety (less frequent for performance)
                if self._frames_written % 1000 == 0:  # Every 1000 frames
                    try:
                        self.h5_file.flush()
                    except Exception as flush_error:
                        logger.warning(f"Flush failed: {flush_error}")
                        
        except Exception as e:
            logger.error(f"Critical error in batch write: {e}")
            self.write_errors += 1
    
    def _write_frame_batch(self, frames: List[np.ndarray], indices: List[int]):
        """Legacy batch write method for compatibility."""
        return self._write_frame_batch_optimized(frames, indices)
    
    def _grow_dataset_to_size(self, required_size: int):
        """Grow dataset to at least the required size."""
        current_size = self.video_dataset.shape[0]
        if required_size <= current_size:
            return
            
        # Grow by at least the growth factor, but ensure we have enough space
        new_size = max(int(current_size * self.growth_factor), required_size)
        
        
        # Resize dataset
        new_shape = (new_size, *self.frame_shape)
        self.video_dataset.resize(new_shape)
    
    def record_frame_async(self, frame: np.ndarray) -> bool:
        """
        Record a frame asynchronously for better performance.
        
        Args:
            frame: Frame data as numpy array
            
        Returns:
            True if frame queued successfully
        """
        if not self.is_recording or self._closed:
            return False
            
        try:
            # Validate frame
            if frame.shape != self.frame_shape:
                logger.warning(f"Frame shape mismatch: expected {self.frame_shape}, got {frame.shape}")
                return False
                
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Queue frame for async writing
            frame_copy = frame.copy()  # Make copy to avoid race conditions
            try:
                self._write_queue.put((frame_copy, self.frame_count), block=False)
                self.frame_count += 1
                return True
            except queue.Full:
                logger.warning("Write queue full, dropping frame")
                return False
                
        except Exception as e:
            logger.error(f"Error queuing frame for async write: {e}")
            return False
    

    
    def _grow_dataset(self):
        """Grow the dataset when it's full with robust error handling."""
        try:
            current_size = self.video_dataset.shape[0]
            new_size = int(current_size * self.growth_factor)
            
            
            # Check disk space before growing
            estimated_growth_mb = (new_size - current_size) * np.prod(self.frame_shape) / (1024**2)
            
            if not self._check_disk_space(estimated_growth_mb / 1024 + 0.5):  # Add 0.5GB buffer
                logger.error("Insufficient disk space to grow dataset")
                self.stop_recording()
                return
            
            # Resize dataset
            new_shape = (new_size, *self.frame_shape)
            self.video_dataset.resize(new_shape)
            
            logger.info(f"Dataset grown to {new_size} frames capacity")
            
        except Exception as e:
            try:
                logger.error(f"Failed to grow dataset: {e}")
            except OSError:
                pass  # Can't log, disk might be full
            self.stop_recording()
    
    def _emergency_cleanup(self) -> bool:
        """
        Emergency cleanup to free disk space by removing old files.
        
        Returns:
            True if space was freed
        """
        try:
            import glob
            import os
            
            # Get logs directory
            logs_dir = os.path.dirname(self.file_path)
            if not os.path.exists(logs_dir):
                return False
            
            # Find old HDF5 files to delete
            old_files = glob.glob(os.path.join(logs_dir, "*.hdf5"))
            old_files.sort(key=os.path.getctime)  # Sort by creation time
            
            space_freed = 0
            files_removed = 0
            
            # Remove older files (keep newest 3)
            for old_file in old_files[:-3]:  # Keep newest 3 files
                try:
                    file_size = os.path.getsize(old_file)
                    os.remove(old_file)
                    space_freed += file_size
                    files_removed += 1
                    
                    # Check if we've freed enough space (target 100MB)
                    if space_freed > 100 * 1024 * 1024:
                        break
                        
                except Exception:
                    continue
            
            if files_removed > 0:
                try:
                    logger.info(f"Emergency cleanup: removed {files_removed} files, freed {space_freed/(1024*1024):.1f}MB")
                except OSError:
                    pass
                return True
                
            return False
            
        except Exception:
            return False
    
    def stop_recording(self) -> bool:
        """
        Stop recording and ensure file is fully closed before returning.
        Robust error handling to prevent crashes.
        
        Returns:
            True if recording stopped successfully
        """
        if not self.is_recording or self._closed:
            logger.warning("No recording in progress")
            return False
        
        try:
            # Set closed flag to prevent further writes
            self._closed = True
            
            logger.info(f"Stopping HDF5 recording after {self.frame_count} frames...")
            
            # Stop accepting new frames immediately
            self.is_recording = False
            
            # Stop async writer and wait for all frames to be written
            try:
                self._stop_async_writer_sync()
            except Exception as e:
                logger.error(f"Error stopping async writer: {e}", exc_info=True)
                # Continue with finalization even if writer stop fails
            
            # Finalize recording immediately (not in background)
            try:
                self._finalize_recording()
            except Exception as e:
                logger.error(f"Error in finalization: {e}", exc_info=True)
                # Ensure file is closed even if finalization fails
                if hasattr(self, 'h5_file') and self.h5_file:
                    try:
                        logger.warning("Force closing HDF5 file due to finalization error")
                        self.h5_file.close()
                        self.h5_file = None
                    except Exception as close_err:
                        logger.error(f"Failed to force close file: {close_err}")
                raise
            
            return True
            
        except Exception as e:
            logger.error(f"Critical error in stop_recording: {e}", exc_info=True)
            # Attempt emergency cleanup
            try:
                if hasattr(self, 'h5_file') and self.h5_file:
                    self.h5_file.close()
                    self.h5_file = None
            except:
                pass
            raise
        
    def _stop_async_writer_sync(self):
        """Stop the async writer thread and wait for completion (synchronous)."""
        try:
            if self._write_thread and self._write_thread.is_alive():
                queue_size = self._write_queue.qsize()
                
                # Signal shutdown immediately
                self._stop_writing.set()
                
                # Send shutdown signal to queue (non-blocking)
                try:
                    self._write_queue.put(None, timeout=0.1)
                except queue.Full:
                    pass  # Queue full during shutdown
                except Exception as e:
                    logger.warning(f"Could not send shutdown signal to queue: {e}")
                
                # Wait for thread to complete with progress logging
                # Timeout: minimum 10s or 30ms per queued frame (reduced from 50ms for speed)
                timeout = max(10.0, queue_size * 0.03)
                
                start_wait = time.time()
                self._write_thread.join(timeout=timeout)
                wait_duration = time.time() - start_wait
                
                if self._write_thread.is_alive():
                    logger.warning(f"Async writer still running after {wait_duration:.1f}s - may lose data")
                else:
                    pass  # Writer completed successfully
            else:
                pass  # No async writer
        except Exception as e:
            logger.error(f"Exception in _stop_async_writer_sync: {e}", exc_info=True)
            # Continue execution - finalization will handle cleanup
    
    def _finalize_recording(self):
        """Complete recording finalization synchronously with optimized operations."""
        start_finalize = time.time()
        
        try:
            # Finalize HDF5 file with statistics
            if self.video_dataset and self.frame_count > 0:
                try:
                    # Resize dataset to exact frame count (fast operation)
                    final_shape = (self.frame_count, *self.frame_shape)
                    self.video_dataset.resize(final_shape)
                except Exception as e:
                    logger.error(f"Failed to resize dataset: {e}")
                
                # Add comprehensive recording statistics (fast - just metadata)
                try:
                    if self.start_time:
                        duration = (datetime.now() - self.start_time).total_seconds()
                        actual_fps = self.frame_count / duration if duration > 0 else 0
                        
                        # Batch attribute writes for efficiency
                        attrs = {
                            'recording_duration_s': duration,
                            'total_frames': self.frame_count,
                            'actual_fps': actual_fps,
                            'finished_at': datetime.now().isoformat(),
                        }
                        
                        # Performance statistics
                        frame_size_bytes = self.frame_shape[0] * self.frame_shape[1] * self.frame_shape[2]
                        total_data_mb = (self.frame_count * frame_size_bytes) / (1024 * 1024)
                        
                        attrs.update({
                            'frame_size_bytes': frame_size_bytes,
                            'total_data_mb': total_data_mb,
                            'fps_efficiency': (actual_fps / self.fps * 100) if self.fps > 0 else 0,
                        })
                        
                        # Write all attributes at once
                        for key, value in attrs.items():
                            try:
                                self.video_dataset.attrs[key] = value
                            except Exception as attr_err:
                                logger.warning(f"Failed to write attribute {key}: {attr_err}")
                        
                except Exception as e:
                    logger.warning(f"Failed to write recording statistics: {e}")
            
            # Flush function generator timeline (fast - small data)
            try:
                self._flush_fg_timeline_buffer()
            except Exception as e:
                logger.warning(f"Error flushing timeline buffer: {e}")
            
            # Final flush and close (the slowest operations due to compression)
            if self.h5_file:
                try:
                    flush_start = time.time()
                    self.h5_file.flush()
                    flush_duration = time.time() - flush_start
                except Exception as e:
                    logger.warning(f"Final flush failed: {e}")
            
            # Close HDF5 file safely
            if self.h5_file:
                try:
                    close_start = time.time()
                    self.h5_file.close()
                    close_duration = time.time() - close_start
                    
                    # Quick verification - just check file exists and is readable
                    try:
                        import os
                        if os.path.exists(self.file_path):
                            file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
                            logger.info(f"HDF5 file saved: {file_size_mb:.1f} MB")
                    except Exception as verify_error:
                        pass  # Verification optional
                        
                except Exception as e:
                    logger.error(f"Error closing HDF5 file: {e}", exc_info=True)
                    raise
                finally:
                    self.h5_file = None
            
            # Reset state
            self.video_dataset = None
            
            total_duration = time.time() - start_finalize
            rec_duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            logger.info(f"File saved: {self.file_path}")
            
        except Exception as e:
            logger.error(f"Error in recording finalization: {e}", exc_info=True)
            # Ensure file is closed even if error occurs
            if hasattr(self, 'h5_file') and self.h5_file:
                try:
                    logger.warning("Emergency close of HDF5 file")
                    self.h5_file.close()
                    self.h5_file = None
                except Exception as emergency_err:
                    logger.error(f"Emergency close failed: {emergency_err}")
            raise
    
    def get_recording_info(self) -> Dict[str, Any]:
        """Get current recording information."""
        info = {
            'is_recording': self.is_recording,
            'frame_count': self.frame_count,
            'file_path': self.file_path,
            'fps': self.fps,
            'compression': 'lzf',
        }
        
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            info['duration_seconds'] = duration
            info['actual_fps'] = self.frame_count / duration if duration > 0 else 0
        
        return info
    
    def __del__(self):
        """Ensure file is closed on destruction with robust cleanup."""
        if hasattr(self, 'h5_file') and self.h5_file:
            try:
                if hasattr(self, 'is_recording') and self.is_recording:
                    logger.warning("HDF5 recorder destroyed while recording - forcing stop")
                    self.stop_recording()
                else:
                    self.h5_file.close()
            except Exception as e:
                pass  # Cleanup error in destructor
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure recording is stopped."""
        if self.is_recording:
            try:
                self.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping recording in context manager: {e}")
        return False  # Don't suppress exceptions


# Utility functions for reading HDF5 video files
def load_hdf5_video_info(file_path: str) -> Dict[str, Any]:
    """
    Load metadata and information from an HDF5 video file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Dictionary containing video information and metadata
    """
    try:
        with h5py.File(file_path, 'r') as f:
            info = {}
            
            # Dataset information (new structure - /raw_data/main_video)
            if 'raw_data' in f and 'main_video' in f['raw_data']:
                video_ds = f['raw_data/main_video']
                info['shape'] = video_ds.shape
                info['dtype'] = str(video_ds.dtype)
                info['compression'] = video_ds.compression
                info['chunks'] = video_ds.chunks
                
                # Dataset attributes (including user metadata)
                for key, value in video_ds.attrs.items():
                    if isinstance(value, bytes):
                        info[key] = value.decode('utf-8')
                    else:
                        info[key] = value
            
            # Load metadata from /meta_data group
            if 'meta_data' in f:
                info['meta_data'] = {}
                for group_name in f['meta_data']:
                    info['meta_data'][group_name] = {}
                    group = f[f'meta_data/{group_name}']
                    for key, value in group.attrs.items():
                        if isinstance(value, bytes):
                            info['meta_data'][group_name][key] = value.decode('utf-8')
                        else:
                            info['meta_data'][group_name][key] = value
            
            return info
            
    except Exception as e:
        logger.error(f"Error loading HDF5 video info from {file_path}: {e}")
        return {}


def load_hdf5_frame(file_path: str, frame_index: int) -> Optional[np.ndarray]:
    """
    Load a specific frame from an HDF5 video file.
    
    Args:
        file_path: Path to the HDF5 file
        frame_index: Index of the frame to load
        
    Returns:
        Frame as numpy array or None if error
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if 'raw_data' not in f or 'main_video' not in f['raw_data']:
                return None
                
            video_ds = f['raw_data/main_video']
            if frame_index >= video_ds.shape[0]:
                return None
                
            return video_ds[frame_index]
            
    except Exception as e:
        logger.error(f"Error loading frame {frame_index} from {file_path}: {e}")
        return None


def load_hdf5_frame_range(file_path: str, start_frame: int, end_frame: int) -> Optional[np.ndarray]:
    """
    Load a range of frames from an HDF5 video file.
    
    Args:
        file_path: Path to the HDF5 file
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (exclusive)
        
    Returns:
        Frames as numpy array (n_frames, height, width, channels) or None if error
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if 'raw_data' not in f or 'main_video' not in f['raw_data']:
                return None
                
            video_ds = f['raw_data/main_video']
            if end_frame > video_ds.shape[0]:
                end_frame = video_ds.shape[0]
                
            if start_frame >= end_frame:
                return None
                
            return video_ds[start_frame:end_frame]
            
    except Exception as e:
        logger.error(f"Error loading frames {start_frame}-{end_frame} from {file_path}: {e}")
        return None


def post_process_compress_hdf5(file_path: str, quality_reduction: bool = True, 
                                parallel: bool = True) -> Optional[str]:
    """
    Apply aggressive post-processing compression to HDF5 video file.
    This is done AFTER recording completes to maximize compression.
    
    Strategy:
    1. Recompress video dataset with higher compression (GZIP level 9)
    2. Apply additional frame quality reduction if requested
    3. Use parallel processing for speed
    
    Args:
        file_path: Path to the HDF5 file to compress
        quality_reduction: If True, reduce quality slightly for better compression
        parallel: If True, use parallel processing (highly recommended)
        
    Returns:
        Path to compressed file (same as input, modified in-place) or None if error
    """
    logger.info(f"Starting post-processing compression: {file_path}")
    start_time = time.time()
    
    try:
        import shutil
        import tempfile
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        
        # Create temporary file for recompressed data
        temp_file = file_path + ".tmp.hdf5"
        
        try:
            # Open original and create new compressed file
            with h5py.File(file_path, 'r') as f_in:
                with h5py.File(temp_file, 'w', libver='latest') as f_out:
                    
                    # Copy and recompress video data with maximum compression
                    if 'raw_data' in f_in and 'main_video' in f_in['raw_data']:
                        logger.info("Recompressing video dataset with maximum compression...")
                        
                        video_in = f_in['raw_data/main_video']
                        n_frames = video_in.shape[0]
                        frame_shape = video_in.shape[1:]
                        
                        # Create output group and dataset with max compression
                        raw_data_out = f_out.create_group('raw_data')
                        raw_data_out.attrs['description'] = f_in['raw_data'].attrs.get('description', b'')
                        
                        # Maximum compression settings - OPTIMIZED FOR SIZE AND SPEED
                        video_out = raw_data_out.create_dataset(
                            'main_video',
                            shape=(n_frames, *frame_shape),
                            dtype=np.uint8,
                            chunks=(20, *frame_shape),  # Even larger chunks = faster (was 10)
                            compression='gzip',
                            compression_opts=9,  # Maximum compression for smallest files (was 6)
                            shuffle=True,  # Improves compression ratio
                            fletcher32=False  # Disable checksum for speed
                        )
                        
                        # Copy attributes
                        for key, value in video_in.attrs.items():
                            video_out.attrs[key] = value
                        video_out.attrs['post_compressed'] = True
                        video_out.attrs['post_compression_level'] = 'maximum'
                        
                        # Process frames in parallel batches - OPTIMIZED FOR SPEED
                        if parallel and n_frames > 100:
                            logger.info(f"Processing {n_frames} frames in parallel...")
                            
                            # LARGER batches for better speed (was 100)
                            batch_size = 500  # Process 500 frames at once
                            num_batches = (n_frames + batch_size - 1) // batch_size
                            
                            def process_batch(batch_idx):
                                start_idx = batch_idx * batch_size
                                end_idx = min(start_idx + batch_size, n_frames)
                                batch_frames = video_in[start_idx:end_idx]
                                
                                # Apply quality reduction if requested (FASTER: skip float conversion)
                                if quality_reduction:
                                    # AGGRESSIVE quality reduction: 80% for maximum compression
                                    # Still preserves enough detail for bead tracking
                                    batch_frames = ((batch_frames.astype(np.uint16) * 204) >> 8).astype(np.uint8)  # ~80%
                                
                                return start_idx, end_idx, batch_frames
                            
                            # Use MORE workers for speed (was 4)
                            with ThreadPoolExecutor(max_workers=8) as executor:
                                futures = [executor.submit(process_batch, i) for i in range(num_batches)]
                                
                                for future in futures:
                                    start_idx, end_idx, batch_frames = future.result()
                                    video_out[start_idx:end_idx] = batch_frames
                                    
                                    if start_idx % 1000 == 0:  # Log less frequently
                                        logger.info(f"Compressed {start_idx}/{n_frames} frames...")
                        else:
                            # Sequential processing for small files
                            logger.info(f"Processing {n_frames} frames sequentially...")
                            for i in range(0, n_frames, 500):  # Larger batches
                                end_idx = min(i + 500, n_frames)
                                batch = video_in[i:end_idx]
                                
                                if quality_reduction:
                                    batch = ((batch.astype(np.uint16) * 204) >> 8).astype(np.uint8)  # ~80%
                                
                                video_out[i:end_idx] = batch
                                
                                if i % 1000 == 0:
                                        logger.info(f"Compressed {i}/{n_frames} frames...")
                    
                    # Copy all other groups and datasets recursively
                    def copy_group(src_group, dst_group, path=''):
                        for key in src_group.keys():
                            if key == 'raw_data':
                                continue  # Already handled
                            
                            item = src_group[key]
                            item_path = f"{path}/{key}" if path else key
                            
                            if isinstance(item, h5py.Group):
                                # Copy group
                                new_group = dst_group.create_group(key)
                                # Copy attributes
                                for attr_key, attr_val in item.attrs.items():
                                    new_group.attrs[attr_key] = attr_val
                                # Recurse
                                copy_group(item, new_group, item_path)
                            elif isinstance(item, h5py.Dataset):
                                # Copy dataset with compression
                                dst_group.create_dataset(
                                    key,
                                    data=item[()],
                                    compression='gzip',
                                    compression_opts=9
                                )
                                # Copy attributes
                                for attr_key, attr_val in item.attrs.items():
                                    dst_group[key].attrs[attr_key] = attr_val
                    
                    logger.info("Copying metadata and other datasets...")
                    copy_group(f_in, f_out)
                    
                    # Add post-processing metadata
                    f_out.attrs['post_processed'] = True
                    f_out.attrs['post_processing_timestamp'] = datetime.now().isoformat()
                    f_out.attrs['quality_reduction_applied'] = quality_reduction
            
            # Get file sizes for comparison
            original_size = os.path.getsize(file_path)
            compressed_size = os.path.getsize(temp_file)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            logger.info(
                f"Post-compression complete: "
                f"Original {original_size/(1024**2):.1f} MB -> "
                f"Compressed {compressed_size/(1024**2):.1f} MB "
                f"({compression_ratio:.1f}% reduction)"
            )
            
            # CRITICAL: Force garbage collection BEFORE attempting file operations
            # This ensures h5py context managers are fully released
            import gc
            gc.collect()
            time.sleep(1)  # Give OS time to release file handles
            
            # Replace original file with compressed version (with retry logic for file locks)
            max_retries = 10  # More retries for robustness
            for attempt in range(max_retries):
                try:
                    # Additional garbage collection per attempt
                    gc.collect()
                    
                    # Remove original file first, then rename temp
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    time.sleep(0.2)  # Brief pause after delete
                    os.rename(temp_file, file_path)
                    logger.info("Successfully replaced original file with compressed version")
                    break
                except (PermissionError, FileExistsError, OSError) as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"File lock detected, retrying ({attempt + 1}/{max_retries})...")
                        time.sleep(1.5)  # Wait longer between retries
                    else:
                        logger.error(f"Failed to replace file after {max_retries} attempts: {e}")
                        logger.warning(f"Compressed file saved as: {temp_file}")
                        # Don't raise - just leave temp file and return it
                        return temp_file
            
            duration = time.time() - start_time
            logger.info(f"Post-processing completed in {duration:.1f} seconds")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error during post-processing compression: {e}", exc_info=True)
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                try:
                    time.sleep(1)
                    os.remove(temp_file)
                except:
                    pass
            return None
            
    except Exception as e:
        logger.error(f"Failed to post-process compress HDF5: {e}", exc_info=True)
        return None


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Example: Create a test recording
    test_file = "test_recording.hdf5"
    frame_shape = (480, 640, 3)  # Height, width, channels
    
    # Create recorder
    recorder = HDF5VideoRecorder(test_file, frame_shape, fps=60.0)
    
    # Test metadata
    metadata = {
        'sample_name': 'Test Sample',
        'operator': 'Test User',
        'notes': 'This is a test recording',
        'temperature': 23.5,
        'humidity': 45.0
    }
    
    # Start recording
    if recorder.start_recording(metadata):
        logger.info("HDF5 test recording started")
        
        # Record test frames
        for i in range(100):
            frame = np.random.randint(0, 256, frame_shape, dtype=np.uint8)
            frame[i % frame_shape[0], :, :] = 255  # Moving white line
            
            if not recorder.record_frame(frame):
                logger.error(f"Failed to record test frame {i}")
                break
        
        # Stop recording
        if recorder.stop_recording():
            logger.info("HDF5 test recording completed successfully")
            
            # Load and verify
            info = load_hdf5_video_info(test_file)
            logger.info(f"Test recording verification: {info}")
            
            # Load a test frame
            test_frame = load_hdf5_frame(test_file, 50)
            if test_frame is not None:
                logger.info(f"Test frame loaded successfully: shape {test_frame.shape}")
            
            # Clean up test file
            os.remove(test_file)
            logger.info("HDF5 test completed and cleaned up")
        else:
            logger.error("Failed to stop test recording")
    else:
        logger.error("Failed to start test recording")