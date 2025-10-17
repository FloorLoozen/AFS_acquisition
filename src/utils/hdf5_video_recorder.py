"""HDF5 Video Recorder for the AFS Tracking System.

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
                 fps: float = 60.0) -> None:
        """
        Initialize the HDF5 video recorder with optimized LZF compression.
        
        Args:
            file_path: Path to save the HDF5 file (str or Path object)
            frame_shape: Shape of each frame (height, width, channels)
            fps: Frames per second for metadata (must be > 0)
            
        Raises:
            ValueError: If frame_shape is invalid or fps <= 0
            OSError: If the parent directory doesn't exist or isn't writable
        """
        # Input validation using validation utilities
        self.frame_shape = validate_frame_shape(frame_shape, "frame_shape")
        self.fps = validate_positive_number(fps, "fps")
        file_path_obj = validate_file_path(file_path, must_exist=False, create_parent=True, field_name="file_path")
        
        self.file_path = str(file_path_obj)
        
        # Recording state with proper type annotations
        self.h5_file: Optional[h5py.File] = None
        self.video_dataset: Optional[h5py.Dataset] = None
        self.is_recording: bool = False
        self.frame_count: int = 0
        self.start_time: Optional[float] = None
        
        # Dataset parameters - optimized for performance
        self.chunk_size = self._calculate_optimal_chunk_size(frame_shape)
        self.initial_size = 2000  # Larger initial allocation to reduce resizing
        self.growth_factor = 2.0  # Exponential growth for better amortized performance
        
        # Performance tracking
        self.write_errors = 0
        self.max_write_errors = 10  # Stop recording after too many errors
        self.last_flush_time = 0
        self.flush_interval = 5.0  # Flush every 5 seconds for data safety
        
        # Additional data storage
        self.settings_group = None
        self._closed = False
        
        # Function generator timeline logging
        self.fg_timeline_dataset = None
        self.fg_timeline_buffer = []
        self.fg_timeline_buffer_size = 1000  # Buffer entries before writing to disk
        
        # Asynchronous write optimization
        self._write_queue = queue.Queue(maxsize=100)  # Limit memory usage
        self._write_thread = None
        self._stop_writing = threading.Event()
        self._write_executor = None
        
        # Frame batching for better I/O performance
        self._frame_batch = []
        self._batch_size = 10  # Batch frames before writing
        
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
            # Ensure directory exists
            dir_path = os.path.dirname(self.file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # Check available disk space before starting
            if not self._check_disk_space():
                logger.error("Insufficient disk space to start recording")
                return False
            
            # Open HDF5 file for writing with robust settings
            self.h5_file = h5py.File(
                self.file_path, 
                'w',
                libver='latest',  # Use latest HDF5 format for best performance
                swmr=False  # Single writer mode for better performance
            )
            
            # Create main folder structure
            data_group = self.h5_file.create_group('data')
            meta_data_group = self.h5_file.create_group('meta_data')
            
            # Create main video dataset under data/video
            # Shape: (n_frames, height, width, channels)
            initial_shape = (self.initial_size, *self.frame_shape)
            max_shape = (None, *self.frame_shape)  # Unlimited frames
            
            # Dataset creation parameters - optimized for speed and compression
            dataset_kwargs = {
                'shape': initial_shape,
                'maxshape': max_shape,
                'dtype': np.uint8,
                'chunks': self.chunk_size,
                'shuffle': True,  # Enable shuffle filter for better compression
                'fillvalue': 0,  # Set fill value for uninitialized data
                'track_times': False,  # Disable timestamp tracking for performance
            }
            
            # Use LZF compression for maximum speed (much faster than GZIP)
            dataset_kwargs['compression'] = 'lzf'
            # LZF doesn't use compression_opts - it's optimized for speed
            dataset_kwargs['fletcher32'] = True  # Add checksum for data integrity
            
            self.video_dataset = data_group.create_dataset('video', **dataset_kwargs)
            
            # Add dataset-level metadata
            self._add_dataset_metadata()
            
            # Add user-provided metadata
            if metadata:
                self._add_user_metadata(metadata, meta_data_group)
            
            # Create hardware settings group under meta_data
            self.settings_group = meta_data_group.create_group('hardware_settings')
            
            # Create placeholder groups for future implementation
            self._create_placeholder_groups(data_group, meta_data_group)
            
            # Create function generator timeline dataset under data
            self._create_fg_timeline_dataset(data_group)
            
            # Set recording state
            self.is_recording = True
            self.frame_count = 0
            self.start_time = datetime.now()
            
            # Start async write thread for better performance
            self._start_async_writer()
            
            logger.info(f"Started HDF5 recording: {self.file_path}")
            logger.info(f"Frame shape: {self.frame_shape}, FPS: {self.fps}, Compression: lzf")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HDF5 recording: {e}")
            if self.h5_file:
                self.h5_file.close()
                self.h5_file = None
            return False
    
    def _check_disk_space(self, min_gb: float = 1.0) -> bool:
        """
        Check if there's sufficient disk space for recording.
        
        Args:
            min_gb: Minimum required space in GB
            
        Returns:
            True if sufficient space available
        """
        try:
            import shutil
            total, used, free = shutil.disk_usage(os.path.dirname(self.file_path) or '.')
            free_gb = free / (1024**3)
            
            if free_gb < min_gb:
                logger.warning(f"Low disk space: {free_gb:.1f}GB available, {min_gb}GB required")
                return False
                
            logger.debug(f"Disk space OK: {free_gb:.1f}GB available")
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
        self.video_dataset.attrs['frame_shape'] = self.frame_shape
        self.video_dataset.attrs['compression'] = 'lzf'
        self.video_dataset.attrs['chunk_size'] = self.chunk_size
        
        # Timestamp information
        self.video_dataset.attrs['created_at'] = datetime.now().isoformat()
        
        # Format information
        self.video_dataset.attrs['format_version'] = '1.0'
        self.video_dataset.attrs['color_format'] = 'BGR'  # OpenCV default
        self.video_dataset.attrs['data_type'] = 'uint8'
        
        # AFS-specific metadata
        self.video_dataset.attrs['system'] = 'AFS_tracking'
        
    def _add_user_metadata(self, metadata: Dict[str, Any], meta_data_group: h5py.Group):
        """Add user-provided metadata directly to the meta_data group."""
        if not meta_data_group:
            return
            
        # Add user metadata directly to meta_data group
        meta_group = meta_data_group
        
        for key, value in metadata.items():
            if value is not None and value != "":
                try:
                    # Handle different data types appropriately
                    if isinstance(value, str):
                        meta_group.attrs[key] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, datetime):
                        meta_group.attrs[key] = value.isoformat()
                    else:
                        # Convert to string as fallback
                        meta_group.attrs[key] = str(value).encode('utf-8')
                        
                except Exception as e:
                    logger.warning(f"Could not save metadata '{key}': {e}")
    
    def _create_placeholder_groups(self, data_group: h5py.Group, meta_data_group: h5py.Group):
        """Create placeholder groups for future functionality."""
        try:
            # Data placeholders (function_generator_timeline is already implemented, so skip it)
            if 'look_up_table' not in data_group:
                lut_group = data_group.create_group('look_up_table')
                lut_group.attrs['description'] = 'Lookup table data'
                lut_group.attrs['status'] = 'placeholder'
            
            # Meta data placeholders under hardware_settings
            if 'hardware_settings' in meta_data_group:
                hw_settings = meta_data_group['hardware_settings']
                
                # Add other hardware settings placeholders
                if 'stage' not in hw_settings:
                    stage_group = hw_settings.create_group('stage')
                    stage_group.attrs['description'] = 'Stage controller settings'
                    stage_group.attrs['status'] = 'placeholder'
                
                if 'function_generator' not in hw_settings:
                    fg_hw_group = hw_settings.create_group('function_generator')
                    fg_hw_group.attrs['description'] = 'Function generator hardware settings'
                    fg_hw_group.attrs['status'] = 'placeholder'
                
                if 'oscilloscope' not in hw_settings:
                    osc_group = hw_settings.create_group('oscilloscope')
                    osc_group.attrs['description'] = 'Oscilloscope hardware settings'
                    osc_group.attrs['status'] = 'placeholder'
            
        except Exception as e:
            logger.warning(f"Could not create hardware metadata groups: {e}")
    
    def record_frame(self, frame: np.ndarray) -> bool:
        """
        Record a single frame to the HDF5 file with robust error handling.
        
        Args:
            frame: Frame data as numpy array (height, width, channels)
            
        Returns:
            True if frame recorded successfully
        """
        if not self.is_recording or not self.video_dataset or self._closed:
            return False
        
        # Check for too many write errors
        if self.write_errors >= self.max_write_errors:
            logger.error(f"Too many write errors ({self.write_errors}), stopping recording")
            self.stop_recording()
            return False
            
        try:
            # Validate frame shape and dtype
            if frame.shape != self.frame_shape:
                logger.warning(f"Frame shape mismatch: expected {self.frame_shape}, got {frame.shape}")
                self.write_errors += 1
                return False
            
            if frame.dtype != np.uint8:
                # Convert to uint8 if needed
                frame = frame.astype(np.uint8)
            
            # Check if we need to resize the dataset
            if self.frame_count >= self.video_dataset.shape[0]:
                self._grow_dataset()
            
            # Write frame to dataset
            self.video_dataset[self.frame_count] = frame
            self.frame_count += 1
            
            # Reset error count on successful write
            self.write_errors = 0
            
            # Smart flushing - time-based rather than frame-based for better performance
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
    
    def add_camera_settings(self, settings: Dict[str, Any]):
        """
        Add camera settings to the HDF5 file.
        
        Args:
            settings: Dictionary of camera settings
        """
        if not self.is_recording or not self.settings_group:
            return False
            
        try:
            camera_group = self.settings_group.create_group('camera')
            for key, value in settings.items():
                if value is not None:
                    if isinstance(value, str):
                        camera_group.attrs[key] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        camera_group.attrs[key] = value
                    else:
                        camera_group.attrs[key] = str(value).encode('utf-8')
            
            logger.debug(f"Saved camera settings: {len(settings)} parameters")
            return True
            
        except Exception as e:
            logger.error(f"Error saving camera settings: {e}")
            return False
    
    def add_stage_settings(self, settings: Dict[str, Any]):
        """
        Add XY stage settings to the HDF5 file.
        
        Args:
            settings: Dictionary of stage settings
        """
        if not self.is_recording or not self.settings_group:
            return False
            
        try:
            stage_group = self.settings_group.create_group('xy_stage')
            for key, value in settings.items():
                if value is not None:
                    if isinstance(value, str):
                        stage_group.attrs[key] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        stage_group.attrs[key] = value
                    else:
                        stage_group.attrs[key] = str(value).encode('utf-8')
            
            logger.debug(f"Saved stage settings: {len(settings)} parameters")
            return True
        except Exception as e:
            logger.error(f"Failed to save stage settings: {e}")
            return False
    
    def log_function_generator_event(self, frequency_mhz: float, amplitude_vpp: float, 
                                   output_enabled: bool = True, 
                                   event_type: str = 'parameter_change'):
        """Log a function generator timeline event.
        
        Args:
            frequency_mhz: Frequency in MHz (13-15 MHz range)
            amplitude_vpp: Amplitude in volts peak-to-peak
            output_enabled: Whether output is enabled
            event_type: Type of event ('parameter_change', 'output_on', 'output_off', etc.)
        """
        if not self.is_recording or not self.fg_timeline_dataset:
            return False
            
        try:
            current_time = time.time()
            # Convert start_time to timestamp if it's a datetime object
            start_timestamp = self.start_time.timestamp() if isinstance(self.start_time, datetime) else self.start_time
            relative_time = current_time - start_timestamp
            
            # Create timeline entry (simplified structure)
            timeline_entry = np.array([
                (relative_time, frequency_mhz, amplitude_vpp, 
                 output_enabled, event_type.encode('utf-8')[:20])
            ], dtype=self.fg_timeline_dataset.dtype)
            
            # Add to buffer
            self.fg_timeline_buffer.extend(timeline_entry)
            
            # Flush buffer if it's getting full
            if len(self.fg_timeline_buffer) >= self.fg_timeline_buffer_size:
                self._flush_fg_timeline_buffer()
            
            # Removed debug logging to reduce performance overhead
            return True
            
        except Exception as e:
            logger.error(f"Error logging FG timeline event: {e}")
            return False
    
    def _flush_fg_timeline_buffer(self):
        """Flush the function generator timeline buffer to disk."""
        if not self.fg_timeline_buffer or not self.fg_timeline_dataset:
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
            
            logger.debug(f"Flushed {new_entries} FG timeline entries to HDF5")
            
        except Exception as e:
            logger.error(f"Error flushing FG timeline buffer: {e}")
            
        except Exception as e:
            logger.error(f"Error saving stage settings: {e}")
            return False
    
    def _create_fg_timeline_dataset(self, data_group: h5py.Group):
        """Create the function generator timeline dataset directly in data group."""
        try:
            # Define compound datatype for timeline entries (simplified)
            timeline_dtype = np.dtype([
                ('timestamp', 'f8'),        # Relative time from recording start (seconds)
                ('frequency_mhz', 'f4'),     # Frequency in MHz (13-15 MHz range)
                ('amplitude_vpp', 'f4'),     # Amplitude in Vpp
                ('output_enabled', '?'),     # Boolean: output on/off
                ('event_type', 'S20')        # Event type: 'parameter_change', 'output_on', 'output_off', etc.
            ])
            
            # Create extensible dataset directly in data group
            self.fg_timeline_dataset = data_group.create_dataset(
                'function_generator_timeline',
                shape=(0,),
                maxshape=(None,),
                dtype=timeline_dtype,
                chunks=True,
                compression='lzf'
            )
            
            # Add metadata about the timeline dataset
            self.fg_timeline_dataset.attrs['description'] = b'Function generator parameter timeline'
            self.fg_timeline_dataset.attrs['timestamp_reference'] = b'Relative to recording start'
            self.fg_timeline_dataset.attrs['frequency_units'] = b'MHz'
            self.fg_timeline_dataset.attrs['amplitude_units'] = b'Volts peak-to-peak'
            self.fg_timeline_dataset.attrs['status'] = b'implemented'
            
            logger.debug("Function generator timeline dataset created")
            
        except Exception as e:
            logger.error(f"Error creating FG timeline dataset: {e}")
            self.fg_timeline_dataset = None
    
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
            logger.debug("Async HDF5 writer thread started")
    
    def _async_write_worker(self):
        """Background worker thread for writing frames to HDF5."""
        batch_frames = []
        batch_indices = []
        
        while not self._stop_writing.is_set() or not self._write_queue.empty():
            try:
                # Get frame data with timeout
                try:
                    frame_data = self._write_queue.get(timeout=0.1)
                    if frame_data is None:  # Shutdown signal
                        break
                    
                    frame, index = frame_data
                    batch_frames.append(frame)
                    batch_indices.append(index)
                    
                    # Write batch when full or queue is empty
                    if len(batch_frames) >= self._batch_size or self._write_queue.empty():
                        if batch_frames:
                            self._write_frame_batch(batch_frames, batch_indices)
                            batch_frames.clear()
                            batch_indices.clear()
                            
                except queue.Empty:
                    # Write any remaining frames in batch
                    if batch_frames:
                        self._write_frame_batch(batch_frames, batch_indices)
                        batch_frames.clear()
                        batch_indices.clear()
                    continue
                    
            except Exception as e:
                logger.error(f"Error in async write worker: {e}")
                
        # Final batch write
        if batch_frames:
            self._write_frame_batch(batch_frames, batch_indices)
            
        logger.debug("Async HDF5 writer thread stopped")
    
    def _write_frame_batch(self, frames: List[np.ndarray], indices: List[int]):
        """Write a batch of frames to the dataset efficiently."""
        if not frames or not self.video_dataset:
            return
            
        try:
            # Check if we need to grow the dataset
            max_index = max(indices)
            if max_index >= self.video_dataset.shape[0]:
                self._grow_dataset_to_size(max_index + 1)
            
            # Write all frames in batch
            for frame, index in zip(frames, indices):
                self.video_dataset[index] = frame
                
        except Exception as e:
            logger.error(f"Error writing frame batch: {e}")
    
    def _grow_dataset_to_size(self, required_size: int):
        """Grow dataset to at least the required size."""
        current_size = self.video_dataset.shape[0]
        if required_size <= current_size:
            return
            
        # Grow by at least the growth factor, but ensure we have enough space
        new_size = max(int(current_size * self.growth_factor), required_size)
        
        logger.debug(f"Growing dataset from {current_size} to {new_size} frames")
        
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
            
            logger.debug(f"Growing dataset from {current_size} to {new_size} frames")
            
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
            logger.error(f"Failed to grow dataset: {e}")
            self.stop_recording()
    
    def stop_recording(self) -> bool:
        """
        Stop recording and finalize the HDF5 file with comprehensive cleanup.
        
        Returns:
            True if recording stopped successfully
        """
        if not self.is_recording or self._closed:
            logger.warning("No recording in progress")
            return False
        
        # Set closed flag to prevent further writes
        self._closed = True
        
        # Stop async writer and wait for queue to drain
        self._stop_async_writer()
        
        try:
            logger.info(f"Stopping HDF5 recording after {self.frame_count} frames...")
            # Resize dataset to actual number of frames recorded
            if self.video_dataset and self.frame_count > 0:
                final_shape = (self.frame_count, *self.frame_shape)
                self.video_dataset.resize(final_shape)
                
                # Add comprehensive recording statistics
                if self.start_time:
                    duration = (datetime.now() - self.start_time).total_seconds()
                    actual_fps = self.frame_count / duration if duration > 0 else 0
                    
                    # Basic recording stats
                    self.video_dataset.attrs['recording_duration_s'] = duration
                    self.video_dataset.attrs['total_frames'] = self.frame_count
                    self.video_dataset.attrs['actual_fps'] = actual_fps
                    self.video_dataset.attrs['finished_at'] = datetime.now().isoformat()
                    
                    # Performance statistics
                    frame_size_bytes = self.frame_shape[0] * self.frame_shape[1] * self.frame_shape[2]
                    total_data_mb = (self.frame_count * frame_size_bytes) / (1024 * 1024)
                    
                    self.video_dataset.attrs['frame_size_bytes'] = frame_size_bytes
                    self.video_dataset.attrs['total_data_mb'] = total_data_mb
                    self.video_dataset.attrs['fps_efficiency'] = (actual_fps / self.fps * 100) if self.fps > 0 else 0
                    
                    # File size information (estimated compressed size)
                    try:
                        import os
                        if os.path.exists(self.file_path):
                            file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
                            self.video_dataset.attrs['file_size_mb'] = file_size_mb
                            compression_ratio = (total_data_mb / file_size_mb) if file_size_mb > 0 else 1.0
                            self.video_dataset.attrs['compression_ratio'] = compression_ratio
                    except Exception as e:
                        logger.debug(f"Could not calculate file size: {e}")
            
            # Flush any remaining function generator timeline events
            try:
                self._flush_fg_timeline_buffer()
            except Exception as e:
                logger.warning(f"Error flushing timeline buffer: {e}")
            
            # Final flush before closing
            if self.h5_file:
                try:
                    self.h5_file.flush()
                except Exception as e:
                    logger.warning(f"Final flush failed: {e}")
            
            # Close HDF5 file safely
            if self.h5_file:
                try:
                    self.h5_file.close()
                except Exception as e:
                    logger.warning(f"Error closing HDF5 file: {e}")
                finally:
                    self.h5_file = None
            
            # Reset state
            self.is_recording = False
            self.video_dataset = None
            self.settings_group = None
            
            duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            logger.info(f"HDF5 recording completed: {self.frame_count} frames in {duration:.1f}s")
            logger.info(f"File saved: {self.file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping HDF5 recording: {e}")
            return False
    
    def _stop_async_writer(self):
        """Stop the async writer thread and wait for completion."""
        if self._write_thread and self._write_thread.is_alive():
            # Signal shutdown
            self._stop_writing.set()
            
            # Send shutdown signal to queue
            try:
                self._write_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            
            # Wait for thread to finish (with timeout)
            self._write_thread.join(timeout=5.0)
            if self._write_thread.is_alive():
                logger.warning("Async writer thread did not stop gracefully")
            
            logger.debug("Async writer stopped")
    
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
                logger.debug(f"Error in HDF5 recorder destructor: {e}")


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
            
            # Dataset information
            if 'data' in f and 'video' in f['data']:
                video_ds = f['data/video']
                info['shape'] = video_ds.shape
                info['dtype'] = str(video_ds.dtype)
                info['compression'] = video_ds.compression
                info['chunks'] = video_ds.chunks
                
                # Dataset attributes
                for key, value in video_ds.attrs.items():
                    if isinstance(value, bytes):
                        info[key] = value.decode('utf-8')
                    else:
                        info[key] = value
            
            # User metadata (directly in meta_data group)
            if 'meta_data' in f:
                meta_group = f['meta_data']
                metadata = {}
                for key, value in meta_group.attrs.items():
                    if isinstance(value, bytes):
                        metadata[key] = value.decode('utf-8')
                    else:
                        metadata[key] = value
                # Only include user metadata if there are any user-defined attributes
                if metadata:
                    info['user_metadata'] = metadata
            
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
            if 'data' not in f or 'video' not in f['data']:
                return None
                
            video_ds = f['data/video']
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
            if 'data' not in f or 'video' not in f['data']:
                return None
                
            video_ds = f['data/video']
            if end_frame > video_ds.shape[0]:
                end_frame = video_ds.shape[0]
                
            if start_frame >= end_frame:
                return None
                
            return video_ds[start_frame:end_frame]
            
    except Exception as e:
        logger.error(f"Error loading frames {start_frame}-{end_frame} from {file_path}: {e}")
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