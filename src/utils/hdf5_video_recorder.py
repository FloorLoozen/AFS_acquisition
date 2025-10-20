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
        
        # Recording state
        self._closed = False
        
        # Metadata storage
        self.camera_settings_saved = False
        self.stage_settings_saved = False
        
        # Function generator timeline logging
        self.fg_timeline_buffer = []
        self.fg_timeline_buffer_size = 1000  # Buffer entries before writing to disk
        
        # High-performance asynchronous writing
        self._write_queue = queue.Queue(maxsize=200)  # Increased buffer for high FPS
        self._write_thread = None
        self._stop_writing = threading.Event()
        self._write_lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Optimized frame batching
        self._frame_batch = []
        self._batch_indices = []
        self._batch_size = 25  # Larger batches for better I/O efficiency
        self._batch_lock = threading.Lock()
        
        # Performance counters
        self._frames_queued = 0
        self._frames_written = 0
        self._batch_writes = 0
        
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
                # Try emergency cleanup first
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
            
            # Open HDF5 file for writing with robust settings
            self.h5_file = h5py.File(
                self.file_path, 
                'w',
                libver='latest',  # Use latest HDF5 format for best performance
                swmr=False  # Single writer mode for better performance
            )
            
            # Create simple video dataset directly in root (original simple structure)
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
            
            self.video_dataset = self.h5_file.create_dataset('video', **dataset_kwargs)
            
            # Add dataset-level metadata
            self._add_dataset_metadata()
            
            # Add user-provided metadata directly to dataset attributes
            if metadata:
                self._add_user_metadata_to_dataset(metadata)
            
            # Create function generator timeline dataset
            self._create_fg_timeline_dataset()
            
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
        Record a single frame with optimized async/sync writing.
        
        Args:
            frame: Frame data as numpy array (height, width, channels)
            use_async: Use asynchronous writing for better performance
            
        Returns:
            True if frame queued/recorded successfully
        """
        if not self.is_recording or not self.video_dataset or self._closed:
            return False
        
        # Check for too many write errors
        if self.write_errors >= self.max_write_errors:
            logger.error(f"Too many write errors ({self.write_errors}), stopping recording")
            self.stop_recording()
            return False
        
        # Fast async path for high performance
        if use_async:
            return self._record_frame_async(frame)
        
        # Synchronous fallback for compatibility
        return self._record_frame_sync(frame)
    
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
        Add camera settings to the HDF5 file as dataset attributes.
        
        Args:
            settings: Dictionary of camera settings
            
        Returns:
            True if settings saved successfully
        """
        if not self.is_recording or not self.video_dataset or self.camera_settings_saved:
            return False
            
        try:
            # Add camera settings as dataset attributes with prefix
            for key, value in settings.items():
                if value is not None:
                    attr_name = f"camera_{key}"
                    if isinstance(value, str):
                        self.video_dataset.attrs[attr_name] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        self.video_dataset.attrs[attr_name] = value
                    else:
                        self.video_dataset.attrs[attr_name] = str(value).encode('utf-8')
            
            self.camera_settings_saved = True
            logger.debug(f"Saved camera settings: {len(settings)} parameters")
            return True
            
        except Exception as e:
            logger.error(f"Error saving camera settings: {e}")
            return False
    
    def add_stage_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Add XY stage settings to the HDF5 file as dataset attributes.
        
        Args:
            settings: Dictionary of stage settings
            
        Returns:
            True if settings saved successfully
        """
        if not self.is_recording or not self.video_dataset or self.stage_settings_saved:
            return False
            
        try:
            # Add stage settings as dataset attributes with prefix
            for key, value in settings.items():
                if value is not None:
                    attr_name = f"stage_{key}"
                    if isinstance(value, str):
                        self.video_dataset.attrs[attr_name] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        self.video_dataset.attrs[attr_name] = value
                    else:
                        self.video_dataset.attrs[attr_name] = str(value).encode('utf-8')
            
            self.stage_settings_saved = True
            logger.debug(f"Saved XY stage settings: {len(settings)} parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save XY stage settings: {e}")
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
        """Create the function generator timeline dataset."""
        if not self.h5_file:
            return
            
        try:
            # Define compound datatype for timeline entries
            timeline_dtype = np.dtype([
                ('timestamp', 'f8'),        # Relative time from recording start (seconds)
                ('frequency_mhz', 'f4'),     # Frequency in MHz (13-15 MHz range)
                ('amplitude_vpp', 'f4'),     # Amplitude in Vpp
                ('output_enabled', '?'),     # Boolean: output on/off
                ('event_type', 'S20')        # Event type: 'parameter_change', 'output_on', 'output_off', etc.
            ])
            
            # Create extensible dataset directly in root
            self.fg_timeline_dataset = self.h5_file.create_dataset(
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
            
            logger.debug("Function generator timeline dataset created")
            
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
            
            logger.debug(f"Flushed {new_entries} FG timeline entries to HDF5")
            
        except Exception as e:
            logger.error(f"Error flushing FG timeline buffer: {e}")
    

    

    

    
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
        Stop recording with immediate UI response and background finalization.
        
        Returns:
            True if recording stop initiated successfully
        """
        if not self.is_recording or self._closed:
            logger.warning("No recording in progress")
            return False
        
        # Set closed flag to prevent further writes
        self._closed = True
        
        logger.info(f"Stopping HDF5 recording after {self.frame_count} frames...")
        
        # Stop accepting new frames immediately for UI responsiveness
        self.is_recording = False
        
        # Start background finalization (non-blocking)
        import threading
        finalize_thread = threading.Thread(
            target=self._finalize_recording_background,
            name="HDF5Finalizer",
            daemon=True
        )
        finalize_thread.start()
        
        return True
        
    def _finalize_recording_background(self):
        """Complete recording finalization in background thread."""
        try:
            # Stop async writer and wait for queue to drain
            self._stop_async_writer()
            
            # Finalize HDF5 file with statistics
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
            self.video_dataset = None
            
            duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            logger.info(f"HDF5 recording completed: {self.frame_count} frames in {duration:.1f}s")
            logger.info(f"File saved: {self.file_path}")
            
        except Exception as e:
            logger.error(f"Error in background recording finalization: {e}")
            # Ensure file is closed even if error occurs
            if hasattr(self, 'h5_file') and self.h5_file:
                try:
                    self.h5_file.close()
                    self.h5_file = None
                except:
                    pass
    
    def _stop_async_writer(self):
        """Stop the async writer thread with non-blocking approach."""
        if self._write_thread and self._write_thread.is_alive():
            queue_size = self._write_queue.qsize()
            logger.debug(f"Stopping async writer: {queue_size} frames in queue")
            
            # Signal shutdown immediately
            self._stop_writing.set()
            
            # Send shutdown signal to queue (non-blocking)
            try:
                self._write_queue.put(None, timeout=0.1)
            except queue.Full:
                # Force clear queue if full
                try:
                    while True:
                        self._write_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._write_queue.put(None, timeout=0.1)
                except:
                    pass
            
            # Use shorter, more reasonable timeout for responsiveness
            # Still scale with queue size but cap at reasonable limit
            base_timeout = 2.0  # Shorter base timeout
            timeout = min(5.0, max(base_timeout, queue_size * 0.02))  # More reasonable scaling
            
            self._write_thread.join(timeout=timeout)
            if self._write_thread.is_alive():
                # Continue in background - don't block UI
                logger.info(f"Async writer continuing in background ({queue_size} frames remaining)")
                # Don't force terminate - let it finish gracefully in background
            else:
                logger.info(f"Async writer stopped: {self._frames_written} frames written, {self._batch_writes} batches")
    
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
            
            # Dataset information (simple structure - video dataset in root)
            if 'video' in f:
                video_ds = f['video']
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
            if 'video' not in f:
                return None
                
            video_ds = f['video']
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
            if 'video' not in f:
                return None
                
            video_ds = f['video']
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