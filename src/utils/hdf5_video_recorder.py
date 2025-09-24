"""
HDF5 Video Recorder for the AFS Tracking System.
Provides high-performance video recording with frame-level access, compression, and metadata storage.
"""

import h5py
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import os

from src.utils.logger import get_logger

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
    
    def __init__(self, file_path: str, frame_shape: Tuple[int, int, int], 
                 fps: float = 60.0, compression: str = 'lzf'):
        """
        Initialize the HDF5 video recorder.
        
        Args:
            file_path: Path to save the HDF5 file
            frame_shape: Shape of each frame (height, width, channels)
            fps: Frames per second for metadata
            compression: Compression type ('lzf', 'gzip', or None)
        """
        self.file_path = file_path
        self.frame_shape = frame_shape
        self.fps = fps
        self.compression = compression
        
        # Recording state
        self.h5_file = None
        self.video_dataset = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        
        # Dataset parameters
        self.chunk_size = self._calculate_optimal_chunk_size(frame_shape)
        self.initial_size = 1000  # Initial number of frames to allocate
        self.growth_factor = 1.5  # Factor to grow dataset when full
        
        # Additional data storage
        self.settings_group = None
        self.timeseries_group = None
        
    def _calculate_optimal_chunk_size(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, ...]:
        """
        Calculate optimal chunk size for the dataset.
        
        Frame-level chunking for optimal random access:
        - Chunk size = (1, height, width, channels) for single frame access
        
        Args:
            frame_shape: Shape of each frame (height, width, channels)
            
        Returns:
            Optimal chunk size tuple
        """
        # Single frame chunks for optimal random access
        return (1, *frame_shape)
    
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
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            # Open HDF5 file for writing
            self.h5_file = h5py.File(self.file_path, 'w')
            
            # Create main video dataset with compression and chunking
            # Shape: (n_frames, height, width, channels)
            initial_shape = (self.initial_size, *self.frame_shape)
            max_shape = (None, *self.frame_shape)  # Unlimited frames
            
            # Dataset creation parameters
            dataset_kwargs = {
                'shape': initial_shape,
                'maxshape': max_shape,
                'dtype': np.uint8,
                'chunks': self.chunk_size,
                'shuffle': True,  # Enable shuffle filter for better compression
            }
            
            # Add compression if specified
            if self.compression:
                dataset_kwargs['compression'] = self.compression
                if self.compression == 'gzip':
                    dataset_kwargs['compression_opts'] = 6  # Medium compression level
            
            self.video_dataset = self.h5_file.create_dataset('video', **dataset_kwargs)
            
            # Add dataset-level metadata
            self._add_dataset_metadata()
            
            # Add user-provided metadata
            if metadata:
                self._add_user_metadata(metadata)
            
            # Create groups for additional data
            self.settings_group = self.h5_file.create_group('settings')
            self.timeseries_group = self.h5_file.create_group('timeseries')
            
            # Set recording state
            self.is_recording = True
            self.frame_count = 0
            self.start_time = datetime.now()
            
            logger.info(f"Started HDF5 recording: {self.file_path}")
            logger.info(f"Frame shape: {self.frame_shape}, FPS: {self.fps}, Compression: {self.compression}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HDF5 recording: {e}")
            if self.h5_file:
                self.h5_file.close()
                self.h5_file = None
            return False
    
    def _add_dataset_metadata(self):
        """Add technical metadata to the video dataset."""
        if not self.video_dataset:
            return
            
        # Recording parameters
        self.video_dataset.attrs['fps'] = self.fps
        self.video_dataset.attrs['frame_shape'] = self.frame_shape
        self.video_dataset.attrs['compression'] = self.compression or 'none'
        self.video_dataset.attrs['chunk_size'] = self.chunk_size
        
        # Timestamp information
        self.video_dataset.attrs['created_at'] = datetime.now().isoformat()
        
        # Format information
        self.video_dataset.attrs['format_version'] = '1.0'
        self.video_dataset.attrs['color_format'] = 'BGR'  # OpenCV default
        self.video_dataset.attrs['data_type'] = 'uint8'
        
        # AFS-specific metadata
        self.video_dataset.attrs['system'] = 'AFS_tracking'
        self.video_dataset.attrs['pixel_size_um'] = 0.1  # Placeholder - should come from hardware
        
    def _add_user_metadata(self, metadata: Dict[str, Any]):
        """Add user-provided metadata to the file."""
        if not self.h5_file:
            return
            
        # Create a metadata group for user data
        meta_group = self.h5_file.create_group('metadata')
        
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
    
    def record_frame(self, frame: np.ndarray) -> bool:
        """
        Record a single frame to the HDF5 file.
        
        Args:
            frame: Frame data as numpy array (height, width, channels)
            
        Returns:
            True if frame recorded successfully
        """
        if not self.is_recording or not self.video_dataset:
            return False
            
        try:
            # Validate frame shape
            if frame.shape != self.frame_shape:
                logger.warning(f"Frame shape mismatch: expected {self.frame_shape}, got {frame.shape}")
                return False
            
            # Check if we need to resize the dataset
            if self.frame_count >= self.video_dataset.shape[0]:
                self._grow_dataset()
            
            # Write frame to dataset
            self.video_dataset[self.frame_count] = frame
            self.frame_count += 1
            
            # Flush to disk periodically for data safety
            if self.frame_count % 100 == 0:
                self.h5_file.flush()
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording frame {self.frame_count}: {e}")
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
            logger.error(f"Error saving stage settings: {e}")
            return False
    
    def add_function_generator_data(self, timestamps: np.ndarray, values: np.ndarray, 
                                   channel_name: str = 'output'):
        """
        Add function generator output data over time.
        
        Args:
            timestamps: Array of timestamps (in seconds from recording start)
            values: Array of output values
            channel_name: Name of the channel/signal
        """
        if not self.is_recording or not self.timeseries_group:
            return False
            
        try:
            # Create dataset for this channel if it doesn't exist
            if channel_name not in self.timeseries_group:
                # Create compound dataset with timestamp and value
                dt = np.dtype([('timestamp', 'f8'), ('value', 'f8')])
                max_points = 1000000  # Allow up to 1M data points
                
                dataset = self.timeseries_group.create_dataset(
                    channel_name,
                    shape=(len(timestamps),),
                    maxshape=(max_points,),
                    dtype=dt,
                    chunks=True,
                    compression='gzip'
                )
                
                # Add metadata
                dataset.attrs['description'] = f'Function generator {channel_name} over time'
                dataset.attrs['units_time'] = 'seconds'
                dataset.attrs['units_value'] = 'volts'
            else:
                dataset = self.timeseries_group[channel_name]
                
                # Resize dataset to accommodate new data
                old_size = dataset.shape[0]
                new_size = old_size + len(timestamps)
                dataset.resize((new_size,))
            
            # Store data
            data = np.empty(len(timestamps), dtype=dataset.dtype)
            data['timestamp'] = timestamps
            data['value'] = values
            
            if channel_name in self.timeseries_group:
                # Append to existing data
                old_size = dataset.shape[0] - len(timestamps)
                dataset[old_size:] = data
            else:
                # First data for this channel
                dataset[:] = data
            
            logger.debug(f"Saved {len(timestamps)} function generator data points for {channel_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving function generator data: {e}")
            return False
    
    def add_stage_position_data(self, timestamps: np.ndarray, x_positions: np.ndarray, 
                               y_positions: np.ndarray):
        """
        Add XY stage position data over time.
        
        Args:
            timestamps: Array of timestamps (in seconds from recording start)
            x_positions: Array of X positions in mm
            y_positions: Array of Y positions in mm
        """
        if not self.is_recording or not self.timeseries_group:
            return False
            
        try:
            # Create compound dataset for stage positions
            dt = np.dtype([('timestamp', 'f8'), ('x_mm', 'f8'), ('y_mm', 'f8')])
            
            if 'stage_position' not in self.timeseries_group:
                max_points = 1000000
                
                dataset = self.timeseries_group.create_dataset(
                    'stage_position',
                    shape=(len(timestamps),),
                    maxshape=(max_points,),
                    dtype=dt,
                    chunks=True,
                    compression='gzip'
                )
                
                dataset.attrs['description'] = 'XY stage position over time'
                dataset.attrs['units_time'] = 'seconds'
                dataset.attrs['units_position'] = 'mm'
            else:
                dataset = self.timeseries_group['stage_position']
                old_size = dataset.shape[0]
                new_size = old_size + len(timestamps)
                dataset.resize((new_size,))
            
            # Store data
            data = np.empty(len(timestamps), dtype=dataset.dtype)
            data['timestamp'] = timestamps
            data['x_mm'] = x_positions
            data['y_mm'] = y_positions
            
            if 'stage_position' in self.timeseries_group:
                old_size = dataset.shape[0] - len(timestamps)
                dataset[old_size:] = data
            else:
                dataset[:] = data
            
            logger.debug(f"Saved {len(timestamps)} stage position data points")
            return True
            
        except Exception as e:
            logger.error(f"Error saving stage position data: {e}")
            return False
    
    def _grow_dataset(self):
        """Grow the dataset when it's full."""
        current_size = self.video_dataset.shape[0]
        new_size = int(current_size * self.growth_factor)
        
        logger.debug(f"Growing dataset from {current_size} to {new_size} frames")
        
        # Resize dataset
        new_shape = (new_size, *self.frame_shape)
        self.video_dataset.resize(new_shape)
    
    def stop_recording(self) -> bool:
        """
        Stop recording and finalize the HDF5 file.
        
        Returns:
            True if recording stopped successfully
        """
        if not self.is_recording:
            logger.warning("No recording in progress")
            return False
        
        try:
            # Resize dataset to actual number of frames recorded
            if self.video_dataset and self.frame_count > 0:
                final_shape = (self.frame_count, *self.frame_shape)
                self.video_dataset.resize(final_shape)
                
                # Add final recording statistics
                if self.start_time:
                    duration = (datetime.now() - self.start_time).total_seconds()
                    actual_fps = self.frame_count / duration if duration > 0 else 0
                    
                    self.video_dataset.attrs['recording_duration_s'] = duration
                    self.video_dataset.attrs['total_frames'] = self.frame_count
                    self.video_dataset.attrs['actual_fps'] = actual_fps
                    self.video_dataset.attrs['finished_at'] = datetime.now().isoformat()
            
            # Close HDF5 file
            if self.h5_file:
                self.h5_file.close()
                self.h5_file = None
            
            # Reset state
            self.is_recording = False
            self.video_dataset = None
            
            duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            logger.info(f"HDF5 recording completed: {self.frame_count} frames in {duration:.1f}s")
            logger.info(f"File saved: {self.file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping HDF5 recording: {e}")
            return False
    
    def get_recording_info(self) -> Dict[str, Any]:
        """Get current recording information."""
        info = {
            'is_recording': self.is_recording,
            'frame_count': self.frame_count,
            'file_path': self.file_path,
            'fps': self.fps,
            'compression': self.compression,
        }
        
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            info['duration_seconds'] = duration
            info['actual_fps'] = self.frame_count / duration if duration > 0 else 0
        
        return info
    
    def __del__(self):
        """Ensure file is closed on destruction."""
        if self.h5_file:
            try:
                self.h5_file.close()
            except:
                pass


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
            if 'video' in f:
                video_ds = f['video']
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
            
            # User metadata
            if 'metadata' in f:
                meta_group = f['metadata']
                metadata = {}
                for key, value in meta_group.attrs.items():
                    if isinstance(value, bytes):
                        metadata[key] = value.decode('utf-8')
                    else:
                        metadata[key] = value
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
        print("Recording started...")
        
        # Record some test frames
        for i in range(100):
            # Create a test frame with changing pattern
            frame = np.random.randint(0, 256, frame_shape, dtype=np.uint8)
            frame[i % frame_shape[0], :, :] = 255  # Moving white line
            
            if not recorder.record_frame(frame):
                print(f"Failed to record frame {i}")
                break
        
        # Stop recording
        if recorder.stop_recording():
            print("Recording completed successfully")
            
            # Load and verify
            info = load_hdf5_video_info(test_file)
            print(f"Recorded video info: {info}")
            
            # Load a test frame
            test_frame = load_hdf5_frame(test_file, 50)
            if test_frame is not None:
                print(f"Successfully loaded frame 50: shape {test_frame.shape}")
            
            # Clean up test file
            os.remove(test_file)
            print("Test completed and cleaned up")
        else:
            print("Failed to stop recording")
    else:
        print("Failed to start recording")