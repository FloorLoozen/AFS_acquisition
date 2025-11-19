"""System-wide constants and configuration values.

This module centralizes all magic numbers and configuration constants
used throughout the application for better maintainability.
"""


class RecordingConstants:
    """HDF5 recording configuration constants."""
    
    # Queue sizes
    MAX_WRITE_QUEUE_SIZE = 800  # Frames buffered for async writing
    MAX_PROCESS_QUEUE_SIZE = 800  # Frames buffered for GPU processing
    
    # Batch processing
    BATCH_SIZE = 200  # Frames per write batch
    MAX_SUB_BATCH = 40  # Maximum sub-batch for memory control
    
    # Performance tuning
    TARGET_CHUNK_MB = 6  # Target chunk size in MB for HDF5
    FLUSH_INTERVAL_SEC = 15.0  # Time between HDF5 flushes
    
    # Timeouts
    WRITE_TIMEOUT_SEC = 10.0  # Minimum timeout for writer thread
    FRAME_TIMEOUT_MS = 30  # Milliseconds per queued frame for timeout calculation
    LUT_ONLY_TIMEOUT_SEC = 2.0  # Short timeout for LUT-only recordings
    
    # Compression
    RECORDING_COMPRESSION_LEVEL = 0  # No compression during recording (speed)
    POST_COMPRESSION_LEVEL = 4  # GZIP level for post-processing
    
    # Emergency thresholds
    MIN_DISK_SPACE_MB = 50  # Minimum free disk space before emergency stop


class DisplayConstants:
    """Live display configuration constants."""
    
    # Frame rates
    LIVE_DISPLAY_FPS = 12  # Target FPS for UI updates
    RECORDING_DISPLAY_FPS = 12  # Display FPS during recording
    
    # Timing
    MIN_DISPLAY_INTERVAL = 1.0 / 12.0  # Minimum time between frames (seconds)
    FALLBACK_TIMER_MS = int(1000.0 / 12.0)  # Timer interval in milliseconds
    
    # Status updates
    STATUS_UPDATE_INTERVAL = 0.5  # Seconds between status bar updates


class CameraConstants:
    """Camera capture configuration constants."""
    
    # Queue management
    DEFAULT_QUEUE_SIZE = 1  # Minimize latency for live view
    RECORDING_QUEUE_SIZE = 10  # Larger queue for high-speed recording
    
    # Frame pool
    FRAME_POOL_SIZE = 10  # Pre-allocated frame buffers
    
    # Performance
    MAX_FPS = 120  # Maximum supported FPS
    TARGET_RECORDING_FPS = 50  # Target FPS for recording
    
    # Test pattern
    TEST_PATTERN_FPS = 30  # FPS for test pattern mode


class GPUConstants:
    """GPU processing configuration constants."""
    
    # Buffer management
    GPU_BUFFER_POOL_SIZE = 4  # Number of pre-allocated GPU buffers
    
    # Batch sizes
    GPU_BATCH_SIZE = 4  # Frames processed in parallel on GPU
    
    # Performance tracking
    GPU_TIMING_SAMPLES = 100  # Number of samples for performance stats


class LoggerConstants:
    """Logging configuration constants."""
    
    # Cache management
    MAX_CACHE_SIZE = 1000  # Maximum number of cached log messages
    CACHE_CLEANUP_INTERVAL = 500  # Cleanup after this many log messages
    
    # Message throttling
    MAX_REPEATS = 2  # Maximum number of repeated messages to show
    
    # Throttle frequencies
    EXPOSURE_CHANGE_FREQUENCY = 5  # Show every Nth exposure change
    GAIN_CHANGE_FREQUENCY = 5  # Show every Nth gain change


class HardwareConstants:
    """Hardware controller configuration constants."""
    
    # Connection
    MAX_CONNECT_RETRIES = 3  # Maximum connection retry attempts
    CONNECT_RETRY_DELAY = 1.0  # Seconds between retries
    
    # Health monitoring
    HEALTH_CHECK_INTERVAL = 5.0  # Seconds between health checks
    
    # Timeouts
    VISA_TIMEOUT_MS = 5000  # VISA communication timeout
    STAGE_MOVE_TIMEOUT_SEC = 30.0  # Maximum time for stage movement


class ValidationConstants:
    """Validation and range limits."""
    
    # Function generator
    MIN_FREQUENCY_MHZ = 0.001  # Minimum frequency (1 kHz)
    MAX_FREQUENCY_MHZ = 30.0  # Maximum frequency (30 MHz)
    MIN_AMPLITUDE_VPP = 0.002  # Minimum amplitude (2 mVpp)
    MAX_AMPLITUDE_VPP = 20.0  # Maximum amplitude (20 Vpp)
    
    # Stage limits (micrometers)
    MIN_STAGE_POSITION_UM = 0  # Minimum stage position
    MAX_STAGE_POSITION_UM = 25000  # Maximum stage position (25mm)


class FileConstants:
    """File and path configuration constants."""
    
    # File formats
    HDF5_EXTENSION = '.hdf5'
    LOG_EXTENSION = '.log'
    CONFIG_EXTENSION = '.json'
    
    # Backup
    MAX_BACKUP_FILES = 3  # Number of backup files to keep
    
    # Emergency cleanup
    EMERGENCY_CLEANUP_THRESHOLD_MB = 100  # Free up to this much space
    MIN_FILE_AGE_SEC = 60  # Don't delete files newer than this


# Convenience exports
__all__ = [
    'RecordingConstants',
    'DisplayConstants', 
    'CameraConstants',
    'GPUConstants',
    'LoggerConstants',
    'HardwareConstants',
    'ValidationConstants',
    'FileConstants',
]
