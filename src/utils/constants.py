"""System-wide constants for AFS Acquisition.

This module contains all configuration constants grouped by functional area.
Constants are organized to make it easy to adjust performance, hardware limits,
and system behavior without changing code.
"""

# ============================================================================
# CAMERA SETTINGS
# ============================================================================
# Camera frame buffering and acquisition
DEFAULT_QUEUE_SIZE = 1              # Queue size for live display (minimize latency)
RECORDING_QUEUE_SIZE = 10           # Queue size during recording (balance throughput/memory)
FRAME_POOL_SIZE = 10                # Pre-allocated frame buffers to reduce GC pressure
MAX_FPS = 120                       # Maximum camera frame rate supported
TARGET_RECORDING_FPS = 50           # Target FPS for recording operations


# ============================================================================
# DISPLAY & UI SETTINGS
# ============================================================================
# Live display refresh rates
LIVE_DISPLAY_FPS = 15               # Target FPS for UI display updates
MIN_DISPLAY_INTERVAL = 1.0 / 15.0   # Minimum time between display updates (seconds)
STATUS_UPDATE_INTERVAL = 0.5        # How often to update status bar (seconds)


# ============================================================================
# RECORDING & DATA MANAGEMENT
# ============================================================================
# HDF5 recording pipeline configuration
MAX_WRITE_QUEUE_SIZE = 800          # Maximum frames buffered for writing
MAX_PROCESS_QUEUE_SIZE = 800        # Maximum frames buffered for processing
BATCH_SIZE = 20                     # Frames written per batch (balance I/O efficiency)
MAX_SUB_BATCH = 10                  # Sub-batch size for parallel processing
TARGET_CHUNK_MB = 3                 # Target HDF5 chunk size in megabytes
FLUSH_INTERVAL_SEC = 15.0           # How often to flush data to disk (seconds)
WRITE_TIMEOUT_SEC = 10.0            # Timeout for write operations (seconds)
FRAME_TIMEOUT_MS = 30               # Timeout waiting for frames (milliseconds)

# Compression settings
RECORDING_COMPRESSION_LEVEL = 0     # Compression during recording (0=none for speed)
POST_COMPRESSION_LEVEL = 4          # Compression after recording (4=good balance)
MIN_DISK_SPACE_MB = 50              # Minimum free disk space before warning (MB)


# ============================================================================
# GPU ACCELERATION (Optimized for i7-14700 with AMD Radeon Pro WX 3100)
# ============================================================================
GPU_BUFFER_POOL_SIZE = 8            # Number of GPU buffers in pool
GPU_BATCH_SIZE = 8                  # Frames processed per GPU batch


# ============================================================================
# HARDWARE CONNECTION & COMMUNICATION
# ============================================================================
# Connection retry and timeout settings
MAX_CONNECT_RETRIES = 3             # Maximum connection attempts before failure
CONNECT_RETRY_DELAY = 1.0           # Delay between retry attempts (seconds)
HEALTH_CHECK_INTERVAL = 5.0         # How often to check hardware health (seconds)
VISA_TIMEOUT_MS = 5000              # VISA communication timeout (milliseconds)


# ============================================================================
# HARDWARE VALIDATION LIMITS
# ============================================================================
# Function Generator limits (Siglent SDG1032X)
MIN_FREQUENCY_MHZ = 0.001           # Minimum frequency: 1 kHz
MAX_FREQUENCY_MHZ = 30.0            # Maximum frequency: 30 MHz
MIN_AMPLITUDE_VPP = 0.002           # Minimum amplitude: 2 mVpp
MAX_AMPLITUDE_VPP = 20.0            # Maximum amplitude: 20 Vpp

# Stage positioning limits (Mad City Labs Nano-Drive)
MIN_STAGE_POSITION_UM = 0           # Minimum stage position: 0 µm
MAX_STAGE_POSITION_UM = 25000       # Maximum stage position: 25 mm (25000 µm)


# ============================================================================
# FILE MANAGEMENT
# ============================================================================
# File extensions and backup settings
HDF5_EXTENSION = '.hdf5'            # HDF5 data file extension
LOG_EXTENSION = '.log'              # Log file extension
CONFIG_EXTENSION = '.json'          # Configuration file extension
MAX_BACKUP_FILES = 3                # Maximum number of backup files to keep
