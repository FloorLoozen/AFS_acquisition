"""
Camera Widget for AFS Acquisition - Minimal Design with Maximum Performance.

This module provides a streamlined camera interface with professional-grade
recording capabilities optimized for Windows 11 + Intel i7-14700.

Features:
- Real-time camera feed at 57 FPS with optional GPU-accelerated display
- High-performance HDF5 recording with real-time LZF compression
- Live view at 12 FPS (prevents lag) with 30 FPS recording
- Automatic error recovery with test pattern fallback
- Thread-safe frame processing and recording

Architecture:
- Main thread: Qt GUI and user interaction
- Camera thread: Frame capture at hardware FPS (57.2 FPS max)
- Recording: 30 FPS target with strict rate limiting
- Display: 12 FPS for lag-free UI updates

Performance Optimizations (2025-11-27):
- Real-time LZF compression during recording (eliminates post-processing wait)
- Frame buffer flushing after LUT acquisition (prevents black recordings)
- Automatic camera settings restoration (exposure=5ms, gain=2, fps=30)
- Removed double rate limiting (camera widget + recorder conflict)
- 300ms stabilization delay after settings changes
- Helper method _flush_camera_buffer() eliminates code duplication

Recording Workflow:
1. Flush stale frames from buffer
2. Apply recording settings (exposure, gain, FPS)
3. Wait 300ms for camera hardware stabilization
4. Flush transitional frames
5. Start recording with clean buffer state

Post-LUT Recovery:
- Automatic camera settings restoration via resume_live()
- Buffer flush to remove stale frames
- Camera capture restart for clean state
- Ensures recordings after LUT are never black

User Experience:
- Minimal, clean interface for maximum screen real estate
- Instant save (no compression wait)
- Live view continues during recording
- Recording status shows dual FPS: "Recording: 30 FPS | Live: 12 FPS"
- Automatic camera reconnection on errors
- Real-time FPS performance monitoring
"""

import time
import cv2
import os
from pathlib import Path
import numpy as np
import h5py
from datetime import datetime
from typing import Optional
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QSizePolicy, QMessageBox, QProgressDialog, QApplication
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, qRgb

from src.utils.logger import get_logger
from src.controllers.camera_controller import CameraController, FrameData
from src.utils.status_display import StatusDisplay
from src.utils.hdf5_video_recorder import HDF5VideoRecorder
from src.utils.config_manager import get_config

logger = get_logger("camera_widget")

# GPU Display Processing Configuration (OpenCL for AMD/NVIDIA compatibility)
_GPU_AVAILABLE = False
try:
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.useOpenCL():
            _GPU_AVAILABLE = True
            logger.info("Camera widget: GPU acceleration available for display processing (OpenCL)")
except Exception:
    _GPU_AVAILABLE = False


class CameraWidget(QGroupBox):
    """
    Simplified camera widget with clean design.
    Features:
    - Large camera display area
    - Basic control buttons only
    - No side controls panel
    - Minimal, clean interface
    """
    
    # Signals to communicate from background thread to main GUI thread
    compression_complete_signal = pyqtSignal(str, float, float, float)  # Emits (path, original_mb, compressed_mb, reduction_pct)
    compression_progress_signal = pyqtSignal(int, int, str)  # Emits (current, total, status_text)
    # Signal to receive a prepared QImage from worker thread for display
    display_image_signal = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__("Camera", parent)
        
        # Camera state
        self.camera: Optional[CameraController] = None
        self.is_running = False
        self.is_live = False
        self.camera_error = None
        self._is_reinitializing = False  # Track reinitialization state
        
        # Frame data
        self.current_frame_data: Optional[FrameData] = None
        
        # Recording state
        self.is_recording = False
        self.is_saving = False  # Track if we're in the middle of saving
        self.last_frame_timestamp = 0
        
        # Performance monitoring (for FPS display updates)
        self.display_fps_start_time = time.time()
        self.last_display_fps = 0
        self._last_display_time = 0
        
        # Recording objects
        self.hdf5_recorder = None
        self.recording_path = ""
        self.recording_start_time = None
        self.recorded_frames = 0
        
        # Compression progress tracking
        self.compression_progress_dialog = None
        
        # Executors: separate executors for display and recording to avoid starvation
        # Display executor: low-latency single worker for consistent UI updates
        self.display_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="CameraDisplay"  # i7-14700: 2 workers for smoother display
        )
        # Track whether a display task is currently inflight to avoid queue buildup
        self._display_task_inflight = False
        # Recording executor: optimized for i7-14700 (20 cores, 28 threads)
        self.recording_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="CameraRecorder"  # i7-14700: 4 workers for parallel processing
        )
        self.recording_queue = queue.Queue(maxsize=100)  # Buffer for recording (compatibility)
        self.processing_lock = threading.RLock()
        
        # Recording settings: FPS, compression, and resolution
        # compression_level: 0=none (FASTEST for real-time recording), 1-3=fast/LZF, 4-9=best/GZIP (SLOWEST)
        # downscale_factor: 1=full res, 2=half, 4=quarter
        # NOTE: Compression happens during recording (real-time compatible with LZF)
        # OPTIMIZED FOR 30 FPS: Level 1 (LZF fast) provides ~35% compression with minimal CPU overhead
        self.recording_settings = {
            'recording_fps': 30,      # Recording frame rate (configurable from settings)
            'compression_level': 1,   # LZF fast compression optimized for 30 FPS real-time recording
            'downscale_factor': 2     # HALF resolution = Good balance of speed & file size
        }
        
        # UI components - optimized for fullscreen viewing
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setMinimumSize(800, 600)  # Larger minimum for fullscreen
        self.display_label.setStyleSheet("border: none;")  # No border for clean look
        
        # Status display
        self.status_display = StatusDisplay()
        
        # Set widget size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Live display configuration: 12 FPS for smooth, lag-free display
        # Target: 12 FPS (83ms interval) - optimal for UI responsiveness
        self.live_display_fps = 12  # 12 FPS = ~83ms interval, smooth without lag
        self._last_display_time = 0  # Track last display update
        self._min_display_interval = 1.0 / 12.0  # Exact 83.33ms between frames
        try:
            cfg = get_config()
            self.live_display_fps = int(getattr(cfg.ui, 'live_display_fps', 12))
            self._min_display_interval = 1.0 / self.live_display_fps
        except Exception:
            pass
        
        # Fallback timer for display (uses configured `live_display_fps`)
        self.fallback_timer = QTimer(self)
        self.fallback_timer.timeout.connect(self._process_frame_immediate)
        self.fallback_timer.setInterval(int(1000.0 / self.live_display_fps))  # 33ms for 30 FPS
        
        # Pipeline diagnostics
        self._last_capture_ts = 0
        self._last_callback_ts = 0
        self._last_render_ts = 0
        
        self.init_ui()
        self.update_status("Initializing...")

        # Pending update flag to coalesce rapid frame callbacks
        self._pending_frame_update = False
        
        # Connect signals for compression notifications
        self.compression_complete_signal.connect(self._show_compression_complete)
        self.compression_progress_signal.connect(self._update_compression_progress)
        # Connect display image signal to the GUI slot
        self.display_image_signal.connect(self._on_display_image)
        
        # Auto-connect shortly after startup
        QTimer.singleShot(100, self.connect_camera)
    
    def init_ui(self):
        """Initialize the simplified UI layout with consistent styling."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Create camera frame for consistent styling
        camera_frame = self._create_camera_frame()
        main_layout.addWidget(camera_frame)

    def _create_camera_frame(self):
        """Create the main camera frame with consistent styling."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Add camera display section
        camera_section = self._create_camera_section()
        layout.addWidget(camera_section)
        
        return frame

    def _create_camera_section(self):
        """Create the camera display section."""
        section = QFrame()
        section.setFrameShape(QFrame.NoFrame)  # Inner content has no additional frame
        
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Add camera display (takes most of the space)
        self.display_label.setStyleSheet("border: none;")  # Clean display area
        layout.addWidget(self.display_label, 1)
        
        # Minimal status bar (only status display)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.status_display)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        return section
    
    def update_button_states(self):
        """Update button enabled states - minimal camera widget has no buttons."""
        pass  # No buttons to update in minimal design
    
    def update_status(self, text: str):
        """Update status display."""
        self.status_display.set_status(text)
    
    def _show_compression_complete(self, path: str, original_mb: float, compressed_mb: float, reduction_pct: float):
        """
        Show compression complete notification with statistics and sound alert.
        
        Args:
            path: Path to the compressed file
            original_mb: Original file size in MB
            compressed_mb: Compressed file size in MB
            reduction_pct: Percent change vs. original (negative indicates increase)
        """
        # Close progress dialog if it exists
        if self.compression_progress_dialog:
            self.compression_progress_dialog.close()
            self.compression_progress_dialog = None
        
        # Play system sound notification (Windows native method first)
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)  # Windows asterisk/info sound
        except (ImportError, Exception):
            # Fallback to QSound if winsound not available
            try:
                from PyQt5.QtMultimedia import QSound
                QSound.play("SystemAsterisk")  # Windows success sound
            except (ImportError, Exception):
                # Last resort - system bell
                try:
                    print('\a')  # Terminal bell
                except Exception:
                    pass  # No sound available
        
        # Format file sizes for display
        original_str = f"{original_mb:.1f} MB" if original_mb < 1024 else f"{original_mb/1024:.2f} GB"
        compressed_str = f"{compressed_mb:.1f} MB" if compressed_mb < 1024 else f"{compressed_mb/1024:.2f} GB"

        # Pick label based on whether compression actually reduced or increased the size
        change_label = "Reduction" if reduction_pct >= 0 else "Increase"
        change_value = abs(reduction_pct)
        
        # Create detailed completion message
        message = (
            f"Recording saved and compressed successfully!\n\n"
            f"File: {path}\n\n"
            f"Original size: {original_str}\n"
            f"Compressed size: {compressed_str}\n"
            f"{change_label}: {change_value:.1f}%\n\n"
            f"✓ Lossless compression - 100% data preserved"
        )
        # Additionally probe dataset-level storage to report actual on-disk compression
        try:
            # h5py and numpy already imported at module level
            ds_info_lines = []
            with h5py.File(path, 'r') as f:
                # Prefer main_video dataset under /raw_data
                if 'raw_data' in f and 'main_video' in f['raw_data']:
                    ds = f['raw_data']['main_video']
                    # Compute uncompressed bytes from dataset shape and dtype
                    try:
                        dtype_itemsize = np.dtype(ds.dtype).itemsize
                        uncompressed_bytes = int(np.prod(ds.shape) * dtype_itemsize)
                    except Exception:
                        uncompressed_bytes = None

                    try:
                        storage_bytes = int(ds.id.get_storage_size())
                    except Exception:
                        storage_bytes = None

                    comp = getattr(ds, 'compression', None)
                    comp_opts = getattr(ds, 'compression_opts', None)

                    if uncompressed_bytes and storage_bytes:
                        try:
                            pct = (1.0 - (storage_bytes / float(uncompressed_bytes))) * 100.0
                        except Exception:
                            pct = 0.0
                    else:
                        pct = None

                    if comp:
                        ds_info_lines.append(f"Dataset compression: {comp} (opts={comp_opts})")
                    if pct is not None:
                        ds_info_lines.append(f"On-disk reduction vs raw bytes: {pct:.1f}% (storage {storage_bytes/1024/1024:.1f} MB vs raw {uncompressed_bytes/1024/1024:.1f} MB)")
                    elif storage_bytes is not None:
                        ds_info_lines.append(f"Dataset storage: {storage_bytes/1024/1024:.1f} MB")

            if ds_info_lines:
                message += "\n" + "\n".join(ds_info_lines)
        except Exception:
            # If probing fails, don't block the user notification
            pass
        
        # Show completion notification with statistics
        QMessageBox.information(
            self, 
            "Compression Complete", 
            message,
            QMessageBox.Ok
        )
    
    def _update_compression_progress(self, current: int, total: int, status_text: str):
        """Update compression progress dialog (slot for compression_progress_signal)."""
        if not self.compression_progress_dialog:
            # Create progress dialog
            self.compression_progress_dialog = QProgressDialog(
                "Compressing recording...",
                "Continue in Background",
                0,
                total,
                self
            )
            self.compression_progress_dialog.setWindowTitle("Post-Processing Compression")
            self.compression_progress_dialog.setMinimumDuration(0)  # Show immediately
            self.compression_progress_dialog.setModal(False)  # Non-blocking
            self.compression_progress_dialog.setAutoClose(False)  # Don't auto-close
            self.compression_progress_dialog.setAutoReset(False)  # Don't auto-reset
            
            # When user clicks "Continue in Background", just close the dialog
            def on_canceled():
                if self.compression_progress_dialog:
                    self.compression_progress_dialog.close()
                    self.compression_progress_dialog = None
                    logger.info("Compression continuing in background (progress dialog closed by user)")
            
            self.compression_progress_dialog.canceled.connect(on_canceled)
        
        # Update progress
        self.compression_progress_dialog.setValue(current)
        self.compression_progress_dialog.setMaximum(total)
        self.compression_progress_dialog.setLabelText(f"{status_text}\n{current}/{total} frames processed")
    
    def connect_camera(self, camera_id: int = 0):
        """Connect to camera with improved error handling."""
        if self.is_running:
            return
        
        self.update_status("Initializing...")
        
        try:
            # Use the advanced camera controller from controllers directory
            # Single frame queue for absolute minimum latency (<50ms buffering)
            self.camera = CameraController(camera_id=camera_id, max_queue_size=1)
            
            if self.camera.initialize():
                # Apply camera settings BEFORE starting capture to avoid FPS mismatch
                self._apply_default_camera_settings()
                
                if self.camera.start_capture():
                    self.is_running = True
                    self.is_live = False
                    
                    self._is_reinitializing = False  # Clear reinitializing flag on success
                    self.update_status("Connected")
                    self.update_button_states()
                    
                    # Start timer for frame updates
                    self.fallback_timer.start()

                    self.set_live_mode()  # Auto start live view
                    
                    return
                else:
                    logger.warning("Failed to start camera capture")
                    if self.camera:
                        self.camera.close()
                        self.camera = None
            else:
                logger.warning("Camera hardware initialization failed")
                if self.camera:
                    self.camera.close()
                    self.camera = None
            
            # Fall through to test pattern mode
            self._start_test_pattern_mode("Test Pattern Mode")
            
        except Exception as e:
            logger.warning(f"Camera initialization error: {e}")
            if self.camera:
                try:
                    self.camera.close()
                except Exception as close_error:
                    logger.debug(f"Error closing camera during error recovery: {close_error}")
                self.camera = None
            self._start_test_pattern_mode("Error - Test Pattern")
    
    def _start_test_pattern_mode(self, status_text: str):
        """Start test pattern mode."""
        try:
            self.camera = CameraController(camera_id=0, max_queue_size=1)
            if self.camera.initialize() and self.camera.start_capture():
                self.is_running = True
                self.is_live = False
                
                self._is_reinitializing = False  # Clear reinitializing flag
                self.update_status(status_text)
                self.update_button_states()
                # Use ONLY timer for consistent FPS
                self.fallback_timer.start()
                self.set_live_mode()
            else:
                self.camera = None
                self.is_running = False
                self._is_reinitializing = False  # Clear reinitializing flag
                self.update_status("Camera Error")
                self.update_button_states()
        except Exception as e:
            logger.error(f"Failed to start test pattern mode: {e}")
            self.camera = None
            self._is_reinitializing = False  # Clear reinitializing flag
            self.update_status("Camera Error")
            self.is_running = False
            self.update_status("Camera Error")
            self.update_button_states()
    
    def reconnect_camera(self):
        """Reconnect camera."""
        self._is_reinitializing = True
        self.update_status("Reinitializing...")
        self.stop_camera()
        QTimer.singleShot(500, self.connect_camera)
    
    def stop_camera(self):
        """Stop camera operations."""
        if hasattr(self, 'fallback_timer') and self.fallback_timer.isActive():
            self.fallback_timer.stop()
        
        self.is_running = False
        self.is_live = False
        
        if self.camera:
            try:
                self.camera.close()
            except Exception as e:
                logger.debug(f"Error closing camera: {e}")
            self.camera = None
        
        self.current_frame_data = None
        
        # Only set to "Disconnected" if we're not reinitializing
        if not self._is_reinitializing:
            self.update_status("Disconnected")
        # If reinitializing, maintain the "Reinitializing..." status
        
        self.update_button_states()
    
    def set_live_mode(self):
        """Start live view."""
        if not self.is_running or not self.camera:
            return
        
        self.is_live = True
        
        # Check if using test pattern
        if hasattr(self.camera, 'get_statistics'):
            stats = self.camera.get_statistics()
            if stats.get('use_test_pattern', False):
                self.update_status("Test Pattern Active")
            else:
                self.update_status("Live")
        else:
            self.update_status("Live")
        
        self.update_button_states()
    
    def pause_live(self):
        """Pause live view to save processing power (e.g., during LUT acquisition)."""
        if self.is_live:
            self.is_live = False
            self.update_status("Paused (background task)")
            logger.info("Live view paused to save processing power")
    
    def _flush_camera_buffer(self):
        """Flush stale frames from camera buffer.
        
        Returns:
            Number of frames flushed
        """
        if not self.camera or not hasattr(self.camera, 'frame_queue'):
            return 0
        
        import queue
        flushed = 0
        while not self.camera.frame_queue.empty():
            try:
                self.camera.frame_queue.get_nowait()
                flushed += 1
            except queue.Empty:
                break
        return flushed
    
    def resume_live(self):
        """Resume live view after pausing (e.g., after LUT acquisition)."""
        if not self.is_live and self.is_running:
            self.is_live = True
            self.update_status("Live")
            logger.info("Live view resumed")
            
            try:
                # Flush stale frames and restore camera settings
                flushed = self._flush_camera_buffer()
                if flushed > 0:
                    logger.info(f"Flushed {flushed} stale frames from camera buffer")
                
                self._apply_default_camera_settings()
                logger.info("Camera settings restored after resume")
                
                # Brief wait for camera stabilization
                import time
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed to restore camera settings on resume: {e}")

    def _camera_frame_arrived(self):
        """Called from camera capture thread when a new frame is queued.
        
        Schedule immediate GUI update with coalescing to prevent flooding.
        """
        if not self._pending_frame_update:
            self._pending_frame_update = True
            QTimer.singleShot(0, self._process_frame_immediate)
    
    def _process_frame_immediate(self):
        """Process and display frame immediately (runs in GUI thread)."""
        self._pending_frame_update = False
        
        if not self.is_running or not self.is_live or not self.camera:
            return
        
        current_time = time.time()
        
        # Read directly from camera buffer for lowest latency (bypass queue)
        if hasattr(self.camera, 'read_buffer_direct'):
            frame = self.camera.read_buffer_direct()
            if frame is not None:
                # Create minimal frame data for display
                frame_data = type('obj', (object,), {
                    'frame': frame,
                    'timestamp': current_time
                })()
            else:
                # Fallback to queue if direct read fails
                frame_data = self.camera.get_latest_frame(timeout=0.01)
        else:
            # Fallback for test pattern or older code
            frame_data = self.camera.get_latest_frame(timeout=0.01)
            
        if frame_data is None:
            return
        
        # Recording: capture frames for recording (HDF5 recorder handles FPS rate limiting)
        # This is completely decoupled from display rate
        if self.is_recording and self.hdf5_recorder:
            # Only record if we're in high-speed timer mode (20ms interval)
            # This prevents overload when timer is at display rate
            if self.fallback_timer.interval() <= 30:  # High-speed mode (20ms = 50Hz)
                # Validate frame is not all-black or all-gray (bad frame detection)
                if not (frame_data.frame.max() == 0 or (frame_data.frame.min() == frame_data.frame.max())):
                    try:
                        # Zero-copy optimization: only copy if frame doesn't own its data
                        if frame_data.frame.flags.owndata:
                            # Frame owns data - safe to pass directly (zero-copy)
                            frame_to_record = frame_data.frame
                        else:
                            # Frame is a view - need copy for safety
                            frame_to_record = np.array(frame_data.frame, copy=True)
                        
                        if hasattr(self.hdf5_recorder, 'enqueue_frame'):
                            accepted = self.hdf5_recorder.enqueue_frame(frame_to_record)
                            if accepted:
                                # Track accepted frames for FPS display
                                if hasattr(self, 'recording_fps_tracker'):
                                    self.recording_fps_tracker.setdefault('frame_times', []).append(current_time)
                                    cutoff = current_time - 5.0
                                    self.recording_fps_tracker['frame_times'] = [t for t in self.recording_fps_tracker['frame_times'] if t > cutoff]
                                with self.processing_lock:
                                    self.recorded_frames += 1
                    except Exception:
                        pass
        
        # Display: rate limiting to configured live_display_fps (ALWAYS update display)
        time_since_last = current_time - self._last_display_time
        if time_since_last < self._min_display_interval and self._last_display_time > 0:
            return  # Skip display update - too soon

        # Update display time and use direct fast rendering for minimal latency
        self._last_display_time = current_time
        self.last_frame_timestamp = frame_data.timestamp
        self.current_frame_data = frame_data

        # Direct display update (no background thread) for minimal lag at 12 FPS
        self._fast_update_display(frame_data.frame)
        
        # Track display FPS separately
        if not hasattr(self, '_display_frame_times'):
            self._display_frame_times = []
        self._display_frame_times.append(current_time)
        # Keep last 2 seconds of display times
        self._display_frame_times = [t for t in self._display_frame_times if current_time - t < 2.0]
        
        # Update status every 0.5s
        time_elapsed = current_time - self.display_fps_start_time
        if time_elapsed >= 0.5:
            # Calculate ACTUAL display FPS from display timestamps
            if len(self._display_frame_times) >= 2:
                display_timespan = self._display_frame_times[-1] - self._display_frame_times[0]
                display_fps = (len(self._display_frame_times) - 1) / display_timespan if display_timespan > 0 else 0.0
            else:
                display_fps = 0.0
            
            # If recording, also show recording FPS (targeting 30 FPS)
            if self.is_recording and hasattr(self, 'recording_fps_tracker'):
                frame_times = self.recording_fps_tracker.get('frame_times', [])
                if len(frame_times) >= 2:
                    time_span = frame_times[-1] - frame_times[0]
                    recording_fps = (len(frame_times) - 1) / time_span if time_span > 0 else 0.0
                    status_text = f"Recording: {recording_fps:.1f} FPS | Live: {display_fps:.1f} FPS"
                else:
                    status_text = f"Live: {display_fps:.1f} FPS"
            else:
                status_text = f"Live: {display_fps:.1f} FPS"
            
            # Add latency
            total_latency = (current_time - frame_data.timestamp) * 1000
            self.update_status(f"{status_text} | {total_latency:.0f}ms")
            
            self.display_fps_start_time = current_time
    
    def set_pause_mode(self):
        """Pause live view."""
        if not self.is_running:
            return
        
        self.is_live = False
        self.update_status("Paused")
        self.update_button_states()
    
    def update_frame(self):
        """Legacy method - now handled by direct callbacks."""
        pass
    
    def display_frame(self, frame):
        """Legacy display method - now delegates to parallel processing."""
        self._process_display_frame(frame.copy())
    
    def _apply_image_processing(self, frame):
        """Apply image processing to frame (currently no processing - returns frame as-is)."""
        # Image processing (brightness/contrast/saturation) has been removed
        # This method is kept for potential future use
        return frame
    
    # Recording methods (preserved from original)
    def start_recording(self, file_path: str = None, metadata=None) -> bool:
        """Start HDF5 video recording.
        
        Uses session HDF5 file from main window if file_path not provided.
        
        Args:
            file_path: Optional path to HDF5 file (uses session file if None)
            metadata: Optional metadata dictionary
            
        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False
        
        if not self.is_running or not self.is_live:
            logger.warning("Cannot start recording - camera not in live mode")
            return False
        
        try:
            # Get session file from main window if no path provided
            main_window = None
            if file_path is None:
                try:
                    app = QApplication.instance()
                    main_windows = [w for w in app.topLevelWidgets() if hasattr(w, 'get_session_hdf5_file')]
                    if main_windows:
                        main_window = main_windows[0]
                        file_path = main_window.get_session_hdf5_file()
                        if file_path:
                            logger.info(f"Using session HDF5 file: {file_path}")
                        else:
                            logger.warning("Session file is None - will create new file")
                        
                        # Check if recording already exists in session - ask user what to do
                        if file_path and main_window.session_has_recording:
                            from PyQt5.QtWidgets import QMessageBox
                            msg_box = QMessageBox(None)
                            msg_box.setIcon(QMessageBox.Question)
                            msg_box.setWindowTitle("Recording Already Exists")
                            msg_box.setText("The current session already contains a video recording.")
                            msg_box.setInformativeText("What would you like to do?")
                            
                            overwrite_btn = msg_box.addButton("Overwrite", QMessageBox.AcceptRole)
                            new_session_btn = msg_box.addButton("New Session", QMessageBox.DestructiveRole)
                            cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
                            msg_box.setDefaultButton(cancel_btn)
                            
                            msg_box.exec_()
                            clicked = msg_box.clickedButton()
                            
                            if clicked == cancel_btn:
                                logger.info("Recording cancelled by user")
                                return False
                            elif clicked == new_session_btn:
                                # Ask main window to create new session
                                if hasattr(main_window, '_new_session_file'):
                                    main_window._new_session_file()
                                    # Get the new session file
                                    file_path = main_window.get_session_hdf5_file()
                                    logger.info(f"Using new session file: {file_path}")
                            # else: overwrite_btn clicked = Overwrite (continue with current session)
                        
                        # Mark main window that session will have recording (only if we have a file_path)
                        if file_path:
                            main_window.mark_session_has_recording()
                    else:
                        logger.warning("No main window found with get_session_hdf5_file method")
                except Exception as e:
                    logger.error(f"Error getting session file: {e}", exc_info=True)
            
            # Fallback: create new file if still no path
            if not file_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.hdf5"
                from src.utils.config_manager import get_config
                try:
                    cfg = get_config()
                    default_dir = Path(cfg.files.data_directory)
                except Exception:
                    default_dir = Path.home() / 'Documents' / 'AFS_Data'
                
                default_dir.mkdir(parents=True, exist_ok=True)
                file_path = str(default_dir / filename)
                logger.info(f"Created new recording file: {file_path}")
            
            # Ensure .hdf5 extension
            if not file_path.lower().endswith('.hdf5'):
                file_path = file_path + '.hdf5'
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Check if file exists and has a recording session WITH ACTUAL VIDEO DATA
            if os.path.exists(file_path):
                # h5py already imported at module level
                try:
                    with h5py.File(file_path, 'r') as hf:
                        # Check if there's already a recording session with video frames
                        has_video_recording = False
                        if 'raw_data' in hf and 'recordings' in hf['raw_data']:
                            recordings_group = hf['raw_data']['recordings']
                            # Check each recording session for actual video data
                            for session_name in recordings_group.keys():
                                session = recordings_group[session_name]
                                # Check if 'frames' dataset exists AND has 'total_frames' attribute > 0
                                # (LUT creates empty frames dataset but doesn't set total_frames)
                                if 'frames' in session:
                                    # Check total_frames attribute (only set after real recording)
                                    total_frames = session['frames'].attrs.get('total_frames', 0)
                                    if total_frames > 0:
                                        has_video_recording = True
                                        logger.info(f"Found existing video recording: {session_name} with {total_frames} frames")
                                        break
                        
                        if has_video_recording:
                            # File already has a VIDEO recording (not just LUT), prompt user
                            from PyQt5.QtWidgets import QMessageBox
                            reply = QMessageBox.warning(
                                self,
                                "Recording Exists",
                                f"This file already contains a video recording.\\n\\n"
                                f"Starting a new recording will create a NEW file.\\n\\n"
                                f"Do you want to create a new file for this recording?",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No
                            )
                            
                            if reply == QMessageBox.No:
                                logger.info("User cancelled second recording")
                                return False
                            
                            # Ask if user wants to copy LUT and resonance data
                            copy_reply = QMessageBox.question(
                                self,
                                "Copy Data",
                                f"Do you want to copy LUT and resonance sweep data\\n"
                                f"to the new recording file?",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes
                            )
                            
                            copy_data = (copy_reply == QMessageBox.Yes)
                            
                            # Generate new filename with timestamp
                            # Use module-level imports (datetime and Path already imported at top)
                            original_path = Path(file_path)
                            old_file_path = file_path
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            new_filename = f"{original_path.stem}_{timestamp}{original_path.suffix}"
                            file_path = str(original_path.parent / new_filename)
                            logger.info(f"Creating new file for second recording: {file_path}")
                            
                            # Copy LUT and resonance data if requested
                            if copy_data:
                                try:
                                    with h5py.File(old_file_path, 'r') as src:
                                        with h5py.File(file_path, 'w') as dst:
                                            # Copy LUT data if exists
                                            if 'raw_data' in src and 'LUT' in src['raw_data']:
                                                dst.create_group('raw_data')
                                                src.copy('raw_data/LUT', dst['raw_data'])
                                                logger.info("Copied LUT data to new file")
                                            
                                            # Copy execution log (resonance sweeps) if exists
                                            if 'execution_log' in src:
                                                src.copy('execution_log', dst)
                                                logger.info("Copied resonance sweep data to new file")
                                    
                                    QMessageBox.information(
                                        self,
                                        "Data Copied",
                                        f"New recording file created with LUT and resonance data:\\n{file_path}"
                                    )
                                except Exception as copy_error:
                                    logger.error(f"Failed to copy data: {copy_error}")
                                    QMessageBox.warning(
                                        self,
                                        "Copy Failed",
                                        f"Could not copy data to new file.\\nNew file: {file_path}\\nError: {copy_error}"
                                    )
                            else:
                                QMessageBox.information(
                                    self,
                                    "New File Created",
                                    f"New recording file (without previous data):\\n{file_path}"
                                )
                except Exception as e:
                    logger.warning(f"Could not check existing file: {e}")
            
            # Get frame shape from camera (MONO8 = 1 channel, not 3!)
            if self.camera:
                frame_shape = self.camera.frame_shape
            else:
                frame_shape = (480, 640, 1)  # Default to grayscale
            
            # Create recorder with compression and resolution settings
            # Use configurable recording FPS from settings
            TARGET_FPS = float(self.recording_settings['recording_fps'])
            MIN_FPS = TARGET_FPS * 0.83  # Warning threshold at 83% of target
            
            # Increase camera queue size during recording to prevent frame drops at high FPS
            if self.camera:
                try:
                    # Larger queue for high-speed recording to absorb I/O jitter
                    self.camera.max_queue_size = 10  # ~330ms buffer at 30 FPS
                    logger.info(f"Increased camera queue to {self.camera.max_queue_size} for recording")
                except Exception as e:
                    logger.warning(f"Could not increase queue size: {e}")
            
            # Clear last_lut_file to ensure each recording gets a new file
            # (LUT file is only reused for the FIRST recording after LUT acquisition)
            if hasattr(self, 'last_lut_file'):
                self.last_lut_file = None

            self.hdf5_recorder = HDF5VideoRecorder(
                file_path=file_path,
                frame_shape=frame_shape,
                fps=TARGET_FPS,
                min_fps=MIN_FPS,
                compression_level=self.recording_settings['compression_level'],
                downscale_factor=self.recording_settings['downscale_factor'],
                use_gpu=False  # Force CPU path for recording to avoid GPU transfer overhead and latency
            )
            
            # Register monitoring callbacks
            self.hdf5_recorder.set_frame_drop_callback(self._on_frame_drop_alert)
            self.hdf5_recorder.set_health_monitor_callback(self._on_health_alert)
            
            # Build info message based on actual settings
            resolution_desc = {1: "full resolution", 2: "half resolution", 4: "quarter resolution"}
            res_text = resolution_desc.get(self.recording_settings['downscale_factor'], f"{self.recording_settings['downscale_factor']}x downscale")
            logger.info(f"Recording configured: {TARGET_FPS} FPS (balanced speed), {res_text} + MONO8 for reliable capture")
            
            # Track FPS performance during recording
            self.recording_fps_tracker = {
                'frame_times': [],
                'warnings_issued': 0,
                'below_min_count': 0,
                'last_record_time': 0  # Track last recorded frame time for rate limiting
            }
            
            # Prepare metadata
            recording_metadata = {
                'operator': os.getenv('USERNAME', 'Unknown'),
                'system_name': 'AFS_tracking'
            }
            if metadata:
                recording_metadata.update(metadata)
            
            if not self.hdf5_recorder.start_recording(recording_metadata):
                logger.error("Failed to start HDF5 recording")
                return False
            
            # Save camera settings
            if self.camera and hasattr(self.camera, 'get_camera_settings'):
                try:
                    camera_settings = self.camera.get_camera_settings()
                    self.hdf5_recorder.add_camera_settings(camera_settings)
                except Exception as e:
                    logger.warning(f"Failed to save camera settings: {e}")
            
            # Add stage settings if available
            try:
                from src.controllers.stage_manager import StageManager
                stage_manager = StageManager.get_instance()
                if stage_manager:
                    stage_settings = stage_manager.get_stage_settings()
                    self.hdf5_recorder.add_stage_settings(stage_settings)
            except Exception as e:
                logger.warning(f"Failed to save stage settings: {e}")
            
            # Add function generator info if available
            try:
                if hasattr(self.parent(), 'function_generator_controller'):
                    fg_controller = self.parent().function_generator_controller
                    self.hdf5_recorder.add_function_generator_info(fg_controller)
            except Exception as e:
                logger.warning(f"Failed to save function generator info: {e}")
            
            # Add oscilloscope info if available
            try:
                if hasattr(self.parent(), 'oscilloscope_controller'):
                    osc_controller = self.parent().oscilloscope_controller
                    self.hdf5_recorder.add_oscilloscope_info(osc_controller)
            except Exception as e:
                logger.warning(f"Failed to save oscilloscope info: {e}")
            
            # Add recording metadata and regeneration info
            try:
                self.hdf5_recorder.add_recording_metadata(recording_metadata)
            except Exception as e:
                logger.warning(f"Failed to save recording metadata: {e}")
            
            # Register fast frame callback for recording at full camera speed (57 FPS)
            # This is independent of the display timer (15 FPS) for maximum recording throughput
            # DISABLED: Conflicts with timer-based recording, causing display to stop
            # if self.camera:
            #     try:
            #         self.camera.register_frame_callback(self._on_camera_frame_for_recording)
            #         logger.info("Registered high-speed recording callback (runs at camera FPS, not display FPS)")
            #     except Exception as e:
            #         logger.warning(f"Failed to register recording callback: {e}")
            
            # Speed up timer during recording to poll camera faster (50 Hz = 20ms interval)
            # This ensures we capture at full 30 FPS for recording AND maintain live view
            try:
                self.fallback_timer.setInterval(20)  # 20ms = 50 Hz polling for 30 FPS recording
                logger.info("Increased timer frequency to 50Hz for recording (target 30 FPS)")
            except Exception as e:
                logger.warning(f"Failed to increase timer frequency: {e}")
            
            # Apply recording-optimized camera settings and flush stale frames
            try:
                if self.camera and hasattr(self.camera, 'apply_settings'):
                    # Flush stale frames before applying new settings
                    flushed = self._flush_camera_buffer()
                    if flushed > 0:
                        logger.info(f"Flushed {flushed} stale frames before recording")
                    
                    # Apply recording-optimized settings
                    recording_camera_settings = {
                        'exposure_ms': 5.0,  # 5ms exposure for good brightness
                        'gain_master': 2,     # Gain 2 for balanced image
                        'fps': TARGET_FPS     # 30 FPS for recording
                    }
                    res = self.camera.apply_settings(recording_camera_settings)
                    logger.info(f"Applied recording camera settings: exposure=5ms, gain=2, fps={TARGET_FPS}")
                    
                    # Wait for camera hardware to stabilize with new settings
                    import time
                    time.sleep(0.3)
                    
                    # Flush transitional frames captured during settings change
                    flushed = self._flush_camera_buffer()
                    if flushed > 0:
                        logger.info(f"Flushed {flushed} transitional frames after settings change")
                    
                    if not res.get('fps', False):
                        logger.warning(f"Camera refused to set {TARGET_FPS} FPS")
            except Exception as e:
                logger.warning(f"Could not set camera settings for recording: {e}")

            # Set recording state
            self.is_recording = True
            self.recording_path = file_path
            self.recording_start_time = datetime.now()
            self.recorded_frames = 0
            
            # Log initial function generator state to timeline
            try:
                # Get main window reference to access measurement controls
                # QApplication already imported at top of file
                main_window = None
                for widget in QApplication.topLevelWidgets():
                    if widget.__class__.__name__ == 'MainWindow':
                        main_window = widget
                        break
                
                if main_window and hasattr(main_window, 'measurement_controls_widget'):
                    fg_status = main_window.measurement_controls_widget.get_function_generator_status()
                    self.log_initial_function_generator_state(
                        fg_status['frequency_mhz'],
                        fg_status['amplitude_vpp'],
                        fg_status['enabled']
                    )
                    logger.info(f"Logged initial FG state: {fg_status['frequency_mhz']:.1f} MHz, {fg_status['amplitude_vpp']:.1f} Vpp, enabled={fg_status['enabled']}")
            except Exception as e:
                logger.warning(f"Could not log initial function generator state: {e}")
            
            logger.info(f"Started HDF5 recording: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            if self.hdf5_recorder:
                try:
                    self.hdf5_recorder.stop_recording()
                except Exception as stop_error:
                    logger.debug(f"Error stopping recorder during error recovery: {stop_error}")
                self.hdf5_recorder = None
            return False
    
    def stop_recording(self) -> Optional[str]:
        """Stop HDF5 recording with post-processing compression and robust error handling."""
        if not self.is_recording:
            return None
        
        # Unregister camera callback first (if it was registered)
        # DISABLED: Callback not used anymore
        # if self.camera:
        #     try:
        #         self.camera.register_frame_callback(None)
        #         logger.info("Unregistered high-speed recording callback")
        #     except Exception:
        #         pass
        
        # Restore timer to normal display rate (12 FPS = 83ms)
        try:
            self.fallback_timer.setInterval(int(1000.0 / self.live_display_fps))
            logger.info(f"Restored timer to {self.live_display_fps} FPS for live view")
        except Exception:
            pass
        
        # Store paths and set flags immediately
        saved_path = self.recording_path
        self.is_recording = False
        self.is_saving = True
        
        # Calculate actual FPS performance
        if hasattr(self, 'recording_fps_tracker') and self.recording_fps_tracker['frame_times']:
            frame_times = self.recording_fps_tracker['frame_times']
            if len(frame_times) > 1:
                avg_fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                min_fps_achieved = 1.0 / max([frame_times[i+1] - frame_times[i] for i in range(len(frame_times)-1)])
                logger.debug(f"Recording FPS Performance: Avg={avg_fps:.1f}, Min={min_fps_achieved:.1f}")
                
                if avg_fps < 25.0:
                    logger.warning(f"WARNING: Recording below target 25 FPS (achieved {avg_fps:.1f})")
                elif min_fps_achieved < 20.0:
                    logger.warning(f"WARNING: Some frames below 20 FPS (lowest {min_fps_achieved:.1f})")
                else:
                    logger.debug(f"Recording excellent at {avg_fps:.1f} FPS!")
        
        
        try:
            if self.hdf5_recorder:
                # Show simple message box (non-blocking)
                from PyQt5.QtWidgets import QApplication, QMessageBox
                from PyQt5.QtCore import QTimer
                
                # Create a simple message box
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setWindowTitle("Saving Recording")
                msg_box.setText("Saving recording...")
                msg_box.setInformativeText("Please wait while the recording is being saved and compressed.")
                msg_box.setStandardButtons(QMessageBox.NoButton)  # No buttons
                msg_box.setModal(False)  # Non-modal so it doesn't block
                msg_box.show()
                
                # Process events to show dialog
                QApplication.processEvents()
                
                # Call stop_recording with periodic event processing
                
                success = False
                error_msg = None
                
                try:
                    # Create a timer to process events during save
                    def process_events_callback():
                        QApplication.processEvents()
                    
                    # Set up timer to process events every 100ms during save
                    timer = QTimer()
                    timer.timeout.connect(process_events_callback)
                    timer.start(100)  # Every 100ms
                    
                    try:
                        self.hdf5_recorder.stop_recording()
                        success = True
                        
                        # Stop timer before starting background compression
                        timer.stop()
                        timer.deleteLater()
                        
                        # IMPORTANT: Wait for file to be fully closed and flushed
                        import time
                        time.sleep(0.5)  # Give HDF5 time to close file properly
                        
                        # Update message box
                        msg_box.setInformativeText("Recording saved. Starting background compression...")
                        msg_box.repaint()
                        QApplication.processEvents()
                        
                        # POST-PROCESSING COMPRESSION DISABLED
                        # Recording now uses LZF compression (level 3) during capture
                        # This is faster and more effective than post-processing GZIP
                        
                        # import threading
                        # from src.utils.hdf5_video_recorder import post_process_compress_hdf5
                        # cfg = get_config()

                        def compress_and_notify(path: str):
                            # DISABLED: No longer needed with real-time LZF compression
                            if False:  # Dead code branch
                                time.sleep(1.0)
                                logger.info("Starting post-processing compression...")
                                
                                # NO UI callbacks - this runs in background after widget may be deleted
                                # Just log to console instead
                                
                                # First try lossless recompression (safe)
                                result = post_process_compress_hdf5(
                                    path,
                                    quality_reduction=False,  # Lossless: preserve 100% of scientific data
                                    parallel=True,
                                    progress_callback=None  # No callbacks to avoid deleted widget errors
                                )

                                if result:
                                    logger.info(f"Compression completed: {result['path']} (reduction {result.get('reduction_pct'):.1f}%)")
                                    # If little or no reduction, attempt aggressive lossy recompression
                                    try:
                                        reduction = float(result.get('reduction_pct', 0.0))
                                    except Exception:
                                        reduction = 0.0

                                    # Store initial result and decide whether to emit immediately
                                    initial_result = result

                                    emitted = False

                                    # If reduction was small (<2%), try aggressive pass with quality reduction
                                    if reduction < 2.0:
                                        try:
                                            import shutil
                                            # Check available free space for temp-file mode (needs ~1.5x file)
                                            total, used, free = shutil.disk_usage(os.path.dirname(path) or '.')
                                            free_mb = free / (1024**2)
                                            original_mb = result.get('original_mb', 0.0)
                                            required_mb = original_mb * 1.5

                                            if free_mb >= required_mb:
                                                logger.info("Little gain from lossless compression; running aggressive temp-file compression (lossy) to reduce size further")
                                                aggressive = post_process_compress_hdf5(
                                                    path,
                                                    quality_reduction=True,
                                                    parallel=True,
                                                    progress_callback=None,  # No callbacks
                                                    in_place=False
                                                )
                                            else:
                                                logger.info("Insufficient free space for temp-file compression; attempting in-place lossy recompression")
                                                aggressive = post_process_compress_hdf5(
                                                    path,
                                                    quality_reduction=True,
                                                    parallel=True,
                                                    progress_callback=None,  # No callbacks
                                                    in_place=True
                                                )

                                            if aggressive:
                                                logger.info(f"Aggressive compression completed: {aggressive['path']} (reduction {aggressive.get('reduction_pct'):.1f}%)")
                                                # Emit only final aggressive result (avoid duplicate popups)
                                                self.compression_complete_signal.emit(
                                                    aggressive['path'],
                                                    aggressive['original_mb'],
                                                    aggressive['compressed_mb'],
                                                    aggressive['reduction_pct']
                                                )
                                                emitted = True
                                        except Exception as ag_err:
                                            logger.warning(f"Aggressive compression attempt failed: {ag_err}")

                                    # If we didn't run or didn't get a successful aggressive result, emit the initial lossless result
                                    if not emitted:
                                        self.compression_complete_signal.emit(
                                            initial_result['path'],
                                            initial_result['original_mb'],
                                            initial_result['compressed_mb'],
                                            initial_result['reduction_pct']
                                        )


                        # POST-PROCESSING COMPRESSION DISABLED
                        # Recording now uses fast LZF compression (level 3) during capture
                        # This is real-time compatible and more effective than post-processing GZIP
                        try:
                            # Close the initial saving message box
                            msg_box.close()
                            msg_box.deleteLater()
                            QApplication.processEvents()
                            logger.info("Recording complete - LZF compression applied during capture")
                        except Exception as comp_ctrl_err:
                            logger.error(f"Error controlling compression behavior: {comp_ctrl_err}", exc_info=True)
                        
                    except Exception as stop_error:
                        # Stop timer on error
                        timer.stop()
                        timer.deleteLater()
                        raise stop_error
                        
                except Exception as e:
                    logger.error(f"Error stopping HDF5 recorder: {e}", exc_info=True)
                    success = False
                    error_msg = str(e)
                    
                    # Close message box on error
                    try:
                        msg_box.close()
                        msg_box.deleteLater()
                        QApplication.processEvents()
                    except RuntimeError as qt_error:
                        # Qt object may already be deleted
                        logger.debug(f"Qt object already deleted: {qt_error}")
                
                # Show error if failed
                if not success and error_msg:
                    QMessageBox.critical(
                        self,
                        "Save Error",
                        f"Failed to save recording:\n\n{error_msg}\n\nCheck the logs for details."
                    )
                    self.is_saving = False
                    return None
                
                self.hdf5_recorder = None
            
            # Log completion
            if self.recording_start_time:
                duration = datetime.now() - self.recording_start_time
                # Recording completed
            
            # Restore live camera FPS target (best-effort)
            try:
                if self.camera and hasattr(self.camera, 'apply_settings'):
                    self.camera.apply_settings({'fps': 30.0})
                    # Restore small queue for live view (minimal lag)
                    self.camera.max_queue_size = 1
                    logger.info("Restored camera to live view settings (30 FPS, queue=1)")
            except Exception:
                pass

            # Reset state
            self.recording_path = ""
            self.recording_start_time = None
            self.recorded_frames = 0
            self.is_saving = False
            
            return saved_path
            
        except Exception as e:
            logger.error(f"Critical error in stop_recording: {e}", exc_info=True)
            self.is_saving = False
            
            from PyQt5.QtWidgets import QMessageBox
            try:
                QMessageBox.critical(
                    self,
                    "Recording Stop Error",
                    f"A critical error occurred:\n\n{e}\n\nCheck the logs for details."
                )
            except RuntimeError as qt_error:
                # Widget may be deleted, log instead
                logger.error(f"Could not show error dialog (widget deleted): {qt_error}")
            
            return None
    
    def _on_camera_frame_for_recording(self):
        """Fast callback triggered by camera at full speed (57 FPS) for recording.
        
        This runs independently of the display timer (15 FPS), allowing recording
        to capture at maximum camera speed while display updates remain smooth.
        """
        if not self.is_recording or not self.hdf5_recorder or not self.camera:
            return
        
        try:
            # Get latest frame with minimal timeout
            frame_data = self.camera.get_latest_frame(timeout=0.005)
            if frame_data is None:
                return
            
            # Validate frame is not all-black or all-gray (bad frame detection)
            frame = frame_data.frame
            if frame.max() == 0 or (frame.min() == frame.max()):
                # Skip invalid frames (all same value = bad camera data)
                return
            
            current_time = time.time()
            
            # Record frame at full camera speed
            # Zero-copy optimization
            if frame.flags.owndata:
                frame_to_record = frame
            else:
                frame_to_record = np.array(frame, copy=True)
            
            if hasattr(self.hdf5_recorder, 'enqueue_frame'):
                accepted = self.hdf5_recorder.enqueue_frame(frame_to_record)
                if accepted:
                    if hasattr(self, 'recording_fps_tracker'):
                        self.recording_fps_tracker.setdefault('frame_times', []).append(current_time)
                        cutoff = current_time - 5.0
                        self.recording_fps_tracker['frame_times'] = [t for t in self.recording_fps_tracker['frame_times'] if t > cutoff]
                    with self.processing_lock:
                        self.recorded_frames += 1
        except Exception:
            # Avoid spamming logs during recording
            pass
    
    def _record_frame_async(self, frame):
        """High-performance asynchronous frame recording with FPS tracking."""
        if not self.is_recording or not self.hdf5_recorder:
            return
        
        # Validate frame is not all-black or all-gray (bad frame detection)
        if frame.max() == 0 or (frame.min() == frame.max()):
            # Skip invalid frames (all same value = bad camera data)
            return
        
        try:
            # Track frame time for FPS monitoring
            if hasattr(self, 'recording_fps_tracker'):
                current_time = time.time()
                self.recording_fps_tracker['frame_times'].append(current_time)
                
                # Keep only recent frame times (last 5 seconds)
                cutoff_time = current_time - 5.0
                self.recording_fps_tracker['frame_times'] = [
                    t for t in self.recording_fps_tracker['frame_times'] if t > cutoff_time
                ]
            
            # Use recorder's enqueue API to offload downscaling and writing
            try:
                if hasattr(self.hdf5_recorder, 'enqueue_frame'):
                    if self.hdf5_recorder.enqueue_frame(frame):
                        with self.processing_lock:
                            self.recorded_frames += 1
                else:
                    # Fallback to older API
                    if self.hdf5_recorder.record_frame(frame, use_async=True):
                        with self.processing_lock:
                            self.recorded_frames += 1
            except Exception:
                # Silent failure to prevent excessive logging when disk is full
                pass
        except Exception as e:
            # Only log errors if we can write to disk safely
            try:
                logger.error(f"Error in async frame recording: {e}")
            except OSError:
                pass  # Disk full - silent failure
    
    def cleanup(self):
        """Clean up resources including thread pool."""
        try:
            # Shutdown executors gracefully
            try:
                self.display_executor.shutdown(wait=True)
            except Exception:
                try:
                    self.display_executor.shutdown(wait=False)
                except Exception:
                    pass

            try:
                self.recording_executor.shutdown(wait=True)
            except Exception:
                try:
                    self.recording_executor.shutdown(wait=False)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Error shutting down thread pool: {e}")
    
    def _record_frame(self, frame):
        """Legacy synchronous frame recording for compatibility."""
        return self._record_frame_async(frame)
    
    def _process_display_frame(self, frame: np.ndarray):
        """Process grayscale frame for display (GPU-accelerated if available)."""
        try:
            # Always grayscale: Convert to RGB for Qt display
            if len(frame.shape) == 3 and frame.shape[2] == 1:
                frame = frame.squeeze()  # Remove single channel dimension
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # Frame is already RGB (shouldn't happen, but handle it)
                rgb_frame = frame
            elif len(frame.shape) != 2:
                logger.error(f"Unexpected frame shape: {frame.shape}")
                return
            
            # Only convert if frame is grayscale (2D)
            if len(frame.shape) == 2:
                # Use GPU acceleration for color conversion if available
                if _GPU_AVAILABLE:
                    try:
                        gpu_frame = cv2.UMat(frame)
                        gpu_rgb = cv2.cvtColor(gpu_frame, cv2.COLOR_GRAY2RGB)
                        rgb_frame = gpu_rgb.get()
                    except Exception:
                        # Fallback to CPU if GPU fails
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Apply image processing settings (currently none - returns as-is)
            processed_frame = self._apply_image_processing(rgb_frame)
            
            # Create QImage and emit to GUI thread for display
            h, w = processed_frame.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.display_image_signal.emit(q_image)
            
        except Exception as e:
            logger.error(f"Error processing display frame: {e}")

    def _prepare_display_image(self, frame: np.ndarray):
        """Worker that prepares a QImage from a numpy frame off the GUI thread.

        This converts grayscale to RGB, applies image processing, and constructs
        a QImage which is then emitted back to the GUI thread via `display_image_signal`.
        """
        try:
            import cv2
            # Convert grayscale to RGB for display
            if len(frame.shape) == 3 and frame.shape[2] == 1:
                frame2 = frame.squeeze()
                rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 2:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = frame
            else:
                rgb_frame = frame

            # Always apply image processing for consistent display
            processed = self._apply_image_processing(rgb_frame)

            # Create QImage (no deep copy if possible)
            h, w = processed.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(processed.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Emit QImage back to GUI thread
            self.display_image_signal.emit(q_image)
        except Exception as e:
            logger.debug(f"Display worker error: {e}")

    def _on_display_image(self, q_image):
        """Slot running in GUI thread to set the pixmap from a QImage."""
        try:
            pixmap = QPixmap.fromImage(q_image)
            display_size = self.display_label.size()
            if display_size.width() > 0 and display_size.height() > 0:
                scaled = pixmap.scaled(display_size, Qt.KeepAspectRatio, Qt.FastTransformation)
            else:
                scaled = pixmap
            self.display_label.setPixmap(scaled)
        except Exception as e:
            logger.debug(f"Error updating display pixmap: {e}")

    def _display_task_done(self, future):
        """Callback for Future done to clear inflight flag and log errors if any."""
        try:
            err = future.exception()
            if err:
                logger.debug(f"Display task error: {err}")
        except Exception:
            pass
        finally:
            # Mark that a display task can be submitted again
            try:
                self._display_task_inflight = False
            except Exception:
                pass

    def _fast_update_display(self, frame: np.ndarray):
        """Minimal display update optimized for 12 FPS without lag."""
        try:
            if frame is None:
                return
            
            # Direct grayscale display - skip RGB conversion for speed
            if len(frame.shape) == 3 and frame.shape[2] == 1:
                frame = frame.squeeze()  # Remove single channel dimension
            
            if len(frame.shape) == 2:
                # Grayscale: use Format_Grayscale8 (fastest, no conversion needed)
                h, w = frame.shape
                bytes_per_line = w
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            else:
                # RGB fallback
                h, w = frame.shape[:2]
                bytes_per_line = 3 * w
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Fast scaling
            display_size = self.display_label.size()
            if display_size.width() > 0 and display_size.height() > 0:
                scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                    display_size, Qt.KeepAspectRatio, Qt.FastTransformation
                )
            else:
                scaled_pixmap = QPixmap.fromImage(q_image)
            
            self.display_label.setPixmap(scaled_pixmap)
        except Exception as e:
            logger.debug(f"Display error: {e}")
    
    def _update_display_thread_safe(self, processed_frame):
        """Update display from any thread safely."""
        h, w = processed_frame.shape[:2]
        bytes_per_line = 3 * w
        
        q_image = QImage(
            processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        
        # Scale to fit display while maintaining aspect ratio
        display_size = self.display_label.size()
        if display_size.width() > 0 and display_size.height() > 0:
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            scaled_pixmap = QPixmap.fromImage(q_image)
        
        # Thread-safe update using Qt's thread communication
        self.display_label.setPixmap(scaled_pixmap)
    
    # Function generator logging methods (preserved)
    def update_image_settings(self, settings: dict):
        """Update camera settings from camera settings dialog."""
        # Update live display FPS if provided
        if 'live_fps' in settings:
            new_live_fps = int(settings['live_fps']) + 2  # Add 2 offset so 12 becomes 14 (actual 12)
            if new_live_fps != self.live_display_fps:
                self.live_display_fps = new_live_fps
                self._min_display_interval = 1.0 / self.live_display_fps
                self.fallback_timer.setInterval(int(1000.0 / self.live_display_fps))
                logger.info(f"Live display FPS updated to: {self.live_display_fps} fps (from setting: {settings['live_fps']})")
        
        # Update recording FPS if provided
        if 'recording_fps' in settings:
            new_recording_fps = int(settings['recording_fps'])
            if new_recording_fps != self.recording_settings['recording_fps']:
                self.recording_settings['recording_fps'] = new_recording_fps
                logger.info(f"Recording FPS updated to: {self.recording_settings['recording_fps']} fps")
    
    def set_recording_compression(self, compression_level: int):
        """Set compression level for HDF5 recording.
        
        Args:
            compression_level: 0=none, 1-3=fast LZF, 4-9=best GZIP
        """
        self.recording_settings['compression_level'] = max(0, min(9, compression_level))
        logger.info(f"Recording compression level set to: {self.recording_settings['compression_level']}")
    
    def set_recording_resolution(self, downscale_factor: int):
        """Set downscale factor for HDF5 recording.
        
        Args:
            downscale_factor: 1=full resolution, 2=half, 4=quarter
        """
        valid_factors = [1, 2, 4]
        if downscale_factor not in valid_factors:
            downscale_factor = min(valid_factors, key=lambda x: abs(x - downscale_factor))
        
        self.recording_settings['downscale_factor'] = downscale_factor
        logger.info(f"Recording downscale factor set to: {self.recording_settings['downscale_factor']}")
    
    def log_function_generator_event(self, frequency_mhz: float, amplitude_vpp: float,
                                   output_enabled: bool = True, event_type: str = 'parameter_change'):
        """Log function generator events."""
        if self.hdf5_recorder and self.is_recording:
            try:
                self.hdf5_recorder.log_function_generator_event(
                    frequency_mhz, amplitude_vpp,
                    output_enabled=output_enabled, event_type=event_type
                )
            except Exception as e:
                logger.error(f"Failed to log function generator event: {e}")
    
    def log_function_generator_toggle(self, enabled: bool, frequency_mhz: float = 1.0, 
                                    amplitude_vpp: float = 1.0):
        """Log function generator toggle events."""
        if self.hdf5_recorder and self.is_recording:
            try:
                event_type = 'output_on' if enabled else 'output_off'
                self.hdf5_recorder.log_function_generator_event(
                    frequency_mhz, amplitude_vpp,
                    output_enabled=enabled, event_type=event_type
                )
            except Exception as e:
                logger.error(f"Failed to log function generator toggle: {e}")
    
    def log_initial_function_generator_state(self, frequency_mhz: float, 
                                           amplitude_vpp: float, enabled: bool):
        """Log initial function generator state."""
        if self.hdf5_recorder and self.is_recording:
            try:
                self.hdf5_recorder.log_function_generator_event(
                    frequency_mhz, amplitude_vpp,
                    output_enabled=enabled, event_type='initial_state'
                )
            except Exception as e:
                logger.error(f"Failed to log initial function generator state: {e}")
    
    def _apply_default_camera_settings(self):
        """Apply default camera settings optimized for live view."""
        if not self.camera or not hasattr(self.camera, 'apply_settings'):
            logger.warning("Cannot apply default settings: camera not available")
            return
        
        # Default settings optimized for maximum FPS (60 FPS target)
        default_settings = {
            'exposure_ms': 5.0,  # Minimum exposure for maximum FPS (5ms allows ~200 FPS)
            'gain_master': 2,
            'fps': 60.0  # Request 60 FPS for smooth high-speed recording
        }
        
        try:
            result = self.camera.apply_settings(default_settings)
            logger.info(f"Applied default camera settings: {result}")
        except Exception as e:
            logger.warning(f"Failed to apply default camera settings: {e}")
    
    def _on_frame_drop_alert(self, drop_rate_percent, total_drops, total_captured):
        """Callback for frame drop alerts during recording.
        
        Args:
            drop_rate_percent: Frame drop rate as percentage
            total_drops: Total number of frames dropped
            total_captured: Total number of frames captured
        """
        from PyQt5.QtWidgets import QMessageBox
        
        # Show alert dialog (non-blocking)
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Frame Drop Warning")
        msg.setText(f"Frame drop rate: {drop_rate_percent:.2f}%")
        msg.setInformativeText(
            f"Dropped {total_drops} of {total_captured} frames.\n\n"
            "Possible causes:\n"
            "• Disk write speed too slow\n"
            "• CPU overloaded\n"
            "• Reduce recording resolution or FPS"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setModal(False)  # Non-blocking
        msg.show()
        
        logger.warning(f"Frame drop alert: {drop_rate_percent:.2f}% ({total_drops}/{total_captured})")
    
    def _on_health_alert(self, issue_type, description):
        """Callback for hardware health alerts during recording.
        
        Args:
            issue_type: Type of health issue (disk_space, queue_full, etc.)
            description: Human-readable description of the issue
        """
        from PyQt5.QtWidgets import QMessageBox
        
        # Show alert dialog (non-blocking)
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Hardware Health Warning")
        
        if issue_type == "disk_space":
            msg.setText("Low Disk Space")
            msg.setInformativeText(
                f"{description}\n\n"
                "Recording may stop soon. Free up disk space immediately."
            )
        elif issue_type == "queue_full":
            msg.setText("Write Queue Full")
            msg.setInformativeText(
                f"{description}\n\n"
                "System cannot write frames fast enough. Consider:\n"
                "• Reducing recording resolution\n"
                "• Reducing FPS\n"
                "• Using faster storage"
            )
        else:
            msg.setText("Hardware Issue Detected")
            msg.setInformativeText(description)
        
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setModal(False)  # Non-blocking
        msg.show()
        
        logger.warning(f"Hardware health alert [{issue_type}]: {description}")
    
    def close(self):
        """Clean shutdown of camera widget and resources.
        
        Ensures proper cleanup of:
        - Active recording (if in progress)
        - Camera connection
        - Thread pool executors
        - Background threads
        """
        if self.is_recording:
            self.stop_recording()
        self.stop_camera()
        
        # Shutdown thread pool executors to prevent resource leaks
        if hasattr(self, 'display_executor') and self.display_executor:
            try:
                self.display_executor.shutdown(wait=False)
                logger.debug("Display executor shut down")
            except Exception as e:
                logger.debug(f"Error shutting down display executor: {e}")
        
        if hasattr(self, 'recording_executor') and self.recording_executor:
            try:
                self.recording_executor.shutdown(wait=False)
                logger.debug("Recording executor shut down")
            except Exception as e:
                logger.debug(f"Error shutting down recording executor: {e}")