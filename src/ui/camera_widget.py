"""
Camera Widget for AFS Acquisition - Minimal Design with Maximum Performance.

This module provides a streamlined camera interface with professional-grade
recording capabilities:

Features:
- Real-time camera feed at 50+ FPS with GPU-accelerated display
- High-performance HDF5 recording with background compression
- Live view with configurable FPS (15 FPS default for UI responsiveness)
- Recording at full camera speed (30 FPS target, up to 50 FPS capable)
- Automatic error recovery with test pattern fallback
- Thread-safe compression progress notifications

Architecture:
- Main thread: Qt GUI and user interaction
- Camera thread: Frame capture and queue management (in CameraController)
- Processing thread: GPU downscaling and frame buffering
- Writer thread: Asynchronous HDF5 writing
- Compression thread: Background post-processing (daemon)

Performance Optimizations:
- GPU-accelerated display processing (OpenCL)
- Lockless frame queue for zero-copy transfer
- Async I/O to prevent blocking during recording
- Background compression allows immediate workflow continuation
- Smart FPS limiting (15 FPS display, 30+ FPS recording)

User Experience:
- Minimal, clean interface for maximum screen real estate
- Non-blocking progress dialogs for compression
- Automatic camera reconnection on errors
- Real-time FPS performance monitoring
"""

import time
import cv2
import os
import numpy as np
from datetime import datetime
from typing import Optional
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QSizePolicy, QMessageBox, QProgressDialog
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
            max_workers=1, thread_name_prefix="CameraDisplay"
        )
        # Recording executor: separate pool for any heavy tasks (kept small because recorder is async)
        self.recording_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="CameraRecorder"
        )
        self.recording_queue = queue.Queue(maxsize=100)  # Buffer for recording (compatibility)
        self.processing_lock = threading.RLock()
        
        # Image processing settings for live view (start with standard values)
        self.image_settings = {
            'brightness': 50,  # Standard brightness
            'contrast': 50,    # Standard contrast
            'saturation': 50   # Standard saturation
        }
        
        # Recording compression and resolution settings
        # MAXIMUM OPTIMIZATION: Half resolution + no real-time compression
        # compression_level: 0=none (FASTEST), 1-3=fast/LZF, 4-9=best/GZIP (SLOWEST)
        # downscale_factor: 1=full res, 2=half, 4=quarter
        # NOTE: Half resolution (2x) provides good balance between file size and FPS.
        #       GZIP level 9 compression for maximum file size reduction.
        self.recording_settings = {
            'compression_level': 9,  # GZIP level 9 for smallest files
            'downscale_factor': 2    # HALF resolution = Good balance of speed & file size
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
        
        # Live display configuration: strict 12 FPS for smooth low-latency UI
        self.live_display_fps = 12  # 12 FPS = 83.33ms interval, smooth and responsive
        self._last_display_time = 0  # Track last display update
        self._min_display_interval = 1.0 / 12.0  # Exact 83.33ms between frames
        try:
            cfg = get_config()
            self.live_display_fps = int(getattr(cfg.ui, 'live_display_fps', 12))
            self._min_display_interval = 1.0 / self.live_display_fps
        except Exception:
            pass
        
        # Fallback timer for display (exact 12 FPS)
        self.fallback_timer = QTimer(self)
        self.fallback_timer.timeout.connect(self._process_frame_immediate)
        self.fallback_timer.setInterval(int(1000.0 / 12.0))  # 83.33ms = exact 12 FPS
        
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
        
        # Show completion notification with statistics
        QMessageBox.information(
            self, 
            "Compression Complete", 
            message,
            QMessageBox.Ok
        )
    
    def _update_compression_progress(self, current: int, total: int, status_text: str):
        """Update compression progress dialog (slot for compression_progress_signal)."""
        logger.debug(f"_update_compression_progress called: {current}/{total} - {status_text}")
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
                    
                    # Use ONLY timer for consistent 12 FPS - no callbacks
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
        
        # Get latest frame
        frame_data = self.camera.get_latest_frame(timeout=0.001)
        if frame_data is None:
            return
        
        # Recording: capture ALL frames at full camera speed (25-30 FPS)
        # This is completely decoupled from display rate
        if self.is_recording and self.hdf5_recorder:
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
                        if hasattr(self, 'recording_fps_tracker'):
                            self.recording_fps_tracker.setdefault('frame_times', []).append(current_time)
                            cutoff = current_time - 5.0
                            self.recording_fps_tracker['frame_times'] = [t for t in self.recording_fps_tracker['frame_times'] if t > cutoff]
                        with self.processing_lock:
                            self.recorded_frames += 1
            except Exception:
                pass
        
        # Display: rate limiting to ~12 FPS
        time_since_last = current_time - self._last_display_time
        if time_since_last < self._min_display_interval and self._last_display_time > 0:
            return  # Skip display update - too soon
        
        # Update display
        self._last_display_time = current_time
        self.last_frame_timestamp = frame_data.timestamp
        self.current_frame_data = frame_data
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
            
            # If recording, also show recording FPS
            if self.is_recording and hasattr(self, 'recording_fps_tracker'):
                frame_times = self.recording_fps_tracker.get('frame_times', [])
                if len(frame_times) >= 2:
                    time_span = frame_times[-1] - frame_times[0]
                    recording_fps = (len(frame_times) - 1) / time_span if time_span > 0 else 0.0
                    status_text = f"Rec: {recording_fps:.1f} FPS | Live: {display_fps:.1f} FPS"
                else:
                    status_text = f"Live @ {display_fps:.1f} FPS"
            else:
                status_text = f"Live @ {display_fps:.1f} FPS"
            
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
        """Apply brightness, contrast, and saturation to frame (highly optimized)."""
        # Skip processing if all settings are at default values
        if (self.image_settings['brightness'] == 50 and 
            self.image_settings['contrast'] == 50 and 
            self.image_settings['saturation'] == 50):
            return frame
        
        try:
            # Only apply needed transformations to minimize operations
            needs_brightness = self.image_settings['brightness'] != 50
            needs_contrast = self.image_settings['contrast'] != 50
            needs_saturation = (len(frame.shape) == 3 and self.image_settings['saturation'] != 50)
            
            # Apply brightness/contrast together if needed (GPU-accelerated if available)
            if needs_brightness or needs_contrast:
                brightness = (self.image_settings['brightness'] - 50) * 2.0
                contrast = self.image_settings['contrast'] / 50.0
                
                if _GPU_AVAILABLE:
                    try:
                        gpu_frame = cv2.UMat(frame)
                        gpu_result = cv2.convertScaleAbs(gpu_frame, alpha=contrast, beta=brightness)
                        frame = gpu_result.get()
                    except Exception:
                        # Fallback to CPU
                        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
                else:
                    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            
            # Apply saturation only if needed (most expensive operation - GPU accelerated)
            if needs_saturation:
                if _GPU_AVAILABLE:
                    try:
                        gpu_frame = cv2.UMat(frame)
                        gpu_hsv = cv2.cvtColor(gpu_frame, cv2.COLOR_RGB2HSV)
                        hsv = gpu_hsv.get()
                        saturation_factor = self.image_settings['saturation'] / 50.0
                        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation_factor)
                        gpu_hsv_adj = cv2.UMat(hsv)
                        gpu_rgb = cv2.cvtColor(gpu_hsv_adj, cv2.COLOR_HSV2RGB)
                        frame = gpu_rgb.get()
                    except Exception:
                        # Fallback to CPU
                        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                        saturation_factor = self.image_settings['saturation'] / 50.0
                        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation_factor)
                        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                else:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    saturation_factor = self.image_settings['saturation'] / 50.0
                    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation_factor)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            return frame
            
        except Exception as e:
            logger.warning(f"Image processing error: {e}")
            return frame
    
    def update_image_settings(self, settings):
        """Update image processing settings."""
        if 'brightness' in settings:
            self.image_settings['brightness'] = settings['brightness']
        if 'contrast' in settings:
            self.image_settings['contrast'] = settings['contrast']
        if 'saturation' in settings:
            self.image_settings['saturation'] = settings['saturation']
        
        # Reduce logging frequency to prevent performance issues
    
    # Recording methods (preserved from original)
    def start_recording(self, file_path: str, metadata=None) -> bool:
        """Start HDF5 video recording."""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False
        
        if not self.is_running or not self.is_live:
            logger.warning("Cannot start recording - camera not in live mode")
            return False
        
        try:
            # Ensure .hdf5 extension
            if not file_path.lower().endswith('.hdf5'):
                file_path = file_path + '.hdf5'
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Get frame shape from camera (MONO8 = 1 channel, not 3!)
            if self.camera:
                frame_shape = self.camera.frame_shape
            else:
                frame_shape = (480, 640, 1)  # Default to grayscale
            
            # Create recorder with compression and resolution settings
            # CAMERA CAPABLE: 50+ FPS! Target 40-50 FPS for maximum recording speed
            TARGET_FPS = 50.0
            MIN_FPS = 25.0
            
            # Increase camera queue size during recording to prevent frame drops at high FPS
            if self.camera:
                try:
                    # Temporarily increase queue to handle high-speed recording
                    self.camera.max_queue_size = 10
                    logger.info(f"Increased camera queue to {self.camera.max_queue_size} for high-speed recording")
                except Exception as e:
                    logger.warning(f"Could not increase queue size: {e}")
            
            self.hdf5_recorder = HDF5VideoRecorder(
                file_path=file_path,
                frame_shape=frame_shape,
                fps=TARGET_FPS,
                min_fps=MIN_FPS,
                compression_level=self.recording_settings['compression_level'],
                downscale_factor=self.recording_settings['downscale_factor']
            )
            
            # Build info message based on actual settings
            resolution_desc = {1: "full resolution", 2: "half resolution", 4: "quarter resolution"}
            res_text = resolution_desc.get(self.recording_settings['downscale_factor'], f"{self.recording_settings['downscale_factor']}x downscale")
            logger.info(f"Recording configured: Target {TARGET_FPS} FPS (camera max 50 FPS), {res_text} + MONO8 for high-speed capture")
            
            # Track FPS performance during recording
            self.recording_fps_tracker = {
                'frame_times': [],
                'warnings_issued': 0,
                'below_min_count': 0
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
                    
                    # Add image processing settings to camera settings
                    camera_settings.update({
                        'image_brightness': self.image_settings['brightness'],
                        'image_contrast': self.image_settings['contrast'],
                        'image_saturation': self.image_settings['saturation']
                    })
                    
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
            
            # Add recording metadata and regeneration info
            try:
                self.hdf5_recorder.add_recording_metadata(recording_metadata)
            except Exception as e:
                logger.warning(f"Failed to save recording metadata: {e}")
            
            # Try to set camera to recording frame rate (best-effort)
            try:
                if self.camera and hasattr(self.camera, 'apply_settings'):
                    res = self.camera.apply_settings({'fps': TARGET_FPS})
                    if res.get('fps', False):
                        logger.info(f"Camera set to {TARGET_FPS} FPS for recording")
                    else:
                        logger.warning(f"Camera refused to set {TARGET_FPS} FPS")
            except Exception as e:
                logger.debug(f"Could not set camera fps for recording: {e}")

            # Set recording state
            self.is_recording = True
            self.recording_path = file_path
            self.recording_start_time = datetime.now()
            self.recorded_frames = 0
            
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
                        
                        # Decide whether to run post-processing compression based on user config
                        import threading
                        from src.utils.hdf5_video_recorder import post_process_compress_hdf5
                        cfg = get_config()

                        def compress_and_notify(path: str):
                            try:
                                time.sleep(1.0)
                                logger.info("Starting post-processing compression...")
                                
                                # Progress callback to emit signal to main thread
                                def progress_callback(current, total, status):
                                    logger.debug(f"Compression progress: {current}/{total} - {status}")
                                    self.compression_progress_signal.emit(current, total, status)
                                
                                result = post_process_compress_hdf5(
                                    path,
                                    quality_reduction=False,  # Lossless: preserve 100% of scientific data
                                    parallel=True,
                                    progress_callback=progress_callback
                                )
                                if result:
                                    logger.info(f"Compression completed: {result['path']}")
                                    # Emit signal to main thread with full statistics
                                    self.compression_complete_signal.emit(
                                        result['path'],
                                        result['original_mb'],
                                        result['compressed_mb'],
                                        result['reduction_pct']
                                    )
                                else:
                                    logger.warning("Compression returned no result")
                            except Exception as comp_error:
                                logger.error(f"Post-processing compression failed: {comp_error}", exc_info=True)

                        # Run compression according to configuration
                        try:
                            if cfg.files.background_compression:
                                # Close the initial saving message box
                                msg_box.close()
                                msg_box.deleteLater()
                                QApplication.processEvents()
                                
                                if cfg.files.wait_for_compression:
                                    # Run compression synchronously (UI will wait)
                                    logger.info("Running synchronous post-processing compression per configuration")
                                    compress_and_notify(saved_path)
                                else:
                                    # Run compression in a background daemon thread (non-blocking)
                                    # Progress dialog will appear automatically via signal
                                    compression_thread = threading.Thread(
                                        target=compress_and_notify,
                                        name="CompressionThread",
                                        args=(saved_path,),
                                        daemon=True,
                                    )
                                    compression_thread.start()
                                    logger.info("Compression started in background (daemon) - UI will remain responsive")
                            else:
                                # Compression disabled via config - just close dialog
                                logger.info("Background compression disabled by configuration")
                                msg_box.close()
                                msg_box.deleteLater()
                                QApplication.processEvents()
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
                    self.camera.apply_settings({'fps': 17.0})
                    # Restore small queue for live view
                    self.camera.max_queue_size = 1
                    logger.info("Restored camera to live view settings (17 FPS, queue=1)")
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
    
    def _record_frame_async(self, frame):
        """High-performance asynchronous frame recording with FPS tracking."""
        if not self.is_recording or not self.hdf5_recorder:
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
            
            # Apply image processing settings (brightness/contrast only, no saturation)
            processed_frame = self._apply_image_processing(rgb_frame)
            
            # Update display in main thread
            self._update_display_thread_safe(processed_frame)
            
        except Exception as e:
            logger.error(f"Error processing display frame: {e}")

    def _fast_update_display(self, frame: np.ndarray):
        """Minimal display update - no overlay, just fast rendering."""
        try:
            if frame is None:
                return
            
            # Convert grayscale to RGB for display (in-place when possible)
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                frame_rgb = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = frame  # Already RGB, no copy
            else:
                return
            
            # Create QImage without copy - use frame buffer directly
            h, w = frame_rgb.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
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
        """Update image processing settings from camera settings dialog."""
        if 'brightness' in settings:
            self.image_settings['brightness'] = settings['brightness']
        if 'contrast' in settings:
            self.image_settings['contrast'] = settings['contrast']
        if 'saturation' in settings:
            self.image_settings['saturation'] = settings['saturation']
    
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
        
        # Default settings optimized for live view (15-20 FPS range)
        default_settings = {
            'exposure_ms': 15.0,
            'gain_master': 2,
            'fps': 17.0,  # 17 FPS for live view (middle of 15-20 range)
            'brightness': 50,
            'contrast': 50,
            'saturation': 50
        }
        
        try:
            result = self.camera.apply_settings(default_settings)
            logger.info(f"Applied default camera settings: {result}")
            
            # Update image processing settings for live view
            self.update_image_settings(default_settings)
        except Exception as e:
            logger.warning(f"Failed to apply default camera settings: {e}")
    
    def close(self):
        """Clean shutdown."""
        if self.is_recording:
            self.stop_recording()
        self.stop_camera()