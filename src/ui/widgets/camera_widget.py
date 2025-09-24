import time
import numpy as np
import cv2
import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from src.utils.logger import get_logger
from src.controllers.camera_controller import CameraController, FrameData
from src.utils.status_display import StatusDisplay
from src.utils.hdf5_video_recorder import HDF5VideoRecorder

logger = get_logger("camera_gui")



class CameraWidget(QGroupBox):
    """Camera widget with status indicator."""

    def __init__(self, parent=None):
        super().__init__("Camera", parent)

        logger.info("Camera initialized")
        
        # Camera state - using high-performance controller
        self.camera: CameraController = None
        self.is_running = False
        self.is_live = False
        self.camera_error = None
        
        # Frame data
        self.current_frame_data: FrameData = None
        self.last_frame_timestamp = 0
        
        # Warning suppression to prevent log spam
        self.last_warning_time = 0
        self.warning_count = 0
        self.log_warning_threshold = 30  # Only log every 30 attempts when repeating
        
        # Reconnection state
        self.reconnection_start_time = 0
        self.reconnection_attempts = 0

        # Display + timers
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        
        # Video recording state
        self.is_recording = False
        self.hdf5_recorder = None
        self.recording_path = ""
        self.recording_start_time = None
        self.recorded_frames = 0
        
        # Frame rate control for recording
        self.target_recording_fps = 25.0
        self.last_recorded_time = 0
        self.frame_interval = 1.0 / self.target_recording_fps
        
        # Recording indicators
        self.recording_indicator = QLabel()
        self.recording_stats = QLabel()
        self.recording_stats.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.recording_indicator.setFixedSize(12, 12)
        self.recording_indicator.setStyleSheet(
            "background-color: gray; border-radius: 6px; margin: 2px;"
        )
        
        # Maximize camera view - set size policy to strongly expand in both directions
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setMinimumSize(320, 240)  # Set minimum size
        
        # Set the widget to strongly prefer expanding
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Display update timer (higher frequency for smooth display)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frame)
        self.update_timer.setInterval(16)  # ~60 FPS display refresh for smooth playback
        
        # Image processing toggles (keep for internal use)
        self.mono_enabled = False
        self.auto_contrast = False
        self.auto_brightness = True  # Enable automatic brightness adjustment
        self.target_brightness = 128  # Target average brightness (0-255)
        self.brightness_adjustment_rate = 0.1  # How fast to adjust (0.0-1.0)

        # Status display
        self.status_display = StatusDisplay()

        self.init_ui()

        # Set initial status to show initialization is starting
        self.update_status("Initializing...")

        # Auto-connect shortly after startup with shorter delay
        QTimer.singleShot(100, self.connect_camera)

    def init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 24, 8, 8)

        # Create a frame to wrap camera view and controls
        camera_frame = QFrame()
        camera_frame.setFrameShape(QFrame.StyledPanel)
        camera_frame.setFrameShadow(QFrame.Raised)
        camera_frame.setLineWidth(1)
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(5, 5, 5, 5)
        
        # Camera display area
        camera_layout.addWidget(self.display_label, 1)

        # Bottom section with controls on left, status on right
        bottom_row = QHBoxLayout()
        
        # Left side: Camera control buttons
        button_layout = QHBoxLayout()
        self.play_button = QPushButton("▶️ Play"); self.play_button.clicked.connect(self.set_live_mode)
        self.pause_button = QPushButton("⏸️ Pause"); self.pause_button.clicked.connect(self.set_pause_mode)
        self.reconnect_button = QPushButton("🔄 Re-initialize"); self.reconnect_button.clicked.connect(self.reconnect_camera)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reconnect_button)
        button_layout.addStretch(1)
        
        # Right side: Status display with recording indicator
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.recording_indicator)
        status_layout.addWidget(self.recording_stats)
        status_layout.addStretch(1)
        status_layout.addWidget(self.status_display)
        
        # Add both sides to bottom row
        bottom_row.addLayout(button_layout, 3)  # Buttons take 3/4 of space
        bottom_row.addLayout(status_layout, 1)  # Status takes 1/4 of space
        
        # Add bottom row to camera layout
        camera_layout.addLayout(bottom_row)
        
        # Add camera frame to main layout
        main.addWidget(camera_frame)

        self.update_button_states()

    def update_button_states(self):
        self.play_button.setEnabled(self.is_running and not self.is_live and self.camera is not None)
        self.pause_button.setEnabled(self.is_running and self.is_live)
        # Reconnect button is always enabled as long as we're running
        self.reconnect_button.setEnabled(self.is_running)
        
    def try_reconnect(self):
        """
        Try to reconnect the camera with a 5-second timeout.
        Shows countdown in the status indicator and gives up after timeout.
        """
        # Check if we've exceeded the 5-second timeout
        elapsed_time = time.time() - self.reconnection_start_time
        self.reconnection_attempts += 1
        
        # Calculate remaining time and update status with countdown
        remaining_time = max(0, int(5.0 - elapsed_time))
        self.update_status(f"Initializing ({remaining_time}s)...")
        
        if elapsed_time > 5.0:  # 5-second timeout (reduced from 10)
            # Give up after 5 seconds and fall back to test pattern
            logger.info("Camera hardware not found - using test pattern")
            self.use_test_pattern = True
            self.camera = None
            self.is_running = True
            self.is_live = False
            self.update_status("Test Pattern Mode")
            self.update_button_states()
            if not self.update_timer.isActive():
                self.update_timer.start()
            self.set_live_mode()  # Start test pattern
            # Re-enable initialize button (text stays the same)
            self.reconnect_button.setEnabled(True)
            return
            
        # Try to connect using high-performance controller
        try:
            self.camera = CameraController(camera_id=0, max_queue_size=10)
            if self.camera.initialize():
                if self.camera.start_capture():
                    # Success!
                    self.is_running = True
                    self.is_live = False
                    self.update_status("Connected")
                    self.update_button_states()
                    if not self.update_timer.isActive():
                        self.update_timer.start()
                    self.set_live_mode()  # Auto start live view
                    logger.info("Camera connected")
                    # Re-enable initialize button (text stays the same)
                    self.reconnect_button.setEnabled(True)
                    return
                else:
                    # Start capture failed
                    if self.camera:
                        self.camera.close()
                        self.camera = None
            else:
                # Initialization failed
                if self.camera:
                    self.camera.close()
                    self.camera = None
            
            QTimer.singleShot(300, self.try_reconnect)  # Faster retry
            
        except Exception as e:
            # Silent retry - only log if it becomes a persistent issue
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
                self.camera = None
            QTimer.singleShot(300, self.try_reconnect)
        
    def reconnect_camera(self):
        """Attempt to initialize/reconnect the camera hardware."""
        self.update_status("Initializing...")
        
        # Stop current camera operations
        self.stop_camera()
        
        # Reset error states
        self.camera_error = None
        self.warning_count = 0
        
        # Disable initialize button during attempt (but keep the text the same)
        self.reconnect_button.setEnabled(False)
        
        # Start initialization attempt with timeout
        self.reconnection_start_time = time.time()
        self.reconnection_attempts = 0
        
        # Small delay to ensure cleanup is complete
        QTimer.singleShot(100, self.try_reconnect)

    def update_status(self, text):
        self.status_display.set_status(text)

    def connect_camera(self, camera_id=0):
        """
        Connect to the camera with the specified ID.
        
        This method is called automatically at startup. For manual reconnection
        with timeout handling, use reconnect_camera() instead.
        
        Args:
            camera_id (int): ID of the camera to connect to (default: 0)
        """
        if self.is_running:
            return
        
        self.update_status("Initializing...")
        
        # Create high-performance camera controller
        try:
            self.update_status("Connecting...")
            self.camera = CameraController(camera_id=camera_id, max_queue_size=10)
            
            # Initialize and start capture
            if self.camera.initialize():
                if self.camera.start_capture():
                    # Camera started successfully
                    logger.info("Camera connected")
                    self.is_running = True
                    self.is_live = False
                    self.update_status("Connected")
                    self.update_button_states()
                    self.update_timer.start()
                    self.set_live_mode()  # Auto start live view
                    return
                else:
                    # Start capture failed
                    logger.warning("Failed to start camera capture")
                    if self.camera:
                        self.camera.close()
                        self.camera = None
            else:
                # Hardware initialization failed
                logger.warning("Camera hardware initialization failed")
                if self.camera:
                    self.camera.close()
                    self.camera = None
            
            # Fall through to test pattern mode
            self._start_test_pattern_mode("Hardware Not Found - Test Pattern")
            
        except Exception as e:
            logger.warning(f"Camera initialization error: {e}")
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
                self.camera = None
            self._start_test_pattern_mode("Hardware Error - Test Pattern")
    
    def _start_test_pattern_mode(self, status_text):
        """Helper method to start test pattern mode with the given status text"""
        try:
            # Create camera controller in test pattern mode
            self.camera = CameraController(camera_id=0, max_queue_size=5)
            if self.camera.initialize() and self.camera.start_capture():
                self.is_running = True
                self.is_live = False
                self.update_status(status_text)
                self.update_button_states()
                self.update_timer.start()
                self.set_live_mode()  # Auto start test pattern
            else:
                # Even test pattern failed
                self.camera = None
                self.is_running = False
                self.update_status("Camera Error")
                self.update_button_states()
        except Exception as e:
            logger.error(f"Failed to start test pattern mode: {e}")
            self.camera = None
            self.is_running = False
            self.update_status("Camera Error")
            self.update_button_states()

    def stop_camera(self):
        if self.update_timer.isActive():
            self.update_timer.stop()
        self.is_running = False
        self.is_live = False
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
            self.camera = None
        self.current_frame_data = None
        self.update_status("Initialize")
        self.update_button_states()

    def set_live_mode(self):
        if not self.is_running:
            logger.warning("Cannot set live mode - camera not running")
            return
            
        if not self.camera:
            logger.warning("Cannot set live mode - no camera")
            self.update_status("Camera not found")
            return
        
        # Check if camera is using test pattern
        stats = self.camera.get_statistics()
        if stats.get('use_test_pattern', False):
            self.is_live = True
            self.update_status("Test Pattern Active")
            self.update_button_states()
        else:
            # Regular camera live mode
            self.is_live = True
            self.update_status("Live")
            self.update_button_states()

    def set_pause_mode(self):
        if not self.is_running:
            return
        self.is_live = False
        self.update_status("Paused")
        self.update_button_states()

    def toggle_mono(self):
        self.mono_enabled = not self.mono_enabled
        self.mono_button.setChecked(self.mono_enabled)

    def toggle_auto_contrast(self):
        self.auto_contrast = not self.auto_contrast
        self.auto_contrast_button.setChecked(self.auto_contrast)



    def update_frame(self):
        """
        Update the camera frame display. Called by update_timer.
        
        Gets the latest frame from the threaded camera controller and displays it.
        Also handles frame recording and statistics display.
        """
        if not self.is_running or not self.is_live:
            return
            
        if not self.camera:
            return
            
        try:
            # Get latest frame from threaded camera
            frame_data = self.camera.get_latest_frame(timeout=0.01)  # Very short timeout for responsive UI
            
            if frame_data is None:
                # No new frame available - this is normal in high-FPS scenarios
                return
            
            # Check if this is actually a new frame
            if frame_data.timestamp <= self.last_frame_timestamp:
                return  # Same frame, don't update
            
            self.last_frame_timestamp = frame_data.timestamp
            self.current_frame_data = frame_data
            
            # Process frame for display
            processed_frame = self.process_frame(frame_data.frame)
            
            # Record frame if recording is active
            self._record_frame(processed_frame)
            
            # Display the frame
            self.display_frame(processed_frame)
            
            # Update status with camera statistics (occasionally)
            if frame_data.frame_number % 60 == 0:  # Update stats every 60 frames
                stats = self.camera.get_statistics()
                if stats['use_test_pattern']:
                    self.update_status(f"Test Pattern - {stats['fps']:.1f} FPS")
                else:
                    self.update_status(f"Live - {stats['fps']:.1f} FPS")
            
        except Exception as e:
            # Log unique errors only
            if str(e) != self.camera_error:
                logger.error(f"Update frame error: {e}")
                self.camera_error = str(e)
                self.update_status("Error")

    def calculate_auto_brightness_adjustment(self, frame):
        """
        Optimized automatic brightness adjustment using subsampling.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            float: Brightness adjustment value
        """
        # Use subsampling for faster analysis (every 4th pixel in each dimension)
        height, width = frame.shape[:2]
        subsampled = frame[::4, ::4]  # 16x faster analysis
        
        # Calculate brightness from green channel only (perceptually most important)
        # This avoids color space conversion overhead
        current_brightness = np.mean(subsampled[:, :, 1])  # Green channel
        
        # Calculate adjustment with rate limiting
        brightness_difference = self.target_brightness - current_brightness
        adjustment = brightness_difference * self.brightness_adjustment_rate
        
        # Limit adjustment to prevent flickering
        return np.clip(adjustment, -20, 20)

    def process_frame(self, frame):
        """
        Optimized frame processing pipeline.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame ready for display
        """
        # Skip processing if no effects are enabled (most common case)
        if not (self.auto_brightness or self.mono_enabled or self.auto_contrast):
            return frame
        
        # Only copy if we need to modify the frame
        img = frame
        
        # Apply automatic brightness adjustment (most expensive operation)
        if self.auto_brightness:
            brightness_adjustment = self.calculate_auto_brightness_adjustment(frame)
            if abs(brightness_adjustment) > 1:  # Only adjust if significant difference
                # Create copy only when needed
                if img is frame:
                    img = frame.copy()
                # Use cv2.add for hardware acceleration where available
                img = cv2.add(img, np.full_like(img, brightness_adjustment, dtype=np.uint8))
        
        # Apply mono conversion (medium cost)
        if self.mono_enabled:
            if img is frame:
                img = frame.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        # Apply auto-contrast (highest cost - skip unless really needed)
        if self.auto_contrast:
            if img is frame:
                img = frame.copy()
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
        return img

    def display_frame(self, frame):
        """
        Optimized frame display with reduced memory allocations.
        
        Args:
            frame: BGR frame to display
        """
        # Convert BGR to RGB in-place to avoid extra memory allocation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        
        # Create QImage directly from the data buffer
        bytes_per_line = 3 * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Get current display size
        display_width = self.display_label.width()
        display_height = self.display_label.height()
        
        # Only scale if display size is valid and different from frame size
        if (display_width > 1 and display_height > 1 and 
            (w != display_width or h != display_height)):
            
            # Use FastTransformation for better performance during live view
            # SmoothTransformation only needed for final/static images
            pix = QPixmap.fromImage(qimg).scaled(
                display_width, display_height, 
                Qt.KeepAspectRatio, Qt.FastTransformation
            )
        else:
            pix = QPixmap.fromImage(qimg)
            
        self.display_label.setPixmap(pix)

    def start_recording(self, file_path, metadata=None):
        """
        Start recording video to the specified HDF5 file path.
        
        Args:
            file_path (str): Full path where the video should be saved (with .hdf5 extension)
            metadata (dict): Optional metadata to include in the recording
            
        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False
        
        if not self.is_running or not self.is_live:
            logger.warning("Cannot start recording - camera not in live mode")
            return False
        
        try:
            # Ensure file path has .hdf5 extension
            if not file_path.lower().endswith('.hdf5'):
                file_path = file_path + '.hdf5'
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Get frame shape from threaded camera
            if self.camera:
                frame_shape = self.camera.frame_shape
            else:
                frame_shape = (480, 640, 3)  # Default shape
            
            # Create HDF5 recorder with no compression for speed
            self.hdf5_recorder = HDF5VideoRecorder(
                file_path=file_path,
                frame_shape=frame_shape,
                fps=25.0,  # Target recording FPS to match live performance
                compression=None  # No compression for maximum speed
            )
            
            # Get camera statistics for metadata
            stats = self.camera.get_statistics() if self.camera else {}
            
            # Start recording with metadata
            recording_metadata = {
                'recording_mode': 'test_pattern' if stats.get('use_test_pattern', False) else 'camera',
                'camera_id': self.camera.camera_id if self.camera else 0,
                'operator': os.getenv('USERNAME', 'Unknown'),
                'system_name': 'AFS_tracking',
                'threaded_capture': True,
                'max_queue_size': self.camera.max_queue_size if self.camera else 0
            }
            
            # Add user-provided metadata
            if metadata:
                recording_metadata.update(metadata)
            
            if not self.hdf5_recorder.start_recording(recording_metadata):
                logger.error("Failed to start HDF5 recording")
                return False
            
            # Extract and save camera settings as metadata
            if self.camera and hasattr(self.camera, 'get_camera_settings'):
                try:
                    camera_settings = self.camera.get_camera_settings()
                    self.hdf5_recorder.add_camera_settings(camera_settings)
                    logger.info(f"Saved camera settings to HDF5: {len(camera_settings)} parameters")
                except Exception as e:
                    logger.warning(f"Failed to save camera settings to HDF5: {e}")
            
            # Extract and save stage settings as metadata
            try:
                from src.controllers.stage_manager import StageManager
                stage_manager = StageManager.get_instance()
                stage_settings = stage_manager.get_stage_settings()
                self.hdf5_recorder.add_stage_settings(stage_settings)
                logger.info(f"Saved stage settings to HDF5: {len(stage_settings)} parameters")
            except Exception as e:
                logger.warning(f"Failed to save stage settings to HDF5: {e}")
            
            # Set recording state
            self.is_recording = True
            self.recording_path = file_path
            self.recording_start_time = datetime.now()
            self.recorded_frames = 0
            self.last_recorded_time = time.time()  # Reset frame rate timer
            
            logger.info(f"Started HDF5 recording to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            if self.hdf5_recorder:
                try:
                    self.hdf5_recorder.stop_recording()
                except:
                    pass
                self.hdf5_recorder = None
            return False

    def stop_recording(self):
        """
        Stop recording video and close the HDF5 file.
        
        Returns:
            str: Path to the saved HDF5 file, or None if recording failed
        """
        if not self.is_recording:
            logger.warning("No recording in progress")
            return None
        
        try:
            # Stop recording
            self.is_recording = False
            
            if self.hdf5_recorder:
                success = self.hdf5_recorder.stop_recording()
                if not success:
                    logger.warning("HDF5 recorder reported failure during stop")
                self.hdf5_recorder = None
            
            # Calculate recording duration
            if self.recording_start_time:
                duration = datetime.now() - self.recording_start_time
                logger.info(f"HDF5 recording stopped. Duration: {duration.total_seconds():.1f}s, "
                           f"Frames: {self.recorded_frames}, Path: {self.recording_path}")
            
            saved_path = self.recording_path
            self.recording_path = ""
            self.recording_start_time = None
            self.recorded_frames = 0
            
            return saved_path
            
        except Exception as e:
            logger.error(f"Error stopping HDF5 recording: {e}")
            return None

    def _record_frame(self, frame):
        """
        Record a frame to the HDF5 file if recording is active.
        Uses frame rate limiting to match target recording FPS.
        
        Args:
            frame: The frame to record (BGR format)
        """
        if not self.is_recording or not self.hdf5_recorder:
            return
        
        # Frame rate limiting - only record at target FPS
        current_time = time.time()
        if current_time - self.last_recorded_time < self.frame_interval:
            return  # Skip this frame to maintain target FPS
        
        try:
            # Ensure frame is the correct size for the recorder
            expected_shape = self.hdf5_recorder.frame_shape
            if frame.shape != expected_shape:
                # Resize frame to match expected shape
                target_height, target_width = expected_shape[:2]
                frame = cv2.resize(frame, (target_width, target_height))
            
            # Record frame to HDF5
            if self.hdf5_recorder.record_frame(frame):
                self.recorded_frames += 1
                self.last_recorded_time = current_time
            else:
                logger.warning(f"Failed to record frame {self.recorded_frames}")
            
        except Exception as e:
            logger.error(f"Error recording frame to HDF5: {e}")

    # Basic keyboard shortcuts for tests
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            if self.is_live:
                self.set_pause_mode()
            else:
                self.set_live_mode()
        super().keyPressEvent(event)