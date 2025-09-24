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
from src.controllers.camera_controller import CameraController
from src.utils.status_display import StatusDisplay
from src.utils.hdf5_video_recorder import HDF5VideoRecorder

logger = get_logger("camera_gui")



class CameraWidget(QGroupBox):
    """Camera widget with status indicator."""

    def __init__(self, parent=None):
        super().__init__("Camera", parent)

        logger.info("Camera initialized")
        
        # Check for camera module availability immediately
        try:
            # Try importing the module - don't need to use it yet
            from pyueye import ueye
            self.pyueye_available = True
        except ImportError as e:
            logger.warning(f"pyueye module import error: {e}")
            self.pyueye_available = False
        except Exception as e:
            logger.warning(f"Error checking pyueye module: {e}")
            self.pyueye_available = False

        # Camera state
        self.camera = None
        self.is_running = False
        self.is_live = False
        self.use_test_pattern = not self.pyueye_available  # Auto set based on module availability
        self.test_frame_counter = 0
        self.camera_error = None
        
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
        
        # Maximize camera view - set size policy to strongly expand in both directions
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setMinimumSize(320, 240)  # Set minimum size
        
        # Set the widget to strongly prefer expanding
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frame)
        self.update_timer.setInterval(33)  # ~30 FPS instead of ~67 FPS for better performance
        
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
        
        # Right side: Status display
        status_layout = QHBoxLayout()
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
            
        # Try to connect
        try:
            self.camera = CameraController(0)  # Try default camera ID
            if self.camera.initialize():
                # Success!
                self.use_test_pattern = False
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
                
            # If initialization failed, clean up and try again in 300ms
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
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
        
        # If pyueye is not available, use test pattern mode immediately
        if not self.pyueye_available:
            self._start_test_pattern_mode("Test Pattern Mode")
            return
        
        # Quick hardware check - don't spend too much time on startup
        try:
            self.update_status("Connecting...")
            self.camera = CameraController(camera_id)
            
            # Use a shorter timeout for initial connection
            if self.camera.initialize():
                # Camera initialized successfully
                logger.info("Camera connected")
                self.use_test_pattern = False
                self.is_running = True
                self.is_live = False
                self.update_status("Connected")
                self.update_button_states()
                self.update_timer.start()
                self.set_live_mode()  # Auto start live view
                return
            else:
                # Hardware initialization failed, use test pattern
                if self.camera:
                    try:
                        self.camera.close()
                    except:
                        pass
                    self.camera = None
                self._start_test_pattern_mode("Hardware Not Found - Test Pattern")
                return
            
        except Exception as e:
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
                self.camera = None
            self._start_test_pattern_mode("Hardware Error - Test Pattern")
    
    def _start_test_pattern_mode(self, status_text):
        """Helper method to start test pattern mode with the given status text"""
        self.camera = None
        self.use_test_pattern = True
        self.is_running = True
        self.is_live = False
        self.update_status(status_text)
        self.update_button_states()
        self.update_timer.start()
        self.set_live_mode()  # Auto start test pattern

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
        self.update_status("Initialize")
        self.update_button_states()

    def set_live_mode(self):
        if not self.is_running:
            logger.warning("Cannot set live mode - camera not running")
            return
        
        # Handle test pattern mode specifically
        if self.use_test_pattern:
            self.is_live = True
            self.update_status("Test Pattern Active")
            self.update_button_states()
            return
            
        # If camera is not found and not in test pattern mode
        if not self.camera:
            logger.warning("Cannot set live mode - no camera and not in test pattern mode")
            self.update_status("Camera not found")
            return
            
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

    def create_blank_frame(self, width=640, height=480):
        """Create a completely black frame for when camera is not found."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # No text - the status indicator already shows "Camera not found"
        self.display_frame(img)

    def create_test_pattern(self, width=640, height=480):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a grid background pattern
        for i in range(0, height, 50):
            cv2.line(frame, (0, i), (width, i), (50, 50, 50), 1)
        for i in range(0, width, 50):
            cv2.line(frame, (i, 0), (i, height), (50, 50, 50), 1)
        
        # Add crosshairs at the center
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (0, 255, 0), 1)
        cv2.circle(frame, (center_x, center_y), 100, (0, 0, 255), 1)
        
        # Add moving elements for the test pattern
        y_pos = (self.test_frame_counter * 2) % height
        x_pos = (self.test_frame_counter * 3) % width
        
        # Moving horizontal line
        cv2.line(frame, (0, y_pos), (width, y_pos), (0, 0, 255), 2)
        # Moving vertical line
        cv2.line(frame, (x_pos, 0), (x_pos, height), (255, 0, 0), 2)
        
        # Pulsing circle
        radius = 50 + 30 * np.sin(self.test_frame_counter / 30.0)
        cv2.circle(frame, (width // 2, height // 2), int(radius), (0, 255, 255), 2)
        
        # Add text to indicate it's a test pattern
        cv2.putText(frame, "TEST PATTERN MODE", (width // 2 - 120, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        cv2.putText(frame, "No camera hardware detected", (width // 2 - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {self.test_frame_counter}", (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        self.test_frame_counter += 1
        return frame

    def check_camera_state(self):
        """Check camera state and update status if needed."""
        if not self.camera:
            return False
            
        if hasattr(self.camera, 'is_open') and not self.camera.is_open:
            # Camera was previously open but is now closed
            if self.camera_error != "Camera disconnected":
                self.camera_error = "Camera disconnected"
                self.update_status("Initialize")
                logger.warning("Camera disconnected")
            return False
            
        return True
        
    def update_frame(self):
        """
        Update the camera frame display. Called by update_timer.
        
        Handles:
        - Test pattern generation
        - Camera frame capture
        - Error handling with log suppression for repetitive errors
        - FPS calculation
        - Frame processing and display
        """
        if not self.is_running:
            return
            
        try:
            if not self.is_live:
                # Skip updates when not in live mode
                return
                
            if self.use_test_pattern:
                # Generate test pattern instead of using camera
                frame = self.create_test_pattern(640, 480)
            else:
                # Camera mode - check state first
                if not self.camera:
                    return
                    
                # Verify camera is still connected and open
                if not self.check_camera_state():
                    return
                
                # Attempt to get a frame
                try:
                    frame = self.camera.get_frame()
                    if frame is None:
                        # Frame retrieval failed but no exception
                        self.warning_count += 1
                        # Don't return immediately, skip a few more attempts then switch to test pattern
                        if self.warning_count > 10:  # After 10 failed attempts (~330ms)
                            logger.info("Camera hardware error - switching to test pattern")
                            self.use_test_pattern = True
                            self.update_status("Hardware Error - Test Pattern")
                            self.warning_count = 0
                        return
                        
                    # Successfully got a frame - reset error state
                    self.warning_count = 0
                    if self.camera_error:
                        self.camera_error = None
                        self.update_status("Live")
                    
                except Exception as e:
                    # Handle camera error during frame capture
                    error_msg = str(e).lower()
                    self.warning_count += 1
                    
                    # Switch to test pattern after several errors instead of just logging
                    if self.warning_count >= 5:  # After 5 errors (~165ms)
                        if "closed camera" in error_msg or "camera not found" in error_msg:
                            logger.info("Camera disconnected")
                            self.use_test_pattern = True
                            self.update_status("Initialize")
                            self.camera_error = "Camera disconnected"
                        else:
                            self.use_test_pattern = True
                            self.update_status("Hardware Error - Test Pattern")
                            self.camera_error = str(e)
                        self.warning_count = 0
                    return
                    
            frame = self.process_frame(frame)
            
            # Record frame if recording is active
            self._record_frame(frame)
            
            self.display_frame(frame)
            
        except Exception as e:
            # Log unique errors only
            if str(e) != self.camera_error:
                logger.error(f"Update frame error: {e}")
                self.camera_error = str(e)
                self.update_status("Error")

    def calculate_auto_brightness_adjustment(self, frame):
        """
        Calculate automatic brightness adjustment based on image analysis.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            float: Brightness adjustment value
        """
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate current average brightness
        current_brightness = np.mean(gray)
        
        # Calculate how much adjustment is needed
        brightness_difference = self.target_brightness - current_brightness
        
        # Apply adjustment rate to smooth the transition
        adjustment = brightness_difference * self.brightness_adjustment_rate
        
        # Limit the adjustment to prevent over-correction
        max_adjustment = 30  # Maximum brightness change per frame
        adjustment = np.clip(adjustment, -max_adjustment, max_adjustment)
        
        return adjustment

    def process_frame(self, frame):
        img = frame.copy()
        
        # Apply automatic brightness adjustment
        if self.auto_brightness:
            brightness_adjustment = self.calculate_auto_brightness_adjustment(img)
            if abs(brightness_adjustment) > 1:  # Only adjust if significant difference
                # Convert to float for calculations to avoid overflow/underflow
                img = img.astype(np.float32)
                img = img + brightness_adjustment
                # Clip values to valid range [0, 255] and convert back to uint8
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Apply mono or auto-contrast if set programmatically
        if self.mono_enabled:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self.auto_contrast:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
        return img

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        
        # Scale to fill the available space while keeping aspect ratio
        if self.display_label.width() > 1 and self.display_label.height() > 1:
            # Use KeepAspectRatio to ensure the image isn't distorted
            pix = pix.scaled(self.display_label.width(), self.display_label.height(), 
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
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
            
            # Determine frame shape from current camera or test pattern
            if self.use_test_pattern:
                frame_shape = (480, 640, 3)  # Test pattern dimensions
            else:
                # Get frame shape from camera (convert ctypes to int)
                if self.camera:
                    height = int(self.camera.height)
                    width = int(self.camera.width)
                    frame_shape = (height, width, 3)
                else:
                    frame_shape = (480, 640, 3)
            
            # Create HDF5 recorder
            self.hdf5_recorder = HDF5VideoRecorder(
                file_path=file_path,
                frame_shape=frame_shape,
                fps=60.0,  # Set to 60 Hz as requested
                compression='lzf'  # Fast compression for random access
            )
            
            # Start recording with metadata
            recording_metadata = {
                'recording_mode': 'test_pattern' if self.use_test_pattern else 'camera',
                'camera_id': getattr(self.camera, 'camera_id', 0) if self.camera else 0,
                'operator': os.getenv('USERNAME', 'Unknown'),
                'system_name': 'AFS_tracking'
            }
            
            # Add user-provided metadata
            if metadata:
                recording_metadata.update(metadata)
            
            if not self.hdf5_recorder.start_recording(recording_metadata):
                logger.error("Failed to start HDF5 recording")
                return False
            
            # Set recording state
            self.is_recording = True
            self.recording_path = file_path
            self.recording_start_time = datetime.now()
            self.recorded_frames = 0
            
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
        
        Args:
            frame: The frame to record (BGR format)
        """
        if not self.is_recording or not self.hdf5_recorder:
            return
        
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