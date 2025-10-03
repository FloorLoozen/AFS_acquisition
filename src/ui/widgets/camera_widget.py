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
        
        # Frame data with performance optimization
        self.current_frame_data: FrameData = None
        self.last_frame_timestamp = 0
        # Removed complex frame control variables
        
        # Performance monitoring
        self.display_fps_counter = 0
        self.display_fps_start_time = time.time()
        self.last_display_fps = 0
        


        # Display + timers
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        
        # Video recording state with robustness
        self.is_recording = False
        self.hdf5_recorder = None
        self.recording_errors = 0
        self.max_recording_errors = 50
        self.recording_path = ""
        self.recording_start_time = None
        self.recorded_frames = 0
        
        # Recording control
        self.target_recording_fps = 25.0
        self.recording_start_timestamp = 0
        

        
        # Maximize camera view - set size policy to strongly expand in both directions
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setMinimumSize(320, 240)  # Set minimum size
        
        # Set the widget to strongly prefer expanding
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # High-frequency timer for frame capture
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frame)
        self.update_timer.setInterval(10)  # 100Hz for optimal frame capture
        
        # Image processing toggles
        self.mono_enabled = False
        self.auto_contrast = False

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





    def update_frame(self):
        """
        Process frames from camera and handle recording.
        """
        if not self.is_running or not self.is_live:
            return
            
        if not self.camera:
            return
            

            
        try:
            # Get latest frame
            frame_data = self.camera.get_latest_frame(timeout=0.001)
            
            if frame_data is None:
                return
            
            self.last_frame_timestamp = frame_data.timestamp
            self.current_frame_data = frame_data
            
            # Record frame if recording
            if self.is_recording:
                self._record_frame(frame_data.frame)
            
            # Always update display (keep it simple)
            processed_frame = self.process_frame(frame_data.frame)
            self.display_frame(processed_frame)
            
            # Update performance monitoring
            self.display_fps_counter += 1
            current_time = time.time()
            if current_time - self.display_fps_start_time >= 2.0:  # Update every 2 seconds
                display_fps = self.display_fps_counter / (current_time - self.display_fps_start_time)
                self.last_display_fps = display_fps
                self.display_fps_counter = 0
                self.display_fps_start_time = current_time
            
            # Update status with performance info (less frequently for better performance)
            if frame_data.frame_number % 150 == 0:  # Update stats every 150 frames (5 seconds at 30fps)
                try:
                    stats = self.camera.get_statistics()
                    display_info = f"Display: {self.last_display_fps:.1f} FPS"
                    if self.is_recording:
                        status_text = f"Recording - {display_info}"
                    elif stats['use_test_pattern']:
                        status_text = f"Test Pattern - {display_info}"
                    else:
                        status_text = f"Live - {display_info}"
                    self.update_status(status_text)
                except:
                    pass  # Don't let stats errors affect display performance
            
        except Exception as e:
            # Log unique errors only
            if str(e) != self.camera_error:
                logger.error(f"Update frame error: {e}")
                self.camera_error = str(e)
                self.update_status("Error")



    def process_frame(self, frame):
        """
        Minimal frame processing optimized for recording performance.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame ready for display
        """
        # Skip processing during recording for maximum performance
        if self.is_recording:
            return frame
            
        # Skip processing if no effects are enabled
        if not (self.mono_enabled or self.auto_contrast):
            return frame
        
        # Apply mono conversion if enabled
        if self.mono_enabled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
        return frame

    def display_frame(self, frame):
        """
        Ultra-optimized frame display for maximum performance.
        
        Args:
            frame: BGR frame to display
        """
        # Skip RGB conversion for test patterns that might already be RGB
        if frame.shape[2] == 3:
            # Convert BGR to RGB - required for proper display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
            
        h, w = rgb_frame.shape[:2]
        
        # Create QImage directly from the data buffer for zero-copy operation
        bytes_per_line = 3 * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Get current display size
        display_width = self.display_label.width()
        display_height = self.display_label.height()
        
        # Only scale if display size is valid and significantly different from frame size
        scale_threshold = 50  # Don't rescale for small differences
        if (display_width > 1 and display_height > 1 and 
            (abs(w - display_width) > scale_threshold or abs(h - display_height) > scale_threshold)):
            
            # Use FastTransformation for maximum performance during live view
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
            
            # Create HDF5 recorder with LZF compression for speed
            self.hdf5_recorder = HDF5VideoRecorder(
                file_path=file_path,
                frame_shape=frame_shape,
                fps=25.0  # Target recording FPS to match live performance
            )
            
            # Get camera statistics for metadata
            stats = self.camera.get_statistics() if self.camera else {}
            
            # Start recording with metadata
            recording_metadata = {
                'operator': os.getenv('USERNAME', 'Unknown'),
                'system_name': 'AFS_tracking'
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
                    # Add image processing settings
                    camera_settings['mono_enabled'] = self.mono_enabled
                    camera_settings['auto_contrast'] = self.auto_contrast
                    
                    self.hdf5_recorder.add_camera_settings(camera_settings)
                    logger.info(f"Saved camera settings to HDF5: {len(camera_settings)} parameters")
                except Exception as e:
                    logger.warning(f"Failed to save camera settings to HDF5: {e}")
            else:
                # Save image processing settings even if camera settings unavailable
                try:
                    image_settings = {
                        'mono_enabled': self.mono_enabled,
                        'auto_contrast': self.auto_contrast
                    }
                    self.hdf5_recorder.add_camera_settings(image_settings)
                    logger.info(f"Saved image processing settings to HDF5: {len(image_settings)} parameters")
                except Exception as e:
                    logger.warning(f"Failed to save image processing settings to HDF5: {e}")
            
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
            self.recording_start_timestamp = time.time()
            self.recorded_frames = 0
            
            logger.info(f"Started HDF5 recording: {file_path}")
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
        Record frame with minimal overhead.
        """
        if not self.is_recording or not self.hdf5_recorder:
            return
        
        try:
            if self.hdf5_recorder.record_frame(frame):
                self.recorded_frames += 1
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.recording_errors += 1

    # Basic keyboard shortcuts for tests
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            if self.is_live:
                self.set_pause_mode()
            else:
                self.set_live_mode()
        super().keyPressEvent(event)