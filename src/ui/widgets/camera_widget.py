import time
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QSizePolicy, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter

# Fix imports to work with both direct run and module run

from src.logger import get_logger
from src.controllers.camera_controller import CameraController

logger = get_logger("camera_gui")


class StatusIndicator(QLabel):
    """
    Circle indicator that changes color based on status.
    Colors:
    - Green: Live, Connected
    - Orange: Initializing, Reconnecting
    - Yellow: Paused
    - Red: Disconnected, Camera not found, Reconnect failed
    - Gray: Unknown status
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.status_color = QColor(128, 128, 128)  # Default gray
        
    def set_status(self, status):
        if status == "Live":
            self.status_color = QColor(0, 255, 0)  # Green
        elif status == "Initializing..." or status.startswith("Reconnecting"):
            self.status_color = QColor(255, 165, 0)  # Orange
        elif status == "Connected":
            self.status_color = QColor(0, 255, 0)  # Green
        elif status == "Paused":
            self.status_color = QColor(255, 255, 0)  # Yellow
        elif status == "Disconnected" or status == "Camera not found" or status == "Reconnect failed":
            self.status_color = QColor(255, 0, 0)  # Red
        else:
            self.status_color = QColor(128, 128, 128)  # Gray
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.status_color)
        painter.setPen(Qt.black)
        painter.drawEllipse(2, 2, 12, 12)
        painter.end()


class CameraWidget(QGroupBox):
    """Camera widget with status indicator."""

    def __init__(self, parent=None):
        super().__init__("Camera", parent)

        # Camera state
        self.camera = None
        self.is_running = False
        self.is_live = False
        self.use_test_pattern = False
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
        
        # Maximize camera view - set size policy to strongly expand in both directions
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setMinimumSize(320, 240)  # Set minimum size
        
        # Set the widget to strongly prefer expanding
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frame)
        self.update_timer.setInterval(15)
        
        # Image processing toggles (keep for internal use)
        self.mono_enabled = False
        self.auto_contrast = False

        # FPS
        self.last_frame_time = 0.0
        self.fps = 0.0
        
        # Status indicator
        self.status_indicator = StatusIndicator()

        self.init_ui()

        # Auto-connect shortly after startup
        QTimer.singleShot(300, self.connect_camera)

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
        self.play_button = QPushButton("▶ Play"); self.play_button.clicked.connect(self.set_live_mode)
        self.pause_button = QPushButton("⏸ Pause"); self.pause_button.clicked.connect(self.set_pause_mode)
        self.reconnect_button = QPushButton("🔄 Reconnect"); self.reconnect_button.clicked.connect(self.reconnect_camera)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reconnect_button)
        button_layout.addStretch(1)
        
        # Right side: Status indicators
        status_layout = QHBoxLayout()
        status_layout.addStretch(1)
        self.status_label = QLabel("Initializing...")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_indicator)
        
        # Add both sides to bottom row
        bottom_row.addLayout(button_layout, 3)  # Buttons take 3/4 of space
        bottom_row.addLayout(status_layout, 1)  # Status takes 1/4 of space
        
        # Add bottom row to camera layout
        camera_layout.addLayout(bottom_row)
        
        # Add camera frame to main layout
        main.addWidget(camera_frame)

        self.update_button_states()
        self.create_placeholder()

    def update_button_states(self):
        self.play_button.setEnabled(self.is_running and not self.is_live and self.camera is not None)
        self.pause_button.setEnabled(self.is_running and self.is_live)
        # Reconnect button is always enabled as long as we're running
        self.reconnect_button.setEnabled(self.is_running)
        
    def try_reconnect(self):
        """
        Try to reconnect the camera with a 10-second timeout.
        Shows countdown in the status indicator and gives up after timeout.
        """
        # Check if we've exceeded the 10-second timeout
        elapsed_time = time.time() - self.reconnection_start_time
        self.reconnection_attempts += 1
        
        # Calculate remaining time and update status with countdown
        remaining_time = max(0, int(10.0 - elapsed_time))
        self.update_status(f"Reconnecting ({remaining_time}s)...")
        
        if elapsed_time > 10.0:  # 10-second timeout
            # Give up after 10 seconds
            logger.warning(f"Camera reconnection timeout after {self.reconnection_attempts} attempts")
            self.update_status("Reconnect failed")
            self.create_blank_frame()
            # Re-enable reconnect button
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
                self.update_timer.start()
                self.set_live_mode()  # Auto start live view
                logger.info(f"Camera reconnected successfully after {self.reconnection_attempts} attempts")
                # Re-enable reconnect button
                self.reconnect_button.setEnabled(True)
                return
                
            # If initialization failed, try again in 500ms (up to 10 seconds total)
            logger.info(f"Reconnection attempt {self.reconnection_attempts} failed, retrying...")
            self.camera = None  # Clear failed connection attempt
            QTimer.singleShot(500, self.try_reconnect)
            
        except Exception as e:
            logger.error(f"Camera reconnect error: {e}")
            self.camera = None
            QTimer.singleShot(500, self.try_reconnect)
        
    def reconnect_camera(self):
        """Attempt to reconnect to the camera for up to 10 seconds."""
        logger.info("Attempting to reconnect camera...")
        self.update_status("Reconnecting...")
        
        # First disconnect if already connected
        if self.camera:
            self.stop_camera()
            
        # Disable reconnect button during reconnection attempt
        self.reconnect_button.setEnabled(False)
        
        # Start a reconnection attempt with timeout
        self.reconnection_start_time = time.time()
        self.reconnection_attempts = 0
        self.try_reconnect()

    def update_status(self, text):
        self.status_label.setText(text)
        self.status_indicator.set_status(text)

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
        try:
            self.camera = CameraController(camera_id)
            if not self.camera.initialize():
                logger.error("Camera initialization failed")
                # Handle failed initialization
                self.camera = None
                self.use_test_pattern = False
                self.is_running = True
                self.is_live = False
                self.update_status("Camera not found")
                self.update_button_states()
                self.create_blank_frame()
                self.update_timer.start()
                return
                
            # Camera initialized successfully
            self.use_test_pattern = False
            self.is_running = True
            self.is_live = False
            self.update_status("Connected")
            self.update_button_states()
            self.update_timer.start()
            self.set_live_mode()  # Auto start live view
            
        except Exception as e:
            logger.error(f"Camera connect failed: {e}")
            # Handle connection exception
            self.camera = None
            self.use_test_pattern = False
            self.is_running = True
            self.is_live = False
            self.update_status("Camera not found")
            self.update_button_states()
            self.create_blank_frame()
            self.update_timer.start()

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
        self.update_status("Disconnected")
        self.update_button_states()

    def set_live_mode(self):
        if not self.is_running:
            return
            
        # If camera is not found, don't try to go to live mode
        if not self.camera and not self.use_test_pattern:
            self.update_status("Camera not found")
            return
            
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

    def create_placeholder(self, width=640, height=480):
        img = np.ones((height, width, 3), dtype=np.uint8) * 64
        self.display_frame(img)
        
    def create_blank_frame(self, width=640, height=480):
        """Create a completely black frame for when camera is not found."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # No text - the status indicator already shows "Camera not found"
        self.display_frame(img)

    def create_test_pattern(self, width=640, height=480):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a background pattern
        frame[::10, ::10] = [32, 32, 32]  # Subtle dotted pattern
        
        # Add moving elements for the test pattern
        y_pos = (self.test_frame_counter * 5) % height
        cv2.line(frame, (0, y_pos), (width, y_pos), (0, 0, 255), 2)
        cv2.circle(frame, (width // 2, height // 2), 50 + (self.test_frame_counter % 60), (255, 0, 0), 2)
        
        # Add text to indicate it's a test pattern
        cv2.putText(frame, "TEST PATTERN", (width // 2 - 80, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        
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
                self.update_status("Disconnected")
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
                        if self.warning_count == 1:  # Only log first occurrence
                            logger.info("No frame available from camera")
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
                    
                    # Only log the first occurrence of each error type
                    if self.warning_count == 1:
                        if "closed camera" in error_msg:
                            self.update_status("Disconnected")
                            self.camera_error = "Camera closed"
                        elif "camera not found" in error_msg:
                            self.update_status("Camera not found")
                            self.camera_error = "Camera not found"
                        else:
                            logger.warning(f"Camera error: {e}")
                            self.update_status("Error")
                            self.camera_error = str(e)
                    return
                    
            # Calculate FPS
            now = time.time()
            if self.last_frame_time:
                dt = max(now - self.last_frame_time, 1e-6)
                self.fps = 1.0 / dt
            self.last_frame_time = now

            frame = self.process_frame(frame)
            cv2.putText(frame, f"{self.fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.display_frame(frame)
            
        except Exception as e:
            # Log unique errors only
            if str(e) != self.camera_error:
                logger.error(f"Update frame error: {e}")
                self.camera_error = str(e)
                self.update_status("Error")

    def process_frame(self, frame):
        img = frame.copy()
        
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

    # Basic keyboard shortcuts for tests
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            if self.is_live:
                self.set_pause_mode()
            else:
                self.set_live_mode()
        super().keyPressEvent(event)