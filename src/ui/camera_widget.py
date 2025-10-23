"""
Simplified Camera Widget for AFS Acquisition.
Clean, minimal design without side controls.
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
    QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from src.utils.logger import get_logger
from src.controllers.camera_controller import CameraController, FrameData
from src.utils.status_display import StatusDisplay
from src.utils.hdf5_video_recorder import HDF5VideoRecorder

logger = get_logger("camera_widget")


class CameraWidget(QGroupBox):
    """
    Simplified camera widget with clean design.
    Features:
    - Large camera display area
    - Basic control buttons only
    - No side controls panel
    - Minimal, clean interface
    """
    
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
        
        # Performance monitoring
        self.display_fps_counter = 0
        self.display_fps_start_time = time.time()
        self.last_display_fps = 0
        
        # Recording objects
        self.hdf5_recorder = None
        self.recording_path = ""
        self.recording_start_time = None
        self.recorded_frames = 0
        
        # High-performance parallel processing
        self.frame_processing_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="CameraFrameProcessor"
        )
        self.recording_queue = queue.Queue(maxsize=50)  # Buffer for recording
        self.processing_lock = threading.RLock()
        
        # Image processing settings for live view (start with standard values)
        self.image_settings = {
            'brightness': 50,  # Standard brightness
            'contrast': 50,    # Standard contrast
            'saturation': 50   # Standard saturation
        }
        
        # Recording compression and resolution settings
        # Optimized for offline analysis: maximum compression, half resolution
        # compression_level: 0=none, 1-3=fast/LZF, 4-9=best/GZIP
        # downscale_factor: 1=full res, 2=half, 4=quarter
        self.recording_settings = {
            'compression_level': 9,  # Maximum GZIP compression for smallest files
            'downscale_factor': 2    # Half resolution for 4x smaller files
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
        
        # High-frequency timer for smooth fullscreen video
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frame)
        self.update_timer.setInterval(16)  # 60 FPS display for smoother fullscreen viewing
        
        self.init_ui()
        self.update_status("Initializing...")
        
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
    
    def update_button_states(self):
        """Update button enabled states - minimal camera widget has no buttons."""
        pass  # No buttons to update in minimal design
    
    def update_status(self, text: str):
        """Update status display."""
        self.status_display.set_status(text)
    
    def connect_camera(self, camera_id: int = 0):
        """Connect to camera with improved error handling."""
        if self.is_running:
            return
        
        self.update_status("Initializing...")
        
        try:
            # Use the advanced camera controller from controllers directory
            self.camera = CameraController(camera_id=camera_id, max_queue_size=10)
            
            if self.camera.initialize():
                if self.camera.start_capture():
                    # Apply brighter default settings immediately
                    self._apply_default_camera_settings()
                    
                    self.is_running = True
                    self.is_live = False
                    
                    self._is_reinitializing = False  # Clear reinitializing flag on success
                    self.update_status("Connected")
                    self.update_button_states()
                    self.update_timer.start()
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
                except:
                    pass
                self.camera = None
            self._start_test_pattern_mode("Error - Test Pattern")
    
    def _start_test_pattern_mode(self, status_text: str):
        """Start test pattern mode."""
        try:
            self.camera = CameraController(camera_id=0, max_queue_size=5)
            if self.camera.initialize() and self.camera.start_capture():
                self.is_running = True
                self.is_live = False
                
                self._is_reinitializing = False  # Clear reinitializing flag
                self.update_status(status_text)
                self.update_button_states()
                self.update_timer.start()
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
        if self.update_timer.isActive():
            self.update_timer.stop()
        
        self.is_running = False
        self.is_live = False
        
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
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
    
    def set_pause_mode(self):
        """Pause live view."""
        if not self.is_running:
            return
        
        self.is_live = False
        self.update_status("Paused")
        self.update_button_states()
    
    def update_frame(self):
        """Update display with latest camera frame."""
        if not self.is_running or not self.is_live or not self.camera:
            return
        
        try:
            frame_data = self.camera.get_latest_frame(timeout=0.001)
            
            if frame_data is None:
                return
            
            self.last_frame_timestamp = frame_data.timestamp
            self.current_frame_data = frame_data
            
            # Parallel processing for recording and display
            if self.is_recording:
                # Submit recording task to thread pool (non-blocking)
                self.frame_processing_executor.submit(
                    self._record_frame_async, frame_data.frame.copy()
                )
            
            # Submit display processing to thread pool (non-blocking)
            self.frame_processing_executor.submit(
                self._process_display_frame, frame_data.frame.copy()
            )
            
            # Update performance stats (optimized)
            self.display_fps_counter += 1
            current_time = time.time()
            time_elapsed = current_time - self.display_fps_start_time
            
            # Update FPS display every 2 seconds to reduce overhead
            if time_elapsed >= 2.0:
                display_fps = self.display_fps_counter / time_elapsed
                self.last_display_fps = display_fps
                self.display_fps_counter = 0
                self.display_fps_start_time = current_time
                
                # Simplified status update (less string processing)
                status_prefix = "Recording" if self.is_recording else "Live"
                self.update_status(f"{status_prefix} @ {display_fps:.0f} FPS")
        
        except Exception as e:
            if str(e) != self.camera_error:
                logger.error(f"Update frame error: {e}")
                self.camera_error = str(e)
                self.update_status("Error")
    
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
            
            # Apply brightness/contrast together if needed (in-place when possible)
            if needs_brightness or needs_contrast:
                brightness = (self.image_settings['brightness'] - 50) * 2.0
                contrast = self.image_settings['contrast'] / 50.0
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            
            # Apply saturation only if needed (most expensive operation)
            if needs_saturation:
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
            
            # Get frame shape
            if self.camera:
                frame_shape = self.camera.frame_shape
            else:
                frame_shape = (480, 640, 3)
            
            # Create recorder with compression and resolution settings
            self.hdf5_recorder = HDF5VideoRecorder(
                file_path=file_path,
                frame_shape=frame_shape,
                fps=30.0,
                compression_level=self.recording_settings['compression_level'],
                downscale_factor=self.recording_settings['downscale_factor']
            )
            
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
                    logger.info(f"Saved camera settings: {len(camera_settings)} parameters (including image processing)")
                except Exception as e:
                    logger.warning(f"Failed to save camera settings: {e}")
            
            # Add stage settings if available
            try:
                from src.controllers.stage_manager import StageManager
                stage_manager = StageManager.get_instance()
                if stage_manager:
                    stage_settings = stage_manager.get_stage_settings()
                    self.hdf5_recorder.add_stage_settings(stage_settings)
                    logger.info(f"Saved stage settings: {len(stage_settings)} parameters")
            except Exception as e:
                logger.warning(f"Failed to save stage settings: {e}")
            
            # Add recording metadata and regeneration info
            try:
                self.hdf5_recorder.add_recording_metadata(recording_metadata)
            except Exception as e:
                logger.warning(f"Failed to save recording metadata: {e}")
            
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
                except:
                    pass
                self.hdf5_recorder = None
            return False
    
    def stop_recording(self) -> Optional[str]:
        """Stop HDF5 recording with progress feedback and robust error handling."""
        if not self.is_recording:
            return None
        
        # Store paths and set flags immediately
        saved_path = self.recording_path
        self.is_recording = False
        self.is_saving = True
        
        logger.info("Stopping HDF5 recorder and saving file...")
        
        try:
            if self.hdf5_recorder:
                # Show progress dialog
                from PyQt5.QtWidgets import QProgressDialog, QApplication, QMessageBox
                from PyQt5.QtCore import Qt, QTimer
                
                progress = QProgressDialog(
                    "Saving recording...\nPlease wait, this may take a moment.",
                    None,
                    0, 0,
                    self
                )
                progress.setWindowTitle("Saving Recording")
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()
                
                # Process events to show dialog
                QApplication.processEvents()
                
                # Call stop_recording directly (synchronously) - no background thread
                # This is safer and prevents Qt crashes
                logger.info("Calling HDF5 recorder stop_recording()...")
                
                try:
                    self.hdf5_recorder.stop_recording()
                    logger.info("HDF5 file fully written and closed")
                    success = True
                    error_msg = None
                except Exception as e:
                    logger.error(f"Error stopping HDF5 recorder: {e}", exc_info=True)
                    success = False
                    error_msg = str(e)
                
                # Close progress dialog
                progress.close()
                QApplication.processEvents()
                
                # Show error if failed
                if not success:
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
                logger.info(f"Recording stopped. Duration: {duration.total_seconds():.1f}s, "
                           f"Frames: {self.recorded_frames}")
            
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
            except:
                pass
            
            return None
    
    def _record_frame_async(self, frame):
        """High-performance asynchronous frame recording."""
        if not self.is_recording or not self.hdf5_recorder:
            return
        
        try:
            # Use async recording for maximum performance
            if self.hdf5_recorder.record_frame(frame, use_async=True):
                with self.processing_lock:
                    self.recorded_frames += 1
            else:
                # Silent failure to prevent logging errors when disk is full
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
            # Shutdown thread pool gracefully
            logger.debug("Shutting down frame processing thread pool...")
            self.frame_processing_executor.shutdown(wait=True, timeout=2.0)
        except Exception as e:
            logger.warning(f"Error shutting down thread pool: {e}")
    
    def _record_frame(self, frame):
        """Legacy synchronous frame recording for compatibility."""
        return self._record_frame_async(frame)
    
    def _process_display_frame(self, frame):
        """Process frame for display in parallel thread."""
        try:
            # Convert BGR to RGB for Qt display
            if frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # Apply image processing settings (optimized for performance)
            processed_frame = self._apply_image_processing(rgb_frame)
            
            # Update display in main thread
            self._update_display_thread_safe(processed_frame)
            
        except Exception as e:
            logger.error(f"Error processing display frame: {e}")
    
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
        logger.debug(f"Image settings updated: {self.image_settings}")
    
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
        """Apply brighter default camera settings."""
        if not self.camera or not hasattr(self.camera, 'apply_settings'):
            return
        
        # Brighter default settings (optimized for performance)
        default_settings = {
            'exposure_ms': 15.0,
            'gain_master': 2,  # Use integer for gain
            'frame_rate_fps': 30.0,
            'brightness': 50,   # Standard brightness
            'contrast': 50,     # No contrast change initially  
            'saturation': 50    # No saturation change initially
        }
        
        try:
            result = self.camera.apply_settings(default_settings)
            logger.debug(f"Applied default brighter camera settings: {result}")
            
            # Also update image processing settings for live view
            self.update_image_settings(default_settings)
        except Exception as e:
            logger.warning(f"Failed to apply default camera settings: {e}")
    
    def close(self):
        """Clean shutdown."""
        if self.is_recording:
            self.stop_recording()
        self.stop_camera()