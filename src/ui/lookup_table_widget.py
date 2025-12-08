"""
Lookup Table Generator Widget for Z-calibration.

Creates a lookup table by capturing diffraction patterns at different Z-positions.
This is used for 3D particle tracking by correlating diffraction patterns with
Z-position of the objective.

Reference: https://pubmed.ncbi.nlm.nih.gov/25419961/
"""

import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QProgressBar, QTextEdit, QFileDialog,
    QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from src.controllers.stage_manager import StageManager
from src.controllers.camera_controller import CameraController
from src.utils.logger import get_logger
from src.utils.status_display import StatusDisplay
from src.utils.hdf5_video_recorder import HDF5VideoRecorder

logger = get_logger("lut_generator")


class LUTAcquisitionThread(QThread):
    """Background thread for LUT acquisition to keep UI responsive."""
    
    progress = pyqtSignal(int, int, str)  # current, total, status message
    finished = pyqtSignal(bool, str)  # success, message
    frame_captured = pyqtSignal(int, float)  # frame_number, z_position
    
    def __init__(self, start_um: float, end_um: float, step_nm: float, 
                 camera, hdf5_recorder, settle_time_ms: int, camera_widget=None, output_path=None):
        super().__init__()
        self.start_um = start_um
        self.end_um = end_um
        self.step_nm = step_nm
        self.camera = camera
        self.hdf5_recorder = hdf5_recorder
        self.settle_time_s = settle_time_ms / 1000.0
        self._stop_requested = False
        self.camera_widget = camera_widget
        self.output_path = output_path  # For standalone mode without recorder
        
    def request_stop(self):
        """Request the acquisition to stop."""
        self._stop_requested = True
        
    def run(self):
        """Execute the LUT acquisition."""
        from PyQt5.QtCore import QMetaObject, Qt, QTimer
        stage_manager = StageManager.get_instance()
        
        try:
            # Pause live view to save processing power during acquisition
            if self.camera_widget and hasattr(self.camera_widget, 'pause_live') and callable(getattr(self.camera_widget, 'pause_live')):
                # Schedule pause_live on the main thread safely. Some CameraWidget implementations
                # don't expose pause_live as a Qt slot, so QMetaObject.invokeMethod fails.
                # Using QTimer.singleShot(0, ...) posts the call to the GUI event loop.
                try:
                    QTimer.singleShot(0, self.camera_widget.pause_live)
                    logger.info("Pausing live view during LUT acquisition")
                except Exception:
                    # Fall back to direct call if scheduling fails
                    try:
                        self.camera_widget.pause_live()
                        logger.info("Pausing live view (direct call) during LUT acquisition")
                    except Exception:
                        logger.debug("Could not pause live view")
            
            # Connect to Z-stage if needed
            if not stage_manager.z_is_connected:
                self.progress.emit(0, 100, "Connecting to Z-stage...")
                if not stage_manager.connect_z():
                    self.finished.emit(False, "Failed to connect to Z-stage")
                    return
            
            # Camera is already initialized and running
            if not self.camera:
                self.finished.emit(False, "No camera available")
                return
            
            # Calculate Z positions
            step_um = self.step_nm / 1000.0  # Convert nm to µm
            num_positions = int((self.end_um - self.start_um) / step_um) + 1
            z_positions = [self.start_um + i * step_um for i in range(num_positions)]
            
            self.progress.emit(0, num_positions, 
                             f"Acquiring {num_positions} frames from {self.start_um:.0f} to {self.end_um:.0f} µm...")
            
            # Move to start position
            self.progress.emit(0, num_positions, f"Moving to start position: {self.start_um:.0f} µm...")
            stage_manager.move_z_to(self.start_um)
            time.sleep(self.settle_time_s)
            
            # Acquire frames at each Z position with optimized pipelining
            lut_frames = []
            lut_z_positions = []
            
            # Parallel strategy: overlap stage movement with frame capture
            # While capturing frame N, start moving to position N+1
            next_z_target = None
            move_start_time = None
            
            for i, z_pos_target in enumerate(z_positions):
                if self._stop_requested:
                    self.progress.emit(i, num_positions, "Stopping...")
                    break
                
                # If we pre-started movement to this position, just wait for settle
                if next_z_target == z_pos_target and i > 0:
                    # Movement already started, calculate remaining settle time
                    elapsed = time.time() - move_start_time
                    remaining_settle = max(0, self.settle_time_s - elapsed)
                    if remaining_settle > 0:
                        time.sleep(remaining_settle)
                else:
                    # Move to Z position (first iteration or catch-up)
                    stage_manager.move_z_to(z_pos_target)
                    time.sleep(self.settle_time_s)
                
                # Read back actual Z position from stage after settling
                actual_z_pos = stage_manager.get_z_position()
                if actual_z_pos is None:
                    logger.warning(f"Failed to read Z position at target {z_pos_target:.0f} µm, using target value")
                    actual_z_pos = z_pos_target
                
                # Capture frame BEFORE starting movement (stage must be stationary!)
                frame = self.camera.get_frame(timeout=1.0)
                if frame is None:
                    logger.warning(f"Failed to capture frame at Z={actual_z_pos:.0f} µm")
                    continue
                
                # Collect frame for LUT with actual Z position
                lut_frames.append(frame)
                lut_z_positions.append(actual_z_pos)
                
                # NOW start moving to NEXT position (after capture is complete)
                if i + 1 < num_positions:
                    next_z_target = z_positions[i + 1]
                    stage_manager.move_z_to(next_z_target)  # Non-blocking command
                    move_start_time = time.time()  # Track when movement started
                else:
                    next_z_target = None
                
                # Emit progress with actual position
                self.frame_captured.emit(i, actual_z_pos)
                self.progress.emit(i + 1, num_positions, 
                                 f"Captured frame {i+1}/{num_positions} at Z={actual_z_pos:.3f} µm")
            
            # Save LUT data to HDF5 file
            if lut_frames:
                # Create recorder if we don't have one (standalone mode)
                recorder_to_use = self.hdf5_recorder
                created_recorder = False
                
                if not recorder_to_use and self.output_path:
                    # Get session file from main window if available
                    session_file = self.output_path
                    try:
                        from PyQt5.QtWidgets import QApplication
                        app = QApplication.instance()
                        if app:
                            main_windows = [w for w in app.topLevelWidgets() if hasattr(w, 'get_session_hdf5_file')]
                            if main_windows:
                                main_window = main_windows[0]
                                session_file_from_main = main_window.get_session_hdf5_file()
                                if session_file_from_main:
                                    session_file = session_file_from_main
                                    logger.info(f"Using session HDF5 file for LUT: {session_file}")
                    except Exception as e:
                        logger.debug(f"Could not get session file: {e}")
                    
                    # Create temporary recorder for LUT file
                    from src.utils.hdf5_video_recorder import HDF5VideoRecorder
                    test_frame = lut_frames[0]
                    recorder_to_use = HDF5VideoRecorder(
                        session_file,
                        frame_shape=test_frame.shape,
                        compression_level=1
                    )
                    # Initialize file for LUT-only mode
                    if recorder_to_use._create_hdf5_file():
                        if 'raw_data' not in recorder_to_use.h5_file:
                            recorder_to_use.h5_file.create_group('raw_data')
                        recorder_to_use._create_execution_data_group()
                        recorder_to_use.is_recording = True
                        created_recorder = True
                        logger.info(f"Opened HDF5 file for LUT: {session_file}")
                    else:
                        recorder_to_use = None
                
                if recorder_to_use:
                    metadata = {
                        'start_position_um': self.start_um,
                        'end_position_um': self.end_um,
                        'step_size_nm': self.step_nm,
                        'settle_time_ms': self.settle_time_s * 1000,
                        'num_positions': len(lut_frames)
                    }
                    recorder_to_use.add_lut_data(lut_frames, lut_z_positions, metadata, optimize_for='max_compression')
                    logger.info(f"LUT data saved: {len(lut_frames)} frames")
                    
                    # Mark main window that LUT was saved
                    if created_recorder:
                        try:
                            from PyQt5.QtWidgets import QApplication
                            app = QApplication.instance()
                            if app:
                                main_windows = [w for w in app.topLevelWidgets() if hasattr(w, 'mark_session_has_lut')]
                                if main_windows:
                                    main_window = main_windows[0]
                                    main_window.mark_session_has_lut()
                                    logger.info("Marked session as containing LUT")
                        except Exception as e:
                            logger.debug(f"Could not mark session has LUT: {e}")
                    
                    # Close standalone recorder
                    if created_recorder:
                        recorder_to_use.h5_file.flush()
                        recorder_to_use.h5_file.close()
                        recorder_to_use.is_recording = False
                        logger.info("Closed standalone LUT file")
                    
                    # If acquisition ran against an active recorder, let camera_widget know the LUT file
                    try:
                        if self.camera_widget and hasattr(recorder_to_use, 'file_path'):
                            setattr(self.camera_widget, 'last_lut_file', str(recorder_to_use.file_path))
                    except Exception:
                        pass
            
            # Return Z-stage to 0 position after LUT acquisition
            try:
                self.progress.emit(num_positions, num_positions, "Returning Z-stage to 0...")
                stage_manager.move_z_to(0.0)
                # Wait a bit for stage to settle
                time.sleep(0.2)
                # Verify position
                actual_pos = stage_manager.get_z_position()
                logger.info(f"Z-stage returned to 0 µm (actual position: {actual_pos:.0f} µm)")
                if abs(actual_pos) > 0.5:
                    logger.warning(f"Z-stage may not have fully returned to 0 (at {actual_pos:.0f} µm)")
            except Exception as e:
                logger.warning(f"Failed to return Z-stage to 0: {e}")
            
            if self._stop_requested:
                self.finished.emit(False, "Acquisition stopped by user")
            else:
                self.finished.emit(True, f"Successfully acquired {len(lut_frames)} frames")
                
        except Exception as e:
            logger.error(f"Error during LUT acquisition: {e}", exc_info=True)
            self.finished.emit(False, f"Error: {str(e)}")


class LUTAcquisitionThreadStandalone(QThread):
    """Background thread for LUT acquisition to keep UI responsive."""
    
    progress = pyqtSignal(int, int, str)  # current, total, status message
    finished = pyqtSignal(bool, str)  # success, message
    frame_captured = pyqtSignal(int, float)  # frame_number, z_position
    
    def __init__(self, start_um: float, end_um: float, step_nm: float, 
                 output_path: str, settle_time_ms: int):
        super().__init__()
        self.start_um = start_um
        self.end_um = end_um
        self.step_nm = step_nm
        self.output_path = output_path
        self.settle_time_s = settle_time_ms / 1000.0
        self._stop_requested = False
        
    def request_stop(self):
        """Request the acquisition to stop."""
        self._stop_requested = True
        
    def run(self):
        """Execute the LUT acquisition."""
        stage_manager = StageManager.get_instance()
        camera = None
        recorder = None
        
        try:
            # Connect to Z-stage if not already connected
            if not stage_manager.z_is_connected:
                self.progress.emit(0, 100, "Connecting to Z-stage...")
                if not stage_manager.connect_z():
                    self.finished.emit(False, "Failed to connect to Z-stage")
                    return
            
            # Initialize camera
            self.progress.emit(0, 100, "Initializing camera...")
            camera = CameraController()
            if not camera.initialize():
                self.finished.emit(False, "Failed to initialize camera")
                return
            
            # Start camera capture
            if not camera.start_capture():
                self.finished.emit(False, "Failed to start camera capture")
                return
            
            # Configure camera for LUT acquisition (same settings as recording)
            # Use full resolution MONO8 format to match recording data
            self.progress.emit(0, 100, "Configuring camera for LUT acquisition...")
            try:
                lut_camera_settings = {
                    'exposure_ms': 5.0,  # 5ms exposure (same as recording)
                    'gain_master': 2,     # Gain 2 (same as recording)
                    'fps': 30.0           # 30 FPS (same as recording)
                }
                camera.apply_settings(lut_camera_settings)
                logger.info("Applied LUT camera settings: exposure=5ms, gain=2, fps=30 (matching recording format)")
                time.sleep(0.3)  # Wait for camera to stabilize
            except Exception as e:
                logger.warning(f"Failed to apply LUT camera settings: {e}")
            
            # Get a test frame to determine dimensions
            test_frame = camera.get_frame(timeout=1.0)
            if test_frame is None:
                self.finished.emit(False, "Failed to capture test frame")
                return
            
            # Calculate Z positions
            step_um = self.step_nm / 1000.0  # Convert nm to µm
            num_positions = int((self.end_um - self.start_um) / step_um) + 1
            z_positions = [self.start_um + i * step_um for i in range(num_positions)]
            
            self.progress.emit(0, num_positions, 
                             f"Acquiring {num_positions} frames from {self.start_um:.0f} to {self.end_um:.0f} µm...")
            
            # Prepare metadata before creating recorder
            metadata = {
                'acquisition_type': 'lookup_table',
                'start_position_um': self.start_um,
                'end_position_um': self.end_um,
                'step_size_nm': self.step_nm,
                'step_size_um': step_um,
                'num_positions': num_positions,
                'settle_time_ms': self.settle_time_s * 1000,
                'timestamp': datetime.now().isoformat(),
                'camera_id': camera.camera_id,
                'frame_width': test_frame.shape[1],
                'frame_height': test_frame.shape[0]
            }
            
            # Get session HDF5 file from main window
            session_file = None
            main_window = None
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                main_windows = [w for w in app.topLevelWidgets() if hasattr(w, 'get_session_hdf5_file')]
                if main_windows:
                    main_window = main_windows[0]
                    session_file = main_window.get_session_hdf5_file()
                    if session_file:
                        self.output_path = session_file
                        logger.info(f"Using session HDF5 file for LUT: {session_file}")
                        
                        # Check if LUT already exists in session - ask user what to do
                        if main_window.session_has_lut:
                            from PyQt5.QtWidgets import QMessageBox
                            msg_box = QMessageBox(None)
                            msg_box.setIcon(QMessageBox.Question)
                            msg_box.setWindowTitle("LUT Already Exists")
                            msg_box.setText("The current session already contains LUT data.")
                            msg_box.setInformativeText("What would you like to do?")
                            
                            overwrite_btn = msg_box.addButton("Overwrite", QMessageBox.AcceptRole)
                            new_session_btn = msg_box.addButton("New Session", QMessageBox.DestructiveRole)
                            cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
                            msg_box.setDefaultButton(cancel_btn)
                            
                            msg_box.exec_()
                            clicked = msg_box.clickedButton()
                            
                            if clicked == cancel_btn:
                                self.finished.emit(False, "LUT acquisition cancelled")
                                return
                            elif clicked == new_session_btn:
                                # Ask main window to create new session
                                if hasattr(main_window, '_new_session_file'):
                                    main_window._new_session_file()
                                    # Get the new session file
                                    session_file = main_window.get_session_hdf5_file()
                                    self.output_path = session_file
                                    logger.info(f"Using new session file: {session_file}")
                            # else: overwrite_btn clicked = Overwrite (continue with current session)
                        
                        # Mark main window that session will have LUT (tentatively)
                        # We'll unmark it in finally block if acquisition was stopped
                        main_window.mark_session_has_lut()
                        # Store reference for cleanup
                        self.main_window_ref = main_window
            except Exception as e:
                logger.debug(f"Could not get session file: {e}")
            
            # Create HDF5 recorder for LUT storage
            # If we have a session file, use it; otherwise use the provided output_path
            recorder = HDF5VideoRecorder(
                self.output_path,
                frame_shape=test_frame.shape,
                compression_level=1  # LZF for any incidental main_video dataset (fast)
            )

            # Initialize file for LUT-only mode (no recording session created)
            # This opens/creates the file and sets up basic structure without creating a recording
            try:
                if not recorder._create_hdf5_file():
                    raise Exception("Failed to create/open HDF5 file")
                
                # Create /raw_data group if it doesn't exist
                if 'raw_data' not in recorder.h5_file:
                    recorder.h5_file.create_group('raw_data')
                
                # Create LUT group
                recorder._create_execution_data_group()
                
                # Set flags to allow add_lut_data to work
                recorder.is_recording = True  # Temporary flag to allow LUT saving
                
                logger.info(f"Opened HDF5 file for LUT: {self.output_path}")
            except Exception as e:
                logger.error(f"Failed to initialize HDF5 file for LUT: {e}")
                raise
            
            # Move to start position
            self.progress.emit(0, num_positions, f"Moving to start position: {self.start_um:.0f} µm...")
            stage_manager.move_z_to(self.start_um)
            time.sleep(self.settle_time_s)
            
            # Prepare in-memory buffers for LUT frames and positions
            lut_frames = []
            lut_z_positions = []

            # Acquire frames at each Z position with optimized pipelining
            # Parallel strategy: overlap stage movement with frame capture and HDF5 writing
            next_z_target = None
            move_start_time = None
            
            for i, z_pos_target in enumerate(z_positions):
                if self._stop_requested:
                    self.progress.emit(i, num_positions, "Stopping...")
                    break
                
                # If we pre-started movement to this position, just wait for remaining settle time
                if next_z_target == z_pos_target and i > 0:
                    # Movement already started, calculate remaining settle time
                    elapsed = time.time() - move_start_time
                    remaining_settle = max(0, self.settle_time_s - elapsed)
                    if remaining_settle > 0:
                        time.sleep(remaining_settle)
                else:
                    # Move to Z position (first iteration)
                    stage_manager.move_z_to(z_pos_target)
                    time.sleep(self.settle_time_s)
                
                # Read back actual Z position from stage after settling
                actual_z_pos = stage_manager.get_z_position()
                if actual_z_pos is None:
                    logger.warning(f"Failed to read Z position at target {z_pos_target:.0f} µm, using target value")
                    actual_z_pos = z_pos_target
                
                # Capture frame BEFORE starting movement (stage must be stationary!)
                frame = camera.get_frame(timeout=1.0)
                if frame is None:
                    logger.warning(f"Failed to capture frame at Z={actual_z_pos:.0f} µm")
                    continue
                
                # Collect frame in memory to later store as LUT (we avoid writing main_video)
                # This keeps LUT storage consistent and allows very fast LZF compression.
                lut_frames.append(frame)
                lut_z_positions.append(actual_z_pos)
                
                # NOW start moving to NEXT position (after capture is complete)
                if i + 1 < num_positions:
                    next_z_target = z_positions[i + 1]
                    stage_manager.move_z_to(next_z_target)  # Non-blocking command
                    move_start_time = time.time()  # Track when movement started
                else:
                    next_z_target = None
                
                # Emit progress
                self.frame_captured.emit(i, actual_z_pos)
                self.progress.emit(i + 1, num_positions, 
                                 f"Captured frame {i+1}/{num_positions} at Z={actual_z_pos:.3f} µm")
            
            # Only save LUT data if acquisition completed successfully (not stopped)
            if not self._stop_requested and recorder and lut_frames:
                try:
                    self.progress.emit(num_positions, num_positions, "Saving LUT frames into HDF5 (max compression)...")
                    metadata.update({'saved_at': datetime.now().isoformat()})
                    # Store LUT with maximum final compression (gzip-9) since standalone LUTs are accessed offline
                    recorder.add_lut_data(lut_frames, lut_z_positions, metadata, optimize_for='max_compression')
                    logger.info(f"Standalone LUT data saved: {len(lut_frames)} frames (max compression)")
                    self.progress.emit(num_positions, num_positions, "LUT frames saved")
                except Exception as e:
                    logger.error(f"Failed to save LUT frames: {e}")
            elif self._stop_requested:
                logger.info(f"LUT acquisition stopped - discarding {len(lut_frames)} partial frames")
                self.progress.emit(num_positions, num_positions, "Acquisition stopped - partial data discarded")

            # Return Z-stage to 0 position after LUT acquisition
            try:
                self.progress.emit(num_positions, num_positions, "Returning Z-stage to 0...")
                stage_manager.move_z_to(0.0)
                # Wait a bit for stage to settle
                time.sleep(0.3)
                # Verify position
                actual_pos = stage_manager.get_z_position()
                logger.info(f"Z-stage returned to 0 µm (actual position: {actual_pos:.0f} µm)")
                if abs(actual_pos) > 0.5:
                    logger.warning(f"Z-stage may not have fully returned to 0 (at {actual_pos:.0f} µm)")
            except Exception as e:
                logger.warning(f"Failed to return Z-stage to 0: {e}")
            
            if self._stop_requested:
                self.finished.emit(False, "Acquisition stopped by user")
            else:
                self.finished.emit(True, f"Successfully acquired {num_positions} frames")
                
        except Exception as e:
            logger.error(f"Error during LUT acquisition: {e}", exc_info=True)
            self.finished.emit(False, f"Error: {str(e)}")
            
        finally:
            # If acquisition was stopped, unmark the session flag
            if self._stop_requested:
                if hasattr(self, 'main_window_ref') and self.main_window_ref:
                    self.main_window_ref.session_has_lut = False
                    logger.info("Cleared session_has_lut flag due to stopped acquisition")
            
            # Cleanup
            if recorder:
                try:
                    # Only save if acquisition completed successfully
                    if not self._stop_requested:
                        self.progress.emit(num_positions, num_positions, "Saving LUT data to HDF5 (compression)...")
                        # Close HDF5 file properly (don't call stop_recording - we never started a recording)
                        if hasattr(recorder, 'h5_file') and recorder.h5_file:
                            recorder.h5_file.flush()
                            recorder.h5_file.close()
                            logger.info("HDF5 file closed successfully after LUT save")
                        recorder.is_recording = False
                        self.progress.emit(num_positions, num_positions, "LUT data saved successfully")
                        # Notify parent widget (if present) about the saved LUT file for session reuse
                        try:
                            if hasattr(self, 'parent_widget') and getattr(self, 'parent_widget'):
                                saved_path = str(getattr(recorder, 'file_path', self.output_path))
                                setattr(self.parent_widget, 'last_lut_file', saved_path)
                                # Also propagate to camera_widget if available on the parent
                                try:
                                    pw = self.parent_widget
                                    if hasattr(pw, 'camera_widget') and getattr(pw, 'camera_widget'):
                                        setattr(pw.camera_widget, 'last_lut_file', saved_path)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    else:
                        # Acquisition was stopped - close file without additional processing
                        logger.info("Closing HDF5 file (partial LUT already discarded)")
                        if hasattr(recorder, 'h5_file') and recorder.h5_file:
                            recorder.h5_file.close()
                except Exception as e:
                    logger.error(f"Error stopping recorder: {e}")
                    
            if camera:
                try:
                    camera.stop_capture()
                    camera.close()
                except Exception as e:
                    logger.error(f"Error closing camera: {e}")


class LookupTableWidget(QDialog):
    """
    Widget for generating lookup tables for Z-position calibration.
    
    Captures a series of diffraction patterns at different Z positions
    to create a reference library for 3D particle tracking.
    """
    
    def __init__(self, camera=None, hdf5_recorder=None, camera_widget=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lookup Table Generator")
        self.setMinimumWidth(450)
        self.setMinimumHeight(300)
        
        self.camera = camera  # Optional: use provided camera instead of creating new one
        self.hdf5_recorder = hdf5_recorder  # Optional: save into existing recording
        self.camera_widget = camera_widget  # Optional: pause/resume live view during acquisition
        self.acquisition_thread: Optional[LUTAcquisitionThread] = None
        self.is_acquiring = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Status display at the top
        self.status_display = StatusDisplay()
        self.status_display.set_status("Ready")
        self.status_display.setContentsMargins(0, 0, 0, 10)  # Add 10px bottom margin
        layout.addWidget(self.status_display)
        
        # Parameters group
        params_group = QGroupBox("Acquisition Parameters")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(8)
        
        # Use grid layout for proper alignment
        from PyQt5.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        
        # Set fixed widths for labels and spinboxes
        label_width = 80
        spinbox_width = 100
        
        # Z range - Row 0
        start_label = QLabel("Start:")
        start_label.setMinimumWidth(label_width)
        start_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(start_label, 0, 0)
        
        self.start_z_spin = QDoubleSpinBox()
        self.start_z_spin.setRange(0, 100)
        self.start_z_spin.setValue(0)
        self.start_z_spin.setSuffix(" µm")
        self.start_z_spin.setDecimals(0)
        self.start_z_spin.setFixedWidth(spinbox_width)
        self.start_z_spin.setAlignment(Qt.AlignRight)
        grid.addWidget(self.start_z_spin, 0, 1)
        
        end_label = QLabel("End:")
        end_label.setMinimumWidth(label_width)
        end_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(end_label, 0, 2)
        
        self.end_z_spin = QDoubleSpinBox()
        self.end_z_spin.setRange(0, 100)
        self.end_z_spin.setValue(100)
        self.end_z_spin.setSuffix(" µm")
        self.end_z_spin.setDecimals(0)
        self.end_z_spin.setFixedWidth(spinbox_width)
        self.end_z_spin.setAlignment(Qt.AlignRight)
        grid.addWidget(self.end_z_spin, 0, 3)
        
        # Step size - Row 1
        step_label = QLabel("Step Size:")
        step_label.setMinimumWidth(label_width)
        step_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(step_label, 1, 0)
        
        self.step_spin = QSpinBox()
        self.step_spin.setRange(10, 10000)
        self.step_spin.setValue(100)
        self.step_spin.setSuffix(" nm")
        self.step_spin.setSingleStep(10)
        self.step_spin.setFixedWidth(spinbox_width)
        self.step_spin.setAlignment(Qt.AlignRight)
        grid.addWidget(self.step_spin, 1, 1)
        
        # Settle time - Row 2
        settle_label = QLabel("Settle Time:")
        settle_label.setMinimumWidth(label_width)
        settle_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        settle_label.setToolTip("Time to wait after moving Z-stage before capturing frame")
        grid.addWidget(settle_label, 2, 0)
        
        self.settle_spin = QSpinBox()
        self.settle_spin.setRange(0, 5000)
        self.settle_spin.setValue(200)
        self.settle_spin.setSuffix(" ms")
        self.settle_spin.setSingleStep(50)
        self.settle_spin.setFixedWidth(spinbox_width)
        self.settle_spin.setAlignment(Qt.AlignRight)
        self.settle_spin.setToolTip("Time to wait after moving Z-stage before capturing frame")
        grid.addWidget(self.settle_spin, 2, 1)
        
        # Add stretch to right side
        grid.setColumnStretch(4, 1)
        
        params_layout.addLayout(grid)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("Start Acquisition")
        self.start_btn.clicked.connect(self._start_acquisition)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_acquisition)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def _start_acquisition(self):
        """Start the LUT acquisition."""
        # Validate parameters
        start = self.start_z_spin.value()
        end = self.end_z_spin.value()
        
        if end <= start:
            return
        
        # Check if we're using external camera/recorder or standalone mode
        # Prefer saving LUT into the active recording HDF5 file when available
        active_recorder = None
        active_camera = None
        if getattr(self, 'hdf5_recorder', None):
            active_recorder = self.hdf5_recorder
            active_camera = getattr(self, 'camera', None)
        elif getattr(self, 'camera_widget', None):
            # If this widget was given a CameraWidget, check for an active recorder there
            cw = self.camera_widget
            if getattr(cw, 'hdf5_recorder', None) and getattr(cw, 'is_recording', False):
                active_recorder = cw.hdf5_recorder
                active_camera = getattr(cw, 'camera', None)
            # Also get camera even if not recording (to avoid creating new camera instance)
            if not active_camera and getattr(cw, 'camera', None):
                active_camera = cw.camera

        if active_recorder and active_camera:
            # Use the recorder attached to the running session so LUT is stored inside
            # the same HDF5 file under /raw_data/LUT
            self.acquisition_thread = LUTAcquisitionThread(
                start_um=start,
                end_um=end,
                step_nm=self.step_spin.value(),
                camera=active_camera,
                hdf5_recorder=active_recorder,
                settle_time_ms=self.settle_spin.value(),
                camera_widget=getattr(self, 'camera_widget', None)
            )
        elif active_camera:
            # Have camera but no recorder - use existing camera with standalone file
            default_dir = Path.cwd() / "raw_data" / "LUT"
            default_dir.mkdir(parents=True, exist_ok=True)
            output_path = default_dir / f"lut_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

            # Use thread that takes existing camera (don't create new camera)
            self.acquisition_thread = LUTAcquisitionThread(
                start_um=start,
                end_um=end,
                step_nm=self.step_spin.value(),
                camera=active_camera,
                hdf5_recorder=None,  # Will create its own file
                settle_time_ms=self.settle_spin.value(),
                camera_widget=getattr(self, 'camera_widget', None),
                output_path=str(output_path)
            )
        else:
            # Standalone mode - create own camera and file (no active recording available)
            default_dir = Path.cwd() / "raw_data" / "LUT"
            default_dir.mkdir(parents=True, exist_ok=True)
            output_path = default_dir / f"lut_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

            # Create standalone thread that will create its own HDF5 file
            self.acquisition_thread = LUTAcquisitionThreadStandalone(
                start_um=start,
                end_um=end,
                step_nm=self.step_spin.value(),
                output_path=str(output_path),
                settle_time_ms=self.settle_spin.value()
            )
            # Let the thread notify this widget of the saved LUT file path
            try:
                self.acquisition_thread.parent_widget = self
            except Exception:
                pass
        
        # Connect signals
        self.acquisition_thread.progress.connect(self._on_progress)
        self.acquisition_thread.finished.connect(self._on_finished)
        self.acquisition_thread.frame_captured.connect(self._on_frame_captured)
        
        # Update UI
        self.is_acquiring = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.start_z_spin.setEnabled(False)
        self.end_z_spin.setEnabled(False)
        self.step_spin.setEnabled(False)
        self.settle_spin.setEnabled(False)
        
        # Start acquisition
        self.acquisition_thread.start()
        
    def _stop_acquisition(self):
        """Stop the acquisition."""
        if self.acquisition_thread:
            self.acquisition_thread.request_stop()
            
    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress update."""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
        self.status_display.set_status(message)
        
    def _on_frame_captured(self, frame_num: int, z_pos: float):
        """Handle frame captured event."""
        pass  # Minimal mode - no logging
        
    def _on_finished(self, success: bool, message: str):
        """Handle LUT acquisition completion."""
        if hasattr(self, 'camera_widget') and self.camera_widget:
            # Resume live view (restores camera settings and flushes buffers)
            if hasattr(self.camera_widget, 'resume_live'):
                self.camera_widget.resume_live()
                logger.info("Resuming live view after LUT acquisition")
            
            # Restart camera capture to ensure clean state (only if not recording)
            try:
                camera = getattr(self.camera_widget, 'camera', None)
                is_recording = getattr(self.camera_widget, 'is_recording', False)
                
                if camera and not is_recording:
                    logger.info("Restarting camera capture after LUT to ensure clean state")
                    
                    if hasattr(camera, 'stop_capture'):
                        camera.stop_capture()
                    
                    import time
                    time.sleep(0.2)
                    
                    if hasattr(camera, 'start_capture'):
                        camera.start_capture()
                    
                    logger.info("Camera capture restarted successfully after LUT")
                elif is_recording:
                    logger.info("Skipping camera restart - recording in progress")
            except Exception as e:
                logger.warning(f"Failed to restart camera capture after LUT: {e}")
        
        self.is_acquiring = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.start_z_spin.setEnabled(True)
        self.end_z_spin.setEnabled(True)
        self.step_spin.setEnabled(True)
        self.settle_spin.setEnabled(True)
        
        if success:
            self.progress_bar.setValue(100)
            # Auto-close dialog after successful acquisition
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(500, self.accept)  # Close after 500ms to show success message
        else:
            pass  # Minimal mode - no logging
            
    def closeEvent(self, event):
        """Handle dialog close."""
        if self.is_acquiring:
            event.ignore()
        else:
            event.accept()
