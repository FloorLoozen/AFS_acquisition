"""Main application window for the AFS Acquisition."""

import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QAction, 
    QMessageBox, QSizePolicy, QApplication, QMenuBar, QMenu
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCloseEvent

from src.utils.logger import get_logger
from src.controllers.device_manager import DeviceManager

# Import type annotations without importing the actual classes for runtime
if TYPE_CHECKING:
    from src.ui.camera_widget import CameraWidget
    from src.ui.frequency_settings_widget import FrequencySettingsWidget
    from src.ui.acquisition_controls_widget import AcquisitionControlsWidget
    from src.ui.measurement_controls_widget import MeasurementControlsWidget
    from src.ui.resonance_finder_widget import ResonanceFinderWidget
    from src.utils.keyboard_shortcuts import KeyboardShortcutManager

logger = get_logger("ui")


class MainWindow(QMainWindow):
    """Main application window for AFS Acquisition.
    
    Provides a 3-row layout with camera view, measurement settings,
    acquisition controls, and measurement controls. Handles hardware
    initialization, keyboard shortcuts, and application lifecycle.
    
    Attributes:
        camera_widget: Live camera display and recording controls
        frequency_settings_widget: File paths and sample information
        acquisition_controls_widget: Recording parameters and controls
        measurement_controls_widget: Measurement execution controls
        keyboard_shortcuts: Global keyboard shortcut manager
    """

    def __init__(self) -> None:
        """Initialize the main application window.
        
        Sets up the UI layout, initializes hardware connections,
        and configures keyboard shortcuts for efficient operation.
        """
        super().__init__()
        
        # Initialize widget references with proper type hints (using TYPE_CHECKING imports)
        self.camera_widget: Optional['CameraWidget'] = None
        self.frequency_settings_widget: Optional['FrequencySettingsWidget'] = None
        self.acquisition_controls_widget: Optional['AcquisitionControlsWidget'] = None
        self.measurement_controls_widget: Optional['MeasurementControlsWidget'] = None
        self.keyboard_shortcuts: Optional['KeyboardShortcutManager'] = None
        self.force_path_designer: Optional['ForcePathDesignerWindow'] = None
        
        # Session management and HDF5 logging
        self.session_hdf5_file: Optional[str] = None
        self.measurement_active: bool = False
        self.measurement_start_time: Optional[float] = None
        
        try:
            self._init_ui()
            
            # Initialize hardware
            # Initialize DeviceManager and hardware
            try:
                self.device_manager = DeviceManager.get_instance()
                logger.info("DeviceManager initialized")
                # Start background health monitor for reconnects (non-blocking)
                try:
                    self.device_manager.start_health_monitor(interval=5.0)
                except Exception:
                    pass  # Health monitor optional
            except Exception:
                self.device_manager = None

            self._initialize_hardware()
            
            # Set up keyboard shortcuts (lazy import)
            from src.utils.keyboard_shortcuts import KeyboardShortcutManager
            self.keyboard_shortcuts = KeyboardShortcutManager(self)
            
            # Delay focus setting to ensure everything is loaded
            QTimer.singleShot(100, self._ensure_main_window_focus)
            
            # Start status updates for real-time camera info
            QTimer.singleShot(2000, self.start_status_updates)  # Start after 2 seconds
            
        except Exception as e:
            logger.error(f"Error during main window initialization: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _create_measurement_hdf5(self) -> bool:
        """Create HDF5 file when measurement starts - DISABLED.
        
        All data is consolidated into video HDF5 files. Session files are not created.
        This method is kept for backwards compatibility but does nothing.
        """
        self.session_hdf5_file = None
        return True

    def _init_ui(self) -> None:
        """Initialize the user interface layout and appearance.
        
        Sets up the main window properties, creates the menu bar and central
        layout, and ensures proper focus for keyboard shortcuts.
        """
        self.setWindowTitle("AFS Acquisition")

        self._create_menu_bar()
        self._create_central_layout()

        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Ready")
        
        # Ensure main window has focus to capture keyboard shortcuts
        self.setFocus()
        self.activateWindow()
        self.raise_()

    def _create_menu_bar(self) -> None:
        """Create and configure the application menu bar.
        
        Sets up File, Hardware, and Help menus with appropriate actions
        and keyboard shortcuts for common operations.
        """
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        self._add_action(file_menu, "Toggle Maximize", "F11", self._toggle_fullscreen)
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", "Ctrl+Q", self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        self._add_action(tools_menu, "Camera Settings", None, self._open_camera_settings)
        self._add_action(tools_menu, "Stage Controller", None, self._open_stage_controls)
        self._add_action(tools_menu, "Resonance Finder", None, self._open_resonance_finder)
        self._add_action(tools_menu, "Lookup Table Generator", None, self._show_not_implemented)
        self._add_action(tools_menu, "Force Path Designer", None, self._open_force_path_designer)

        # Help menu
        help_menu = menubar.addMenu("Help")
        self._add_action(help_menu, "About", None, self._open_about)

    def _add_action(self, menu: QMenu, text: str, shortcut: str, callback) -> QAction:
        """Helper to add menu actions with consistent formatting.
        
        Args:
            menu: The menu to add the action to
            text: Display text for the menu item
            shortcut: Keyboard shortcut string (e.g., 'Ctrl+S')
            callback: Function to call when action is triggered
            
        Returns:
            The created QAction object
        """
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        menu.addAction(action)

    def _create_central_layout(self):
        """Create main layout: left column (3 rows of controls) + right column (camera)."""
        central = QWidget(self)
        layout = QGridLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # Left column widgets
        self._create_frequency_settings_widget(layout, 0, 0)
        self._create_acquisition_controls_widget(layout, 1, 0)
        self._create_measurement_controls_widget(layout, 2, 0)
        
        # Right column camera (spans all 3 rows) - lazy import for heavy camera module
        try:
            from src.ui.camera_widget import CameraWidget
            self.camera_widget = CameraWidget()
            self.camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.camera_widget.setMinimumWidth(400)
            layout.addWidget(self.camera_widget, 0, 1, 3, 1)
        except Exception as e:
            logger.error(f"Error creating camera widget: {e}")
            raise
        
        # Set proportions: 45% controls, 55% camera
        layout.setColumnStretch(0, 45)
        layout.setColumnStretch(1, 55)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 1)
        
        self.setCentralWidget(central)
        
    def _create_frequency_settings_widget(self, layout, row, col):
        """Create and setup the frequency settings widget."""
        try:
            from src.ui.frequency_settings_widget import FrequencySettingsWidget
            self.frequency_settings_widget = FrequencySettingsWidget()
            layout.addWidget(self.frequency_settings_widget, row, col)
        except Exception as e:
            logger.error(f"Error creating measurement settings widget: {e}")
            raise
        
    def _create_acquisition_controls_widget(self, layout, row, col):
        """Create and add acquisition controls widget."""
        try:
            from src.ui.acquisition_controls_widget import AcquisitionControlsWidget
            self.acquisition_controls_widget = AcquisitionControlsWidget()
            
            # Set measurement settings reference
            if self.frequency_settings_widget:
                self.acquisition_controls_widget.set_frequency_settings_widget(self.frequency_settings_widget)
            
            # Connect recording signals
            self.acquisition_controls_widget.start_recording_requested.connect(self._handle_start_recording)
            self.acquisition_controls_widget.stop_recording_requested.connect(self._handle_stop_recording)
            self.acquisition_controls_widget.save_recording_requested.connect(self._handle_save_recording)
            
            layout.addWidget(self.acquisition_controls_widget, row, col)
            
        except Exception as e:
            logger.error(f"Error creating acquisition controls widget: {e}")
            raise
    
    def _create_measurement_controls_widget(self, layout, row, col):
        """Create and add measurement controls widget."""
        try:
            from src.ui.measurement_controls_widget import MeasurementControlsWidget
            self.measurement_controls_widget = MeasurementControlsWidget()
            
            # Connect function generator signals to HDF5 timeline logging
            self.measurement_controls_widget.function_generator_toggled.connect(self._on_function_generator_toggled)
            self.measurement_controls_widget.function_generator_settings_changed.connect(self._on_function_generator_settings_changed)
            
            layout.addWidget(self.measurement_controls_widget, row, col)
            
        except Exception as e:
            logger.error(f"Error creating measurement controls widget: {e}")
            raise

    # Menu handlers
    def _show_not_implemented(self):
        """Show not implemented message."""
        QMessageBox.information(self, "Not Implemented", "This feature will be implemented later.")

    def _open_stage_controls(self):
        """Open XY Stage control dialog."""
        try:
            QApplication.restoreOverrideCursor()
            
            from src.ui.stages_widget import StagesWidget
            
            # Store reference to prevent garbage collection
            if not hasattr(self, '_stage_dialog') or not self._stage_dialog.isVisible():
                self._stage_dialog = StagesWidget(self)
                self._stage_dialog.show()
            else:
                self._stage_dialog.activateWindow()
                self._stage_dialog.raise_()
        except Exception as e:
            logger.error(f"Failed to open stage dialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open stage controls: {e}")

    def _open_camera_settings(self):
        """Open camera settings dialog."""
        try:
            from src.ui.camera_settings_widget import CameraSettingsWidget
            
            # Get camera controller from camera widget
            camera_controller = None
            if hasattr(self.camera_widget, 'camera'):
                camera_controller = self.camera_widget.camera
            
            # Store reference to prevent garbage collection
            if not hasattr(self, '_camera_settings_dialog') or not self._camera_settings_dialog.isVisible():
                self._camera_settings_dialog = CameraSettingsWidget(camera_controller, self)
                # Connect settings applied signal
                self._camera_settings_dialog.settings_applied.connect(self._on_camera_settings_applied)
                self._camera_settings_dialog.show()
            else:
                # Update controller reference in case it changed
                if hasattr(self._camera_settings_dialog, 'camera'):
                    self._camera_settings_dialog.camera = camera_controller
                    self._camera_settings_dialog.load_current_settings()
                self._camera_settings_dialog.activateWindow()
                self._camera_settings_dialog.raise_()
        except Exception as e:
            logger.error(f"Failed to open camera settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open camera settings: {e}")
    
    def _on_camera_settings_applied(self, settings: dict):
        """Handle camera settings being applied."""
        # The settings have already been applied to the camera controller
        # Also update image processing settings in the camera widget
        if self.camera_widget and hasattr(self.camera_widget, 'update_image_settings'):
            self.camera_widget.update_image_settings(settings)

    def _toggle_fullscreen(self):
        """Toggle between maximized and normal window size."""
        if self.isMaximized():
            self.showNormal()
            logger.info("Switched to normal window mode")
        else:
            self.showMaximized()
            logger.info("Switched to maximized mode")
    
    def _open_resonance_finder(self):
        """Open resonance finder window with oscilloscope display."""
        # DISABLED: Resonance finder functionality commented out for now
        # Oscilloscope integration will be added in the future
        logger.info("Resonance Finder is currently disabled (not implemented)")
        QMessageBox.information(self, "Feature Disabled", 
                              "Resonance Finder is currently disabled.\n\n"
                              "This feature will be re-enabled in a future update when "
                              "oscilloscope integration is complete.")
        
        # # FUTURE: Uncomment this code when oscilloscope is ready
        # try:
        #     from src.ui.resonance_finder_widget import ResonanceFinderWidget
        #     
        #     # Create or show resonance finder window
        #     if not hasattr(self, '_resonance_finder_window') or not self._resonance_finder_window:
        #         # Get shared oscilloscope controller if available
        #         osc = getattr(self, 'oscilloscope_controller', None)
        #         fg = None
        #         if self.measurement_controls_widget:
        #             fg = self.measurement_controls_widget.get_function_generator_controller()
        #         
        #         self._resonance_finder_window = ResonanceFinderWidget(funcgen=fg, oscilloscope=osc)
        #     
        #     # Show and bring to front
        #     self._resonance_finder_window.show()
        #     self._resonance_finder_window.activateWindow()
        #     self._resonance_finder_window.raise_()
        #     
        #     logger.info("Opened Resonance Finder window")
        #     
        # except Exception as e:
        #     logger.error(f"Failed to open Resonance Finder: {e}")
        #     import traceback
        #     error_details = traceback.format_exc()
        #     QMessageBox.critical(self, "Error", 
        #         f"Failed to open Resonance Finder:\n{e}\n\nCheck the log for details.")
        #     logger.error(f"Resonance Finder error details:\n{error_details}")
    
    def _open_force_path_designer(self):
        """Open Force Path Designer window."""
        try:
            from src.ui.force_path_designer_widget import ForcePathDesignerWindow
            
            # Create or show force path designer window
            if not hasattr(self, '_force_path_designer_window') or not self._force_path_designer_window:
                self._force_path_designer_window = ForcePathDesignerWindow()
                
                # Set the main window reference for measurement-driven logging
                self._force_path_designer_window.designer_widget.set_main_window(self)
                
                # Connect function generator controller if available
                if (hasattr(self, 'measurement_controls_widget') and 
                    self.measurement_controls_widget and 
                    hasattr(self.measurement_controls_widget, 'fg_controller') and
                    self.measurement_controls_widget.fg_controller):
                    self._force_path_designer_window.set_function_generator_controller(
                        self.measurement_controls_widget.fg_controller
                    )
                
            # Show and bring to front
            self._force_path_designer_window.show()
            self._force_path_designer_window.activateWindow()
            self._force_path_designer_window.raise_()
            
            logger.info("Opened Force Path Designer window")
            
        except Exception as e:
            logger.error(f"Failed to open Force Path Designer: {e}")
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Error", 
                f"Failed to open Force Path Designer:\n{e}\n\nCheck the log for details.")
            logger.error(f"Force Path Designer error details:\n{error_details}")
    
    def _open_about(self):
        """Show about dialog."""
        QMessageBox.information(self, "About", 
            "AFS Acquisition v3\n\n"
            "Automated acquisition system for AFS using IDS cameras "
            "and MCL MicroDrive XY stage hardware.")
    
    def start_measurement_session(self):
        """Start measurement session (lightweight)."""
        if self.measurement_active:
            logger.warning("Measurement session already active")
            return
        
        try:
            # Quick hardware validation
            if not self.camera_widget or not hasattr(self.camera_widget, 'camera'):
                logger.error("Camera not available for measurement")
                return
            
            self.measurement_active = True
            self.measurement_start_time = time.time()
            self.session_hdf5_file = None  # No session files - data goes to video files
            
            logger.info("Measurement session started")
            
        except Exception as e:
            logger.error(f"Failed to start measurement session: {e}")
            self.measurement_active = False
    
    def stop_measurement_session(self):
        """Stop measurement session (lightweight)."""
        if not self.measurement_active:
            logger.warning("No measurement session active")
            return
        
        try:
            self.measurement_active = False
            
            if self.measurement_start_time:
                duration = time.time() - self.measurement_start_time
                logger.info(f"Measurement session stopped - Duration: {duration:.1f}s")
            
            self.measurement_start_time = None
            
        except Exception as e:
            logger.error(f"Error stopping measurement session: {e}")

    def log_execution_data(self, execution_type: str, data: dict):
        """Log execution data to video HDF5 file (consolidated storage)."""
        if not self.measurement_active:
            return  # Don't log if no measurement active
            
        # Route execution data to video HDF5 file instead of session file
        if hasattr(self, 'camera_widget') and self.camera_widget:
            if hasattr(self.camera_widget, 'hdf5_recorder') and self.camera_widget.hdf5_recorder:
                hdf5_recorder = self.camera_widget.hdf5_recorder
                if hdf5_recorder.is_recording:
                    success = hdf5_recorder.log_execution_data(execution_type, data)
                    if success:
                        pass  # Data logged successfully
                    else:
                        logger.warning(f"Failed to log execution data to video HDF5: {execution_type}")
                else:
                    pass  # Not recording
            else:
                pass  # No recorder
        else:
            pass  # No camera widget

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle application close with proper cleanup."""
        # Check if recording is still active or currently saving
        if hasattr(self, 'camera_widget') and self.camera_widget:
            if self.camera_widget.is_recording or self.camera_widget.is_saving:
                # Determine the message based on state
                if self.camera_widget.is_saving:
                    message = (
                        "The recording is currently being saved.\n\n"
                        "Please wait for the save operation to complete.\n\n"
                        "You cannot close the application while saving is in progress."
                    )
                    title = "Save in Progress"
                else:
                    message = (
                        "A recording is currently in progress.\n\n"
                        "The recording will be stopped and saved.\n"
                        "The application will remain open.\n\n"
                        "Do you want to stop the recording now?"
                    )
                    title = "Recording in Progress"
                
                # Show appropriate dialog
                if self.camera_widget.is_saving:
                    # Just inform - don't allow close
                    QMessageBox.warning(
                        self,
                        title,
                        message
                    )
                    event.ignore()
                    return
                else:
                    # Ask to stop recording
                    reply = QMessageBox.question(
                        self,
                        title,
                        message,
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        # Stop recording but DON'T close the application
                        logger.info("Stopping recording (application will stay open)")
                        try:
                            self.camera_widget.stop_recording()
                            logger.info("Recording stopped - application remains open")
                            # Prevent closing - just stop the recording
                            event.ignore()
                            return
                        except Exception as e:
                            logger.error(f"Error stopping recording: {e}")
                            QMessageBox.warning(
                                self,
                                "Error",
                                f"Failed to stop recording properly.\n\nError: {e}"
                            )
                            event.ignore()
                            return
                    else:
                        # User chose to keep recording
                        event.ignore()
                        return
        
        logger.info("Application closing - cleaning up")
        
        try:
            # Stop any running measurements
            if self.measurement_active:
                self.stop_measurement_session()
            
            # Close force path designer if open
            if hasattr(self, 'force_path_designer') and self.force_path_designer:
                try:
                    self.force_path_designer.close()
                except:
                    pass
            
            # Cleanup function generator (critical for avoiding connection issues)
            if hasattr(self, 'measurement_controls_widget') and self.measurement_controls_widget:
                try:
                    self.measurement_controls_widget.cleanup()
                except Exception as e:
                    logger.error(f"Function generator cleanup error: {e}")
            
            # Cleanup camera
            if hasattr(self, 'camera_widget') and self.camera_widget:
                try:
                    self.camera_widget.close()
                except Exception as e:
                    pass  # Camera cleanup error
            
            # Cleanup stage controller
            try:
                from src.controllers.stage_manager import StageManager
                stage_manager = StageManager.get_instance()
                stage_manager.disconnect()
            except Exception as e:
                pass  # Stage cleanup error
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        event.accept()
    
    def get_frequency_settings(self) -> Optional['FrequencySettingsWidget']:
        """Get the frequency settings widget."""
        return self.frequency_settings_widget
    
    def get_save_path(self) -> str:
        """Get the configured save path."""
        if self.frequency_settings_widget:
            return self.frequency_settings_widget.get_save_path()
        return ""
    
    def get_function_generator_controller(self):
        """Get the function generator controller instance."""
        if self.measurement_controls_widget and hasattr(self.measurement_controls_widget, 'get_function_generator_controller'):
            return self.measurement_controls_widget.get_function_generator_controller()
        return None
    
    def _handle_start_recording(self, file_path):
        """Handle start recording request."""
        if not self.camera_widget or not self.camera_widget.is_running:
            self.acquisition_controls_widget.recording_failed(
                "Camera is not running. Please ensure camera is connected and running.")
            return
        
        try:
            # Compression and resolution are now fixed defaults in camera_widget
            # Maximum compression (9) and half resolution (2) for offline analysis
            
            if hasattr(self.camera_widget, 'start_recording'):
                # Gather metadata from measurement settings
                metadata = {}
                if self.frequency_settings_widget:
                    metadata['sample_name'] = self.frequency_settings_widget.get_sample_information()
                    metadata['measurement_notes'] = self.frequency_settings_widget.get_notes()
                    metadata['save_path'] = self.frequency_settings_widget.get_save_path()
                
                success = self.camera_widget.start_recording(file_path, metadata)
                if success:
                    # Log initial function generator state
                    if self.measurement_controls_widget and hasattr(self.camera_widget, 'log_initial_function_generator_state'):
                        try:
                            frequency = 1.0
                            amplitude = 1.0
                            enabled = False
                            
                            try:
                                frequency = float(self.measurement_controls_widget.frequency_edit.text())
                                amplitude = float(self.measurement_controls_widget.amplitude_edit.text())
                                enabled = self.measurement_controls_widget.fg_toggle_button.isChecked()
                            except (ValueError, AttributeError):
                                pass
                            
                            self.camera_widget.log_initial_function_generator_state(frequency, amplitude, enabled)
                        except Exception as e:
                            logger.warning(f"Failed to log initial function generator state: {e}")
                    
                    self.acquisition_controls_widget.recording_started_successfully()
                    self.statusBar().showMessage(f"HDF5 recording started: {file_path}")
                else:
                    self.acquisition_controls_widget.recording_failed("Failed to start HDF5 recording in camera.")
            else:
                self.acquisition_controls_widget.recording_failed("Camera widget does not support recording.")
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.acquisition_controls_widget.recording_failed(f"Error starting recording: {str(e)}")
    
    def _handle_stop_recording(self):
        """Handle stop recording request."""
        try:
            if hasattr(self.camera_widget, 'stop_recording'):
                saved_path = self.camera_widget.stop_recording()
                if saved_path:
                    self.acquisition_controls_widget.recording_stopped_successfully(saved_path)
                    self.statusBar().showMessage(f"HDF5 recording stopped: {saved_path}")
                else:
                    self.acquisition_controls_widget.recording_failed("Failed to stop HDF5 recording properly.")
            else:
                self.acquisition_controls_widget.recording_failed("Camera widget does not support recording.")
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.acquisition_controls_widget.recording_failed(f"Error stopping recording: {str(e)}")
    
    def _handle_save_recording(self, file_path):
        """Handle save recording request."""
        import os
        import shutil
        
        try:
            # Get the original recorded file path
            if hasattr(self.acquisition_controls_widget, 'original_recording_path'):
                actual_recorded_path = self.acquisition_controls_widget.original_recording_path
                
                # Check if we need to rename/move the file
                if actual_recorded_path and actual_recorded_path != file_path and os.path.exists(actual_recorded_path):
                    # Wait briefly for file to be fully written
                    QTimer.singleShot(500, lambda: self._perform_file_move(actual_recorded_path, file_path))
                    return
            
            # File is already in the right place
            self.statusBar().showMessage(f"HDF5 recording saved: {file_path}")
            QTimer.singleShot(3000, self.acquisition_controls_widget.clear_status)
            
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            self.acquisition_controls_widget.recording_failed(f"Error saving recording: {str(e)}")
    
    def _perform_file_move(self, source_path: str, dest_path: str):
        """Perform file move operation with retries."""
        import os
        import shutil
        import time
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Check if source still exists
                if not os.path.exists(source_path):
                    logger.warning(f"Source file no longer exists: {source_path}")
                    return
                
                # Try to move the file
                shutil.move(source_path, dest_path)
                logger.info(f"Successfully moved recording to: {dest_path}")
                self.statusBar().showMessage(f"HDF5 recording saved: {dest_path}")
                QTimer.singleShot(3000, self.acquisition_controls_widget.clear_status)
                return
                
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    # Wait and retry
                    time.sleep(0.3 * (attempt + 1))
                else:
                    # Last attempt - try copy instead of move
                    try:
                        shutil.copy2(source_path, dest_path)
                        logger.info(f"Copied recording to: {dest_path}")
                        try:
                            os.remove(source_path)
                        except:
                            logger.warning(f"Could not remove original: {source_path}")
                        self.statusBar().showMessage(f"HDF5 recording saved: {dest_path}")
                        QTimer.singleShot(3000, self.acquisition_controls_widget.clear_status)
                        return
                    except Exception as copy_error:
                        logger.error(f"Failed to save recording: {copy_error}")
                        self.acquisition_controls_widget.recording_failed(
                            "Error saving recording: file is locked or inaccessible.")
                        return
    
    def _on_function_generator_toggled(self, enabled: bool):
        """Handle function generator on/off events for timeline logging."""
        if self.camera_widget and hasattr(self.camera_widget, 'log_function_generator_toggle'):
            # Get current settings from measurement controls
            frequency = 1.0  # Default
            amplitude = 1.0  # Default
            
            if self.measurement_controls_widget:
                try:
                    frequency = float(self.measurement_controls_widget.frequency_edit.text())
                    amplitude = float(self.measurement_controls_widget.amplitude_edit.text())
                except (ValueError, AttributeError):
                    pass  # Use defaults
            
            self.camera_widget.log_function_generator_toggle(enabled, frequency, amplitude)
    
    def _on_function_generator_settings_changed(self, frequency_mhz: float, amplitude_vpp: float):
        """Handle function generator settings changes for timeline logging."""
        if self.camera_widget and hasattr(self.camera_widget, 'log_function_generator_event'):
            # Determine if output is currently enabled
            output_enabled = False
            if self.measurement_controls_widget and hasattr(self.measurement_controls_widget, 'fg_toggle_button'):
                output_enabled = self.measurement_controls_widget.fg_toggle_button.isChecked()
            
            self.camera_widget.log_function_generator_event(
                frequency_mhz, amplitude_vpp, 
                output_enabled=output_enabled, 
                event_type='parameter_change'
            )
        
    def _initialize_hardware(self):
        """Initialize all hardware components at startup with fast-fail for quick start."""
        # Hardware initialization starts
        self.statusBar().showMessage("Initializing hardware...")
        
        # Collect hardware status for non-camera components first
        hardware_status = {}
        
        try:
            hardware_status["XY Stage"] = self._init_xy_stage()
            hardware_status["Function Generator"] = self._init_function_generator()
            
            # DISABLED: Oscilloscope functionality commented out for now
            # This will be re-enabled when oscilloscope integration is complete
            self.oscilloscope_controller = None
            
            # # FUTURE: Uncomment this when oscilloscope is ready
            # if hasattr(self, 'device_manager') and getattr(self.device_manager, '_disable_osc', False):
            #     logger.info("Oscilloscope disabled for fast startup - not included in hardware status")
            #     self.oscilloscope_controller = None
            # else:
            #     # Create a shared oscilloscope controller for other widgets only if enabled
            #     try:
            #         from src.controllers.oscilloscope_controller import get_oscilloscope_controller
            #         self.oscilloscope_controller = get_oscilloscope_controller()
            #         try:
            #             # Fast connection attempt - don't block startup
            #             if self.oscilloscope_controller.connect(fast_fail=True):
            #                 hardware_status["Oscilloscope"] = {"connected": True, "message": "Oscilloscope connected"}
            #             else:
            #                 hardware_status["Oscilloscope"] = {"connected": False, "message": "Not found (will retry in background)"}
            #         except Exception as e:
            #             hardware_status["Oscilloscope"] = {"connected": False, "message": f"Fast connect failed: {str(e)}"}
            #     except Exception as e:
            #         self.oscilloscope_controller = None

            # Handle camera separately - wait for it to fully initialize
            self._init_camera_with_status_check(hardware_status)
            
        except Exception as e:
            logger.error(f"Hardware initialization error: {e}")
            self.statusBar().showMessage(f"Hardware initialization error: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
    
    def _init_camera(self):
        """Initialize camera hardware."""
        try:
            if self.camera_widget and hasattr(self.camera_widget, 'is_running'):
                if self.camera_widget.is_running:
                    # Check if using test pattern or real hardware
                    use_test_pattern = getattr(self.camera_widget, 'use_test_pattern', False)
                    if use_test_pattern:
                        logger.info("Camera running in test pattern mode")
                        return {"connected": False, "message": "Using test pattern (no hardware detected)"}
                    else:
                        return {"connected": True, "message": "Camera hardware detected"}
                else:
                    return {"connected": False, "message": "Still initializing"}
            else:
                logger.warning("Camera widget creation failed")
                return {"connected": False, "message": "Camera widget creation failed"}
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return {"connected": False, "message": f"Error: {str(e)}"}
            
    def _init_xy_stage(self):
        """Initialize XY stage hardware."""
        try:
            from src.controllers.stage_manager import StageManager
            stage_manager = StageManager.get_instance()
            if stage_manager.connect():
                return {"connected": True, "message": "XY stage hardware connected"}
            else:
                logger.warning("XY stage connection failed")
                return {"connected": False, "message": "XY stage hardware not found"}
        except Exception as e:
            logger.warning(f"XY stage initialization failed: {e}")
            return {"connected": False, "message": f"Initialization error: {str(e)}"}
    
    def _init_function_generator(self):
        """Initialize function generator hardware."""
        try:
            if self.measurement_controls_widget and hasattr(self.measurement_controls_widget, 'get_function_generator_controller'):
                # Allow time for connection
                QApplication.processEvents()
                
                fg_controller = self.measurement_controls_widget.get_function_generator_controller()
                if fg_controller and fg_controller.is_connected:
                    return {"connected": True, "message": "Function generator connected"}
                else:
                    return {"connected": False, "message": "VISA resource not found"}
            else:
                logger.warning("Function generator controller not available")
                return {"connected": False, "message": "Controller not available"}
        except Exception as e:
            logger.error(f"Function generator check failed: {e}")
            return {"connected": False, "message": f"Error: {str(e)}"}
    
    def _show_hardware_status_warning(self, hardware_status):
        """Show hardware status warning dialog if there are connection issues."""
        try:
            from src.ui.hardware_status_dialog import show_hardware_status_warning
            
            result = show_hardware_status_warning(hardware_status, self)
            
            if result == 2:  # Retry requested
                logger.info("User requested hardware connection retry")
                self.statusBar().showMessage("Retrying hardware connections...")
                
                # Retry hardware initialization after a short delay
                QTimer.singleShot(500, self._retry_hardware_initialization)
            elif result == 1:  # Continue anyway
                logger.info("User chose to continue with hardware connection issues")
                self._show_initialization_complete()
            elif result is None:  # No issues, dialog not shown
                self._show_initialization_complete()
                
        except ImportError as e:
            logger.error(f"Could not import hardware status dialog: {e}")
            # Fallback: show simple message box for critical issues
            disconnected = [name for name, status in hardware_status.items() 
                          if not status.get('connected', False)]
            if disconnected:
                QMessageBox.warning(self, "Hardware Warning", 
                                  f"Hardware not connected: {', '.join(disconnected)}")
        except Exception as e:
            logger.error(f"Error showing hardware status dialog: {e}")
    
    def _retry_hardware_initialization(self):
        """Retry hardware initialization."""
        
        # Collect hardware status again
        hardware_status = {}
        
        try:
            hardware_status["XY Stage"] = self._init_xy_stage()
            hardware_status["Function Generator"] = self._init_function_generator()
            
            # Check camera again
            camera_status = self._init_camera()
            hardware_status["Camera"] = camera_status
            
            # Show status immediately on retry
            self._check_and_show_hardware_status(hardware_status)
            
        except Exception as e:
            logger.error(f"Hardware retry error: {e}")
            self.statusBar().showMessage(f"Hardware retry error: {str(e)}")
    
    def _init_camera_with_status_check(self, hardware_status):
        """Initialize camera and wait for completion."""
        self.statusBar().showMessage("Initializing camera...")
        
        # Start camera initialization
        initial_status = self._init_camera()
        
        if initial_status["connected"]:
            # Camera already connected
            hardware_status["Camera"] = initial_status
            self._check_and_show_hardware_status(hardware_status)
        else:
            # Camera still initializing, wait up to 5 seconds
            self._wait_for_camera_initialization(hardware_status, max_wait_time=5)
    
    def _wait_for_camera_initialization(self, hardware_status, max_wait_time=5):
        """Wait for camera to finish initializing."""
        self.camera_wait_elapsed = 0
        
        def check_camera_status():
            self.camera_wait_elapsed += 1
            
            camera_status = self._init_camera()
            
            if camera_status["connected"]:
                # Camera now connected
                hardware_status["Camera"] = camera_status
                self._check_and_show_hardware_status(hardware_status)
            elif self.camera_wait_elapsed >= max_wait_time:
                # Timeout
                logger.warning("Camera initialization timeout")
                hardware_status["Camera"] = {"connected": False, "message": "Initialization timeout"}
                self._check_and_show_hardware_status(hardware_status)
            else:
                # Keep waiting
                self.statusBar().showMessage(f"Camera initializing... ({self.camera_wait_elapsed}s)")
                QTimer.singleShot(1000, check_camera_status)
        
        # Start checking after 1 second
        QTimer.singleShot(1000, check_camera_status)
    
    def _check_and_show_hardware_status(self, hardware_status):
        """Check hardware status and show warning if needed."""
        # Show hardware status warning if needed
        self._show_hardware_status_warning(hardware_status)
        
        # Show completion message
        QTimer.singleShot(500, self._show_initialization_complete)
    
    def _show_initialization_complete(self):
        """Show initialization complete message."""
        self.statusBar().showMessage("Hardware initialization complete")

    def _ensure_main_window_focus(self):
        """Ensure main window has focus for keyboard shortcuts to work."""
        self.setFocus()
        self.activateWindow()
        self.raise_()
    
    def update_camera_status(self):
        """Update status bar with camera information."""
        try:
            if self.camera_widget and hasattr(self.camera_widget, 'camera') and self.camera_widget.camera:
                pass  # Status updates disabled
            else:
                pass  # Status updates disabled
        except Exception as e:
            pass  # Status update error
    
    def start_status_updates(self):
        """Start periodic status updates."""
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_camera_status)
        self.status_timer.start(2000)  # Update every 2 seconds