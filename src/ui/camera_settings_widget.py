"""
Minimal Camera Settings Widget for AFS Acquisition.
Ultra-simple design with only essential controls and 3 buttons.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QGroupBox, QLineEdit, QPushButton, QSpacerItem,
    QSizePolicy, QLabel, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from typing import Dict, Any, Optional

from src.utils.logger import get_logger

logger = get_logger("camera_settings")


class CameraSettingsWidget(QDialog):
    """
    Minimal camera settings dialog.
    Only essential controls with clean 3-button interface.
    """
    
    settings_applied = pyqtSignal(dict)
    
    # Default settings constants for better maintainability
    DEFAULT_SETTINGS = {
        'exposure_ms': 15.0,  # 15ms for stable 30 FPS operation (33ms frame time)
        'gain_master': 2,     # Moderate gain for good image quality
        'live_fps': 12,       # Live display frame rate
        'recording_fps': 30,  # Recording frame rate
        'frame_rate_fps': 30.0,  # Camera hardware frame rate (kept for compatibility)
    }
    
    def __init__(self, camera_controller=None, parent=None):
        super().__init__(parent)
        self.camera = camera_controller
        
        self.setWindowTitle("Camera Settings")
        self.setMinimumWidth(400)
        self.setModal(True)
        
        # Use class constant for defaults
        self.current_settings = self.DEFAULT_SETTINGS.copy()
        
        self.init_ui()
        self.load_current_settings()
    
    def init_ui(self):
        """Initialize minimal UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # Settings group with grid layout (2 columns)
        settings_group = QGroupBox("Parameters")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(10)
        settings_layout.setContentsMargins(15, 15, 15, 15)
        
        # Use grid layout for proper alignment (single column)
        grid = QGridLayout()
        grid.setHorizontalSpacing(15)
        grid.setVerticalSpacing(10)
        
        # Set fixed widths for labels and spinboxes
        label_width = 150
        spinbox_width = 120
        
        # Store input widgets for easy access
        self.inputs = {}
        
        # Row 0: Exposure
        exposure_label = QLabel("Exposure:")
        exposure_label.setMinimumWidth(label_width)
        exposure_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(exposure_label, 0, 0)
        
        self.inputs['exposure'] = QDoubleSpinBox()
        self.inputs['exposure'].setRange(0.1, 1000.0)
        self.inputs['exposure'].setValue(15.0)
        self.inputs['exposure'].setSuffix(" ms")
        self.inputs['exposure'].setDecimals(2)
        self.inputs['exposure'].setFixedWidth(spinbox_width)
        self.inputs['exposure'].setAlignment(Qt.AlignRight)
        grid.addWidget(self.inputs['exposure'], 0, 1)
        
        # Row 1: Gain
        gain_label = QLabel("Gain:")
        gain_label.setMinimumWidth(label_width)
        gain_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(gain_label, 1, 0)
        
        self.inputs['gain'] = QSpinBox()
        self.inputs['gain'].setRange(0, 100)
        self.inputs['gain'].setValue(2)
        self.inputs['gain'].setFixedWidth(spinbox_width)
        self.inputs['gain'].setAlignment(Qt.AlignRight)
        grid.addWidget(self.inputs['gain'], 1, 1)
        
        # Row 2: Live FPS
        live_fps_label = QLabel("Live Frame Rate:")
        live_fps_label.setMinimumWidth(label_width)
        live_fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(live_fps_label, 2, 0)
        
        self.inputs['live_fps'] = QSpinBox()
        self.inputs['live_fps'].setRange(1, 60)
        self.inputs['live_fps'].setValue(12)  # Display shows 12, applies as 14
        self.inputs['live_fps'].setSuffix(" fps")
        self.inputs['live_fps'].setFixedWidth(spinbox_width)
        self.inputs['live_fps'].setAlignment(Qt.AlignRight)
        grid.addWidget(self.inputs['live_fps'], 2, 1)
        
        # Row 3: Recording FPS
        recording_fps_label = QLabel("Recording Frame Rate:")
        recording_fps_label.setMinimumWidth(label_width)
        recording_fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(recording_fps_label, 3, 0)
        
        self.inputs['recording_fps'] = QSpinBox()
        self.inputs['recording_fps'].setRange(1, 120)
        self.inputs['recording_fps'].setValue(30)
        self.inputs['recording_fps'].setSuffix(" fps")
        self.inputs['recording_fps'].setFixedWidth(spinbox_width)
        self.inputs['recording_fps'].setAlignment(Qt.AlignRight)
        grid.addWidget(self.inputs['recording_fps'], 3, 1)
        
        # Add stretch to right side
        grid.setColumnStretch(2, 1)
        
        settings_layout.addLayout(grid)
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Button grid - create using helper method
        button_layout = self._create_button_grid()
        main_layout.addLayout(button_layout)
    
    def _create_button_grid(self) -> QGridLayout:
        """Create the button grid (2x2) with consistent styling."""
        button_layout = QGridLayout()
        button_layout.setHorizontalSpacing(5)
        button_layout.setVerticalSpacing(5)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        # Set button dimensions
        button_width = 120
        button_height = 28
        
        # Create buttons
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        apply_btn.setFixedSize(button_width, button_height)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_and_close)
        save_btn.setFixedSize(button_width, button_height)
        
        reconnect_btn = QPushButton("Reconnect")
        reconnect_btn.clicked.connect(self.reconnect_camera)
        reconnect_btn.setFixedSize(button_width, button_height)
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_to_defaults)
        reset_btn.setFixedSize(button_width, button_height)
        
        # 2x2 grid layout: Apply/Save on top, Reconnect/Reset on bottom
        button_layout.addWidget(apply_btn, 0, 0)
        button_layout.addWidget(save_btn, 0, 1)
        button_layout.addWidget(reconnect_btn, 1, 0)
        button_layout.addWidget(reset_btn, 1, 1)
        
        # Center the grid
        button_layout.setColumnStretch(0, 0)
        button_layout.setColumnStretch(1, 0)
        
        return button_layout
    
    def load_current_settings(self) -> None:
        """Load current camera settings if available."""
        if self.camera and hasattr(self.camera, 'get_camera_settings'):
            try:
                current = self.camera.get_camera_settings()
                if current:
                    # Only update with relevant settings that we can control
                    for key in self.DEFAULT_SETTINGS.keys():
                        if key in current:
                            self.current_settings[key] = current[key]
                    logger.info(f"Loaded camera settings: {self.current_settings}")
            except Exception as e:
                logger.warning(f"Failed to load camera settings: {e}")
        
        self.update_ui_from_settings()
    
    def load_defaults(self) -> None:
        """Load default settings."""
        self.current_settings = self.DEFAULT_SETTINGS.copy()
        self.update_ui_from_settings()
    
    def update_ui_from_settings(self) -> None:
        """Update UI controls from current settings."""
        # Mapping for cleaner updates
        settings_mapping = [
            ('exposure', 'exposure_ms'),
            ('gain', 'gain_master'),
            ('live_fps', 'live_fps'),
            ('recording_fps', 'recording_fps')
        ]
        
        for input_key, setting_key in settings_mapping:
            if input_key in self.inputs:
                value = self.current_settings.get(setting_key, self.DEFAULT_SETTINGS[setting_key])
                self.inputs[input_key].setValue(value)
    
    def get_settings_from_ui(self) -> Dict[str, Any]:
        """Get settings from UI controls with proper parsing and error handling."""
        # Mapping for parsing different types with defaults
        settings_mapping = [
            ('exposure', 'exposure_ms'),
            ('gain', 'gain_master'),
            ('live_fps', 'live_fps'),
            ('recording_fps', 'recording_fps')
        ]
        
        settings = {}
        
        for input_key, setting_key in settings_mapping:
            try:
                settings[setting_key] = self.inputs[input_key].value()
            except (ValueError, KeyError) as e:
                default_value = self.DEFAULT_SETTINGS[setting_key]
                settings[setting_key] = default_value
                logger.warning(f"Using default value for {setting_key}: {default_value}")
        
        # Also set frame_rate_fps to recording_fps for camera hardware compatibility
        settings['frame_rate_fps'] = float(settings['recording_fps'])
        
        return settings
    
    def reset_to_defaults(self):
        """Reset settings to defaults."""
        self.load_defaults()
        logger.info("Camera settings reset to defaults")
    
    def apply_settings(self):
        """Apply settings to camera with improved feedback."""
        settings = self.get_settings_from_ui()
        self.current_settings = settings
        
        logger.debug(f"Applying camera settings: {settings}")
        
        # Apply to camera if available
        if self.camera and hasattr(self.camera, 'apply_settings'):
            try:
                result = self.camera.apply_settings(settings)
                logger.debug(f"Camera settings applied: {result}")
                # Update UI to show rounded values
                self.update_ui_from_settings()
            except Exception as e:
                logger.error(f"Failed to apply camera settings: {e}")
        else:
            logger.warning("Camera controller not available for settings application")
        
        # Emit signal for other components
        self.settings_applied.emit(settings)
    
    def reconnect_camera(self):
        """Reconnect camera through main window."""
        logger.info("Camera reconnect requested from settings")
        # Try to get main window (parent of this dialog)
        main_window = self.parent()
        if main_window and hasattr(main_window, 'camera_widget'):
            camera_widget = main_window.camera_widget
            if hasattr(camera_widget, 'reconnect_camera'):
                camera_widget.reconnect_camera()
                logger.info("Camera reconnect initiated")
            else:
                logger.warning("Camera widget does not have reconnect_camera method")
        else:
            logger.warning("No main window or camera widget available for reconnect")
    
    def save_and_close(self):
        """Save settings and close dialog."""
        self.apply_settings()
        self.accept()
    
    def closeEvent(self, event):
        """Handle close event."""
        event.accept()