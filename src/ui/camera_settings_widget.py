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
        'frame_rate_fps': 30.0,  # 30 FPS baseline (display at 15 FPS, record at 30 FPS)
        'brightness': 50,
        'contrast': 50,
        'saturation': 50
    }
    
    def __init__(self, camera_controller=None, parent=None):
        super().__init__(parent)
        self.camera = camera_controller
        
        self.setWindowTitle("Camera Settings")
        self.setMinimumWidth(450)
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
        
        # Use grid layout for proper alignment
        grid = QGridLayout()
        grid.setHorizontalSpacing(15)
        grid.setVerticalSpacing(10)
        
        # Set fixed widths for labels and spinboxes
        label_width = 100
        spinbox_width = 100
        
        # Store input widgets for easy access
        self.inputs = {}
        
        # Row 0: Exposure and Gain
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
        
        gain_label = QLabel("Gain:")
        gain_label.setMinimumWidth(label_width)
        gain_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(gain_label, 0, 2)
        
        self.inputs['gain'] = QSpinBox()
        self.inputs['gain'].setRange(0, 100)
        self.inputs['gain'].setValue(2)
        self.inputs['gain'].setFixedWidth(spinbox_width)
        self.inputs['gain'].setAlignment(Qt.AlignRight)
        grid.addWidget(self.inputs['gain'], 0, 3)
        
        # Row 1: Frame Rate and Brightness
        fps_label = QLabel("Frame Rate:")
        fps_label.setMinimumWidth(label_width)
        fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(fps_label, 1, 0)
        
        self.inputs['fps'] = QDoubleSpinBox()
        self.inputs['fps'].setRange(1.0, 120.0)
        self.inputs['fps'].setValue(30.0)
        self.inputs['fps'].setSuffix(" fps")
        self.inputs['fps'].setDecimals(2)
        self.inputs['fps'].setFixedWidth(spinbox_width)
        self.inputs['fps'].setAlignment(Qt.AlignRight)
        grid.addWidget(self.inputs['fps'], 1, 1)
        
        brightness_label = QLabel("Brightness:")
        brightness_label.setMinimumWidth(label_width)
        brightness_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(brightness_label, 1, 2)
        
        self.inputs['brightness'] = QSpinBox()
        self.inputs['brightness'].setRange(0, 100)
        self.inputs['brightness'].setValue(50)
        self.inputs['brightness'].setFixedWidth(spinbox_width)
        self.inputs['brightness'].setAlignment(Qt.AlignRight)
        grid.addWidget(self.inputs['brightness'], 1, 3)
        
        # Row 2: Contrast and Saturation
        contrast_label = QLabel("Contrast:")
        contrast_label.setMinimumWidth(label_width)
        contrast_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(contrast_label, 2, 0)
        
        self.inputs['contrast'] = QSpinBox()
        self.inputs['contrast'].setRange(0, 100)
        self.inputs['contrast'].setValue(50)
        self.inputs['contrast'].setFixedWidth(spinbox_width)
        self.inputs['contrast'].setAlignment(Qt.AlignRight)
        grid.addWidget(self.inputs['contrast'], 2, 1)
        
        saturation_label = QLabel("Saturation:")
        saturation_label.setMinimumWidth(label_width)
        saturation_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(saturation_label, 2, 2)
        
        self.inputs['saturation'] = QSpinBox()
        self.inputs['saturation'].setRange(0, 100)
        self.inputs['saturation'].setValue(50)
        self.inputs['saturation'].setFixedWidth(spinbox_width)
        self.inputs['saturation'].setAlignment(Qt.AlignRight)
        grid.addWidget(self.inputs['saturation'], 2, 3)
        
        # Add stretch to right side
        grid.setColumnStretch(4, 1)
        
        settings_layout.addLayout(grid)
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Button row - create using helper method
        button_layout = self._create_button_row()
        main_layout.addLayout(button_layout)
    
    def _create_button_row(self) -> QHBoxLayout:
        """Create the button row with consistent styling."""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        # Button configurations
        button_configs = [
            ("Apply", self.apply_settings),
            ("Reset", self.reset_to_defaults),
            ("Reconnect", self.reconnect_camera),
            ("Save", self.save_and_close)
        ]
        
        button_layout.addStretch()
        
        for text, callback in button_configs:
            button = QPushButton(text)
            button.clicked.connect(callback)
            button_layout.addWidget(button)
        
        button_layout.addStretch()
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
            ('fps', 'frame_rate_fps'),
            ('brightness', 'brightness'),
            ('contrast', 'contrast'),
            ('saturation', 'saturation')
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
            ('fps', 'frame_rate_fps'),
            ('brightness', 'brightness'),
            ('contrast', 'contrast'),
            ('saturation', 'saturation')
        ]
        
        settings = {}
        
        for input_key, setting_key in settings_mapping:
            try:
                settings[setting_key] = self.inputs[input_key].value()
            except (ValueError, KeyError) as e:
                default_value = self.DEFAULT_SETTINGS[setting_key]
                settings[setting_key] = default_value
                logger.warning(f"Using default value for {setting_key}: {default_value}")
        
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