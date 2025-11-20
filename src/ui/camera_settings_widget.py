"""
Minimal Camera Settings Widget for AFS Acquisition.
Ultra-simple design with only essential controls and 3 buttons.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QGroupBox, QLineEdit, QPushButton, QSpacerItem,
    QSizePolicy, QLabel
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
        self.setFixedSize(320, 280)
        self.setModal(True)
        
        # Use class constant for defaults
        self.current_settings = self.DEFAULT_SETTINGS.copy()
        
        self.init_ui()
        self.load_current_settings()
    
    def init_ui(self):
        """Initialize minimal UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        
        # Title
        title_label = QLabel("Camera Parameters")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Settings group
        settings_group = QGroupBox()
        settings_layout = QFormLayout(settings_group)
        settings_layout.setSpacing(6)
        settings_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        
        # Create parameter controls using helper method to reduce redundancy
        parameter_configs = [
            ("exposure", "Exposure (ms):", "15.0"),    # Stable exposure for 30 FPS
            ("gain", "Gain:", "2"),                     # Moderate gain
            ("fps", "Frame Rate (fps):", "30.0"),      # 30 FPS baseline
            ("brightness", "Brightness:", "50"),
            ("contrast", "Contrast:", "50"),
            ("saturation", "Saturation:", "50")
        ]
        
        # Store input widgets for easy access
        self.inputs = {}
        
        for param_name, label_text, placeholder in parameter_configs:
            input_widget = self._create_parameter_input(placeholder)
            self.inputs[param_name] = input_widget
            settings_layout.addRow(label_text, input_widget)
        
        main_layout.addWidget(settings_group)
        
        # Spacer
        main_layout.addItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Button row - create using helper method
        button_layout = self._create_button_row()
        main_layout.addLayout(button_layout)
    
    def _create_parameter_input(self, placeholder: str) -> QLineEdit:
        """Create a standardized parameter input widget."""
        input_widget = QLineEdit()
        input_widget.setFixedWidth(80)
        input_widget.setPlaceholderText(placeholder)
        return input_widget
    
    def _create_button_row(self) -> QHBoxLayout:
        """Create the button row with consistent styling."""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        # Button configurations
        button_configs = [
            ("Apply", self.apply_or_reset),
            ("Reconnect", self.reconnect_camera),
            ("Save", self.save_and_close)
        ]
        
        button_layout.addStretch()
        
        for text, callback in button_configs:
            button = QPushButton(text)
            button.setFixedWidth(70)
            button.clicked.connect(callback)
            button_layout.addWidget(button)
            
            # Store specific buttons for later reference
            if text == "Apply":
                self.apply_button = button
        
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
            ('exposure', 'exposure_ms', lambda v: f"{v:.2f}"),
            ('gain', 'gain_master', str),
            ('fps', 'frame_rate_fps', lambda v: f"{v:.2f}"),
            ('brightness', 'brightness', str),
            ('contrast', 'contrast', str),
            ('saturation', 'saturation', str)
        ]
        
        for input_key, setting_key, formatter in settings_mapping:
            if input_key in self.inputs:
                value = self.current_settings.get(setting_key, self.DEFAULT_SETTINGS[setting_key])
                self.inputs[input_key].setText(formatter(value))
    
    def get_settings_from_ui(self) -> Dict[str, Any]:
        """Get settings from UI controls with proper parsing and error handling."""
        # Mapping for parsing different types with defaults
        parse_mapping = [
            ('exposure', 'exposure_ms', lambda x: round(float(x), 2), 15.0),
            ('gain', 'gain_master', int, 2),
            ('fps', 'frame_rate_fps', lambda x: round(float(x), 2), 30.0),
            ('brightness', 'brightness', int, 50),
            ('contrast', 'contrast', int, 75),
            ('saturation', 'saturation', int, 70)
        ]
        
        settings = {}
        
        for input_key, setting_key, parser, default_value in parse_mapping:
            try:
                text_value = self.inputs[input_key].text() or str(default_value)
                settings[setting_key] = parser(text_value)
            except (ValueError, KeyError):
                settings[setting_key] = default_value
                logger.warning(f"Using default value for {setting_key}: {default_value}")
        
        return settings
    
    def apply_or_reset(self):
        """Apply settings or reset to defaults (toggles function)."""
        if self.apply_button.text() == "Apply":
            self.apply_settings()
            self.apply_button.setText("Reset")
        else:
            self.load_defaults()
            self.apply_button.setText("Apply")
    
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