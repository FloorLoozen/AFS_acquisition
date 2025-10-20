"""
Minimal Camera Settings Widget for the AFS Tracking System.
Ultra-simple design with only essential controls and 3 buttons.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QGroupBox, QLineEdit, QPushButton, QSpacerItem,
    QSizePolicy, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.utils.logger import get_logger

logger = get_logger("camera_settings")


class CameraSettingsWidget(QDialog):
    """
    Minimal camera settings dialog.
    Only essential controls with clean 3-button interface.
    """
    
    settings_applied = pyqtSignal(dict)
    
    def __init__(self, camera_controller=None, parent=None):
        super().__init__(parent)
        self.camera = camera_controller
        
        self.setWindowTitle("Camera Settings")
        self.setFixedSize(320, 280)
        self.setModal(True)
        
        # Default settings (brighter values)
        self.default_settings = {
            'exposure_ms': 15.0,
            'gain_master': 2,     # Use integer for gain
            'frame_rate_fps': 30.0,
            'brightness': 50,     # Standard brightness
            'contrast': 75,       # Higher contrast  
            'saturation': 70      # More vivid
        }
        
        # Current settings
        self.current_settings = self.default_settings.copy()
        
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
        
        # Essential parameter controls (compact)
        self.exposure_input = QLineEdit()
        self.exposure_input.setFixedWidth(80)
        self.exposure_input.setPlaceholderText("15.0")
        settings_layout.addRow("Exposure (ms):", self.exposure_input)
        
        self.gain_input = QLineEdit()
        self.gain_input.setFixedWidth(80) 
        self.gain_input.setPlaceholderText("2")
        settings_layout.addRow("Gain:", self.gain_input)
        
        self.fps_input = QLineEdit()
        self.fps_input.setFixedWidth(80)
        self.fps_input.setPlaceholderText("30.0")
        settings_layout.addRow("Frame Rate (fps):", self.fps_input)
        
        self.brightness_input = QLineEdit()
        self.brightness_input.setFixedWidth(80)
        self.brightness_input.setPlaceholderText("50")
        settings_layout.addRow("Brightness:", self.brightness_input)
        
        self.contrast_input = QLineEdit()
        self.contrast_input.setFixedWidth(80)
        self.contrast_input.setPlaceholderText("75")
        settings_layout.addRow("Contrast:", self.contrast_input)
        
        self.saturation_input = QLineEdit()
        self.saturation_input.setFixedWidth(80)
        self.saturation_input.setPlaceholderText("70")
        settings_layout.addRow("Saturation:", self.saturation_input)
        
        main_layout.addWidget(settings_group)
        
        # Spacer
        main_layout.addItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Button layout (exactly 3 buttons)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        # Apply/Reset button
        self.apply_button = QPushButton("Apply")
        self.apply_button.setFixedWidth(70)
        self.apply_button.clicked.connect(self.apply_or_reset)
        
        # Reconnect button  
        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.setFixedWidth(70)
        self.reconnect_button.clicked.connect(self.reconnect_camera)
        
        # Save button (close dialog)
        self.save_button = QPushButton("Save")
        self.save_button.setFixedWidth(70)
        self.save_button.clicked.connect(self.save_and_close)
        
        button_layout.addStretch()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reconnect_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
    
    def load_current_settings(self):
        """Load current camera settings if available."""
        if self.camera and hasattr(self.camera, 'get_camera_settings'):
            try:
                current = self.camera.get_camera_settings()
                if current:
                    # Only update with relevant settings that we can control
                    for key in self.default_settings.keys():
                        if key in current:
                            self.current_settings[key] = current[key]
                    logger.info(f"Loaded camera settings: {self.current_settings}")
            except Exception as e:
                logger.warning(f"Failed to load camera settings: {e}")
        
        self.update_ui_from_settings()
    
    def load_defaults(self):
        """Load default settings."""
        self.current_settings = self.default_settings.copy()
        self.update_ui_from_settings()
    
    def update_ui_from_settings(self):
        """Update UI controls from current settings."""
        self.exposure_input.setText(f"{self.current_settings.get('exposure_ms', 15.0):.2f}")
        self.gain_input.setText(str(self.current_settings.get('gain_master', 2)))
        self.fps_input.setText(f"{self.current_settings.get('frame_rate_fps', 30.0):.2f}")
        self.brightness_input.setText(str(self.current_settings.get('brightness', 50)))
        self.contrast_input.setText(str(self.current_settings.get('contrast', 75)))
        self.saturation_input.setText(str(self.current_settings.get('saturation', 70)))
    
    def get_settings_from_ui(self) -> dict:
        """Get settings from UI controls with proper rounding."""
        try:
            settings = {}
            
            # Parse numeric values with error handling and rounding
            try:
                settings['exposure_ms'] = round(float(self.exposure_input.text() or "15.0"), 2)
            except ValueError:
                settings['exposure_ms'] = 15.0
            
            try:
                settings['gain_master'] = int(self.gain_input.text() or "2")
            except ValueError:
                settings['gain_master'] = 2
            
            try:
                settings['frame_rate_fps'] = round(float(self.fps_input.text() or "30.0"), 2)
            except ValueError:
                settings['frame_rate_fps'] = 30.0
            
            try:
                settings['brightness'] = int(self.brightness_input.text() or "50")
            except ValueError:
                settings['brightness'] = 50
            
            try:
                settings['contrast'] = int(self.contrast_input.text() or "75")
            except ValueError:
                settings['contrast'] = 75
            
            try:
                settings['saturation'] = int(self.saturation_input.text() or "70")
            except ValueError:
                settings['saturation'] = 70
            
            return settings
            
        except Exception as e:
            logger.error(f"Error parsing settings: {e}")
            return self.default_settings.copy()
    
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