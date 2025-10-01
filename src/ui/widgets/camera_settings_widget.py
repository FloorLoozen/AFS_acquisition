"""
Camera Settings Widget for the AFS Tracking System.
Provides intuitive controls for camera configuration.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QSlider, QGroupBox, QGridLayout,
    QMessageBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from src.utils.logger import get_logger

logger = get_logger("camera_settings")


class CameraSettingsWidget(QDialog):
    """
    Camera settings dialog with intuitive controls for exposure, gain, frame rate, etc.
    """
    
    settings_changed = pyqtSignal(dict)  # Emitted when settings change
    
    def __init__(self, camera_controller=None, parent=None):
        super().__init__(parent)
        self.camera_controller = camera_controller
        self.setWindowTitle("Camera Settings")
        self.setModal(True)
        self.resize(400, 400)
        
        # Current settings cache
        self._current_settings = {}
        
        self.init_ui()
        self.load_current_settings()
        
        # Auto-refresh timer for live settings display
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_current_values)
        self.refresh_timer.start(1000)  # Refresh every second
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Exposure settings
        exposure_group = QGroupBox("Exposure")
        exposure_layout = QFormLayout(exposure_group)
        exposure_layout.setContentsMargins(8, 8, 8, 8)
        exposure_layout.setVerticalSpacing(8)
        
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.1, 1000.0)  # 0.1ms to 1000ms
        self.exposure_spin.setSingleStep(0.1)
        self.exposure_spin.setSuffix(" ms")
        self.exposure_spin.setValue(33.33)
        self.exposure_spin.valueChanged.connect(self.on_exposure_changed)
        exposure_layout.addRow("Exposure Time:", self.exposure_spin)
        
        # Exposure slider for quick adjustment
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 1000)  # 0.1ms to 100ms (scaled by 10)
        self.exposure_slider.setValue(333)
        self.exposure_slider.valueChanged.connect(self.on_exposure_slider_changed)
        exposure_layout.addRow("Quick Adjust:", self.exposure_slider)
        
        layout.addWidget(exposure_group)
        
        # Gain settings
        gain_group = QGroupBox("Gain")
        gain_layout = QFormLayout(gain_group)
        gain_layout.setContentsMargins(8, 8, 8, 8)
        gain_layout.setVerticalSpacing(8)
        
        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 100)
        self.gain_spin.setValue(0)
        self.gain_spin.valueChanged.connect(self.on_gain_changed)
        gain_layout.addRow("Master Gain:", self.gain_spin)
        
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 100)
        self.gain_slider.setValue(0)
        self.gain_slider.valueChanged.connect(self.on_gain_slider_changed)
        gain_layout.addRow("Quick Adjust:", self.gain_slider)
        
        layout.addWidget(gain_group)
        
        # Current status
        status_group = QGroupBox("Current Status")
        status_layout = QFormLayout(status_group)
        status_layout.setContentsMargins(8, 8, 8, 8)
        status_layout.setVerticalSpacing(6)
        
        self.current_exposure_label = QLabel("--")
        self.current_gain_label = QLabel("--")
        self.connection_status_label = QLabel("--")
        
        status_layout.addRow("Current Exposure:", self.current_exposure_label)
        status_layout.addRow("Current Gain:", self.current_gain_label)
        status_layout.addRow("Connection:", self.connection_status_label)
        
        layout.addWidget(status_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        self.refresh_btn = QPushButton("Refresh Settings")
        self.refresh_btn.clicked.connect(self.load_current_settings)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_all_settings)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.apply_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def set_camera_controller(self, controller):
        """Set the camera controller instance."""
        self.camera_controller = controller
        self.load_current_settings()
    
    def load_current_settings(self):
        """Load current settings from camera controller."""
        if not self.camera_controller:
            self.connection_status_label.setText("No controller")
            return
        
        try:
            settings = self.camera_controller.get_camera_settings()
            self._current_settings = settings
            
            # Update UI with current settings
            if 'exposure_ms' in settings and settings['exposure_ms'] != 'unavailable':
                self.exposure_spin.setValue(float(settings['exposure_ms']))
                self.current_exposure_label.setText(f"{settings['exposure_ms']:.1f} ms")
            
            if 'gain_master' in settings and settings['gain_master'] != 'unavailable':
                self.gain_spin.setValue(int(settings['gain_master']))
                self.current_gain_label.setText(str(settings['gain_master']))
            
            # Update connection status
            if self.camera_controller.is_initialized:
                if self.camera_controller.use_test_pattern:
                    self.connection_status_label.setText("Test Pattern Mode")
                else:
                    self.connection_status_label.setText("Hardware Connected")
            else:
                self.connection_status_label.setText("Not Initialized")
                
        except Exception as e:
            logger.error(f"Failed to load camera settings: {e}")
            self.connection_status_label.setText("Error loading settings")
    
    def refresh_current_values(self):
        """Refresh the current status display."""
        if not self.camera_controller:
            return
            
        try:
            # Update current settings display if needed
            pass
        except Exception:
            pass
    
    def get_current_settings(self):
        """Get settings as configured in the UI."""
        return {
            'exposure_ms': self.exposure_spin.value(),
            'gain_master': self.gain_spin.value(),
        }
    
    def on_exposure_changed(self, value):
        """Handle exposure spinbox change."""
        self.exposure_slider.blockSignals(True)
        self.exposure_slider.setValue(int(value * 10))  # Scale for slider
        self.exposure_slider.blockSignals(False)
        self.apply_setting('exposure_ms', value)
    
    def on_exposure_slider_changed(self, value):
        """Handle exposure slider change."""
        exposure_value = value / 10.0  # Scale back from slider
        self.exposure_spin.blockSignals(True)
        self.exposure_spin.setValue(exposure_value)
        self.exposure_spin.blockSignals(False)
        self.apply_setting('exposure_ms', exposure_value)
    
    def on_gain_changed(self, value):
        """Handle gain change."""
        self.gain_slider.blockSignals(True)
        self.gain_slider.setValue(value)
        self.gain_slider.blockSignals(False)
        self.apply_setting('gain_master', value)
    
    def on_gain_slider_changed(self, value):
        """Handle gain slider change."""
        self.gain_spin.blockSignals(True)
        self.gain_spin.setValue(value)
        self.gain_spin.blockSignals(False)
        self.apply_setting('gain_master', value)
    

    

    

    
    def apply_setting(self, setting_name, value):
        """Apply a single setting to the camera."""
        if not self.camera_controller:
            return
        
        try:
            # Apply the setting using controller methods
            success = False
            if setting_name == 'exposure_ms':
                success = self.camera_controller.set_exposure(value)
            elif setting_name == 'gain_master':
                success = self.camera_controller.set_gain(value)
            else:
                logger.warning(f"Unknown setting: {setting_name}")
                return
            
            if success:
                logger.info(f"Successfully applied {setting_name} = {value}")
            else:
                logger.warning(f"Failed to apply {setting_name} = {value}")
            
            # Emit signal for other components
            settings = {setting_name: value}
            self.settings_changed.emit(settings)
            
        except Exception as e:
            logger.error(f"Error applying setting {setting_name}: {e}")
            QMessageBox.warning(self, "Setting Error", 
                              f"Failed to apply {setting_name}: {str(e)}")
    
    def apply_all_settings(self):
        """Apply all current settings to the camera."""
        if not self.camera_controller:
            QMessageBox.warning(self, "No Camera", "No camera controller available")
            return
        
        settings = self.get_current_settings()
        
        try:
            # Apply each setting
            for key, value in settings.items():
                self.apply_setting(key, value)
            
            QMessageBox.information(self, "Settings Applied", 
                                  "All camera settings have been applied successfully.")
            
        except Exception as e:
            logger.error(f"Failed to apply settings: {e}")
            QMessageBox.warning(self, "Apply Error", 
                              f"Failed to apply settings: {str(e)}")
    
    def closeEvent(self, event):
        """Handle dialog close."""
        self.refresh_timer.stop()
        super().closeEvent(event)