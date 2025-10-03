"""
Measurement Controls Widget for the AFS Tracking System.
Provides controls for function generator and other measurement hardware.
"""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFrame, QWidget, QCheckBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.utils.logger import get_logger
from src.controllers.function_generator_controller import FunctionGeneratorController

logger = get_logger("measurement_controls")


class MeasurementControlsWidget(QGroupBox):
    """Widget for controlling measurement hardware including function generator."""
    
    # Signals for function generator control
    function_generator_toggled = pyqtSignal(bool)  # on/off state
    function_generator_settings_changed = pyqtSignal(float, float)  # frequency_mhz, amplitude
    
    def __init__(self, parent=None):
        super().__init__("Measurement Controls", parent)
        
        # Function generator controller
        self.fg_controller = None
        
        # Default settings
        self.default_frequency = 14.0  # MHz
        self.default_amplitude = 4.0   # Vpp
        
        self._init_ui()
        self._initialize_function_generator()

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 24, 8, 8)
        
        # Create function generator frame
        fg_frame = self._create_function_generator_frame()
        main_layout.addWidget(fg_frame)

    def _create_function_generator_frame(self):
        """Create the main function generator frame."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Add function generator section
        fg_section = self._create_function_generator_section()
        layout.addWidget(fg_section)
        layout.addStretch(1)
        
        return frame

    def _create_function_generator_section(self):
        """Create the function generator control section."""
        section = QWidget()
        layout = QVBoxLayout(section)
        
        # Create all control rows
        self._add_connection_row(layout)
        self._add_output_row(layout)
        
        layout.addSpacing(20)  # Separator
        
        self._add_frequency_row(layout)
        self._add_amplitude_row(layout)
        
        return section
    
    def _add_connection_row(self, layout):
        """Add connection status row with label and connect button."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Status:")
        label.setMinimumWidth(150)
        
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.connect_button = QPushButton("Connect")
        self.connect_button.setFixedWidth(60)
        self.connect_button.clicked.connect(self._on_connect_clicked)
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.status_label, 1)
        row_layout.addWidget(self.connect_button)
        
        layout.addLayout(row_layout)
    
    def _add_output_row(self, layout):
        """Add output enable/disable row."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Output:")
        label.setMinimumWidth(150)
        
        self.fg_enable_checkbox = QCheckBox("Enable Function Generator")
        self.fg_enable_checkbox.stateChanged.connect(self._on_fg_toggle)
        
        spacer = QLabel("")
        spacer.setFixedWidth(60)
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.fg_enable_checkbox, 1)
        row_layout.addWidget(spacer)
        
        layout.addLayout(row_layout)
    
    def _add_frequency_row(self, layout):
        """Add frequency control row."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Frequency:")
        label.setMinimumWidth(150)
        
        self.frequency_spinbox = QDoubleSpinBox()
        self.frequency_spinbox.setRange(0.1, 200.0)  # 0.1 MHz to 200 MHz
        self.frequency_spinbox.setSingleStep(0.1)
        self.frequency_spinbox.setSuffix(" MHz")
        self.frequency_spinbox.setDecimals(3)
        self.frequency_spinbox.setValue(self.default_frequency)
        self.frequency_spinbox.valueChanged.connect(self._on_settings_changed)
        
        spacer = QLabel("")
        spacer.setFixedWidth(60)
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.frequency_spinbox, 1)
        row_layout.addWidget(spacer)
        
        layout.addLayout(row_layout)
    
    def _add_amplitude_row(self, layout):
        """Add amplitude control row."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Amplitude:")
        label.setMinimumWidth(150)
        
        self.amplitude_spinbox = QDoubleSpinBox()
        self.amplitude_spinbox.setRange(0.1, 20.0)  # 0.1 V to 20 V peak-to-peak
        self.amplitude_spinbox.setSingleStep(0.1)
        self.amplitude_spinbox.setSuffix(" Vpp")
        self.amplitude_spinbox.setDecimals(2)
        self.amplitude_spinbox.setValue(self.default_amplitude)
        self.amplitude_spinbox.valueChanged.connect(self._on_settings_changed)
        
        spacer = QLabel("")
        spacer.setFixedWidth(60)
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.amplitude_spinbox, 1)
        row_layout.addWidget(spacer)
        
        layout.addLayout(row_layout)

    def _initialize_function_generator(self):
        """Initialize the function generator controller."""
        try:
            self.fg_controller = FunctionGeneratorController()
            logger.info("Function generator controller created")
            
            # Attempt auto-connection
            if self.fg_controller.connect():
                logger.info("Function generator auto-connected successfully")
                self._update_connection_status()
            else:
                logger.info("Function generator auto-connection failed - manual connection available")
                self._update_connection_status()
        except Exception as e:
            logger.error(f"Failed to create function generator controller: {e}")
            self.fg_controller = None
            self._update_connection_status()
    
    def _on_connect_clicked(self):
        """Handle connect button click."""
        if not self.fg_controller:
            # Retry creating the controller
            self._initialize_function_generator()
            return
        
        if self.fg_controller.is_connected():
            # Disconnect
            self.fg_controller.disconnect()
            logger.info("Function generator disconnected")
            self._update_connection_status()
        else:
            # Connect
            if self.fg_controller.connect():
                logger.info("Function generator connected successfully")
                self._update_connection_status()
            else:
                logger.error("Failed to connect to function generator")
                self._update_connection_status()
    
    def _update_connection_status(self):
        """Update the connection status display."""
        if self.fg_controller and self.fg_controller.is_connected():
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.connect_button.setText("Disconnect")
            
            # Enable controls
            self.fg_enable_checkbox.setEnabled(True)
            self.frequency_spinbox.setEnabled(True)
            self.amplitude_spinbox.setEnabled(True)
        elif not self.fg_controller:
            self.status_label.setText("Controller Error")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.connect_button.setText("Retry")
            
            # Disable controls
            self.fg_enable_checkbox.setChecked(False)
            self.fg_enable_checkbox.setEnabled(False)
            self.frequency_spinbox.setEnabled(False)
            self.amplitude_spinbox.setEnabled(False)
        else:
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.connect_button.setText("Connect")
            
            # Disable controls and turn off output
            self.fg_enable_checkbox.setChecked(False)
            self.fg_enable_checkbox.setEnabled(False)
            self.frequency_spinbox.setEnabled(False)
            self.amplitude_spinbox.setEnabled(False)
    
    def _on_fg_toggle(self, state):
        """Handle function generator on/off toggle."""
        is_enabled = state == Qt.Checked
        
        if not self.fg_controller or not self.fg_controller.is_connected():
            logger.warning("Function generator not connected")
            self.fg_enable_checkbox.setChecked(False)
            return
        
        if is_enabled:
            # Turn on with current settings
            frequency = self.frequency_spinbox.value()
            amplitude = self.amplitude_spinbox.value()
            
            success = self.fg_controller.output_sine_wave(amplitude, frequency, channel=1)
            if not success:
                logger.error("Failed to enable function generator output")
                self.fg_enable_checkbox.setChecked(False)
            else:
                logger.info(f"Function generator enabled: {frequency:.3f} MHz, {amplitude:.2f} Vpp")
        else:
            # Turn off
            success = self.fg_controller.stop_all_outputs()
            if not success:
                logger.error("Failed to disable function generator output")
            else:
                logger.info("Function generator disabled")
        
        # Emit signal
        self.function_generator_toggled.emit(is_enabled and self.fg_controller.is_connected())
    
    def _on_settings_changed(self):
        """Handle frequency or amplitude changes."""
        if not self.fg_controller or not self.fg_controller.is_connected():
            return
        
        # If output is enabled, update immediately
        if self.fg_enable_checkbox.isChecked():
            frequency = self.frequency_spinbox.value()
            amplitude = self.amplitude_spinbox.value()
            
            success = self.fg_controller.output_sine_wave(amplitude, frequency, channel=1)
            if success:
                logger.info(f"Function generator settings updated: {frequency:.3f} MHz, {amplitude:.2f} Vpp")
            else:
                logger.error("Failed to update function generator settings")
        
        # Emit signal
        frequency = self.frequency_spinbox.value()
        amplitude = self.amplitude_spinbox.value()
        self.function_generator_settings_changed.emit(frequency, amplitude)

    # Public methods for external control
    
    def get_function_generator_controller(self):
        """Get the function generator controller instance."""
        return self.fg_controller
    
    def is_function_generator_enabled(self) -> bool:
        """Check if function generator output is enabled."""
        return self.fg_enable_checkbox.isChecked()
    
    def get_frequency(self) -> float:
        """Get current frequency setting in MHz."""
        return self.frequency_spinbox.value()
    
    def get_amplitude(self) -> float:
        """Get current amplitude setting in Vpp."""
        return self.amplitude_spinbox.value()
    
    def set_frequency(self, frequency_mhz: float):
        """Set frequency in MHz."""
        self.frequency_spinbox.setValue(frequency_mhz)
    
    def set_amplitude(self, amplitude_vpp: float):
        """Set amplitude in Vpp."""
        self.amplitude_spinbox.setValue(amplitude_vpp)
    
    def enable_function_generator(self, enable: bool = True):
        """Enable or disable function generator output."""
        self.fg_enable_checkbox.setChecked(enable)
    
    def get_function_generator_status(self) -> dict:
        """Get function generator status information."""
        status = {
            'enabled': self.is_function_generator_enabled(),
            'frequency_mhz': self.get_frequency(),
            'amplitude_vpp': self.get_amplitude(),
            'connected': False
        }
        
        if self.fg_controller:
            controller_status = self.fg_controller.get_output_status()
            status.update(controller_status)
        
        return status

    def cleanup(self):
        """Clean up resources when widget is destroyed."""
        if self.fg_controller:
            self.fg_controller.disconnect()
            logger.info("Function generator controller cleanup completed")