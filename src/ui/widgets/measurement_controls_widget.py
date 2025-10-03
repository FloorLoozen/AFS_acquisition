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
        self.default_frequency = 10.0  # MHz
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
        
        # Create simplified control rows
        self._add_enable_row(layout)
        self._add_frequency_row(layout)
        self._add_amplitude_row(layout)
        
        return section
    
    def _add_enable_row(self, layout):
        """Add function generator on/off button row."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Function Generator:")
        label.setMinimumWidth(150)
        
        self.fg_toggle_button = QPushButton("OFF")
        self.fg_toggle_button.setCheckable(True)
        self.fg_toggle_button.setFixedWidth(60)
        self.fg_toggle_button.clicked.connect(self._on_fg_toggle)
        
        spacer = QLabel("")
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.fg_toggle_button)
        row_layout.addWidget(spacer, 1)
        
        layout.addLayout(row_layout)
    
    def _add_frequency_row(self, layout):
        """Add frequency control row."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Frequency:")
        label.setMinimumWidth(150)
        
        self.frequency_edit = QLineEdit()
        self.frequency_edit.setText(str(self.default_frequency))
        self.frequency_edit.setFixedWidth(80)
        self.frequency_edit.setPlaceholderText("0.1-10.0")
        self.frequency_edit.textChanged.connect(self._on_frequency_changed)
        self.frequency_edit.editingFinished.connect(self._on_frequency_enter)
        
        mhz_label = QLabel("mhz")
        mhz_label.setMinimumWidth(40)
        
        spacer = QLabel("")
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.frequency_edit)
        row_layout.addWidget(mhz_label)
        row_layout.addWidget(spacer, 1)
        
        layout.addLayout(row_layout)
    
    def _add_amplitude_row(self, layout):
        """Add amplitude control row."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Amplitude:")
        label.setMinimumWidth(150)
        
        self.amplitude_edit = QLineEdit()
        self.amplitude_edit.setText(str(self.default_amplitude))
        self.amplitude_edit.setFixedWidth(80)
        self.amplitude_edit.setPlaceholderText("0.1-20.0")
        self.amplitude_edit.textChanged.connect(self._on_amplitude_changed)
        self.amplitude_edit.editingFinished.connect(self._on_amplitude_enter)
        
        vpp_label = QLabel("vpp")
        vpp_label.setMinimumWidth(40)
        
        spacer = QLabel("")
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.amplitude_edit)
        row_layout.addWidget(vpp_label)
        row_layout.addWidget(spacer, 1)
        
        layout.addLayout(row_layout)

    def _initialize_function_generator(self):
        """Initialize the function generator controller."""
        try:
            self.fg_controller = FunctionGeneratorController()
            logger.info("Function generator controller created")
            
            # Attempt auto-connection silently
            if self.fg_controller.connect():
                logger.info("Function generator connected")
            else:
                logger.info("Function generator not available - will try when needed")
        except Exception as e:
            logger.error(f"Failed to create function generator controller: {e}")
            self.fg_controller = None
    
    def _ensure_connection(self):
        """Ensure function generator is connected, try to reconnect if needed."""
        if not self.fg_controller:
            self._initialize_function_generator()
            return self.fg_controller is not None
        
        if not self.fg_controller.is_connected():
            # Try to reconnect silently
            return self.fg_controller.connect()
        
        return True
    
    def _on_fg_toggle(self, checked):
        """Handle function generator on/off button toggle."""
        if checked:
            # Try to connect if not connected
            if not self._ensure_connection():
                logger.warning("Function generator not available")
                self.fg_toggle_button.setChecked(False)
                return
            
            # Turn on with current settings
            try:
                frequency = float(self.frequency_edit.text())
                amplitude = float(self.amplitude_edit.text())
            except ValueError:
                frequency = self.default_frequency
                amplitude = self.default_amplitude
            
            success = self.fg_controller.output_sine_wave(amplitude, frequency, channel=1)
            if not success:
                logger.error("Failed to enable function generator")
                self.fg_toggle_button.setChecked(False)
                self.fg_toggle_button.setText("OFF")
            else:
                logger.info(f"Function generator ON: {frequency:.3f} mhz, {amplitude:.2f} vpp")
                self.fg_toggle_button.setText("ON")
        else:
            # Turn off
            if self.fg_controller and self.fg_controller.is_connected():
                success = self.fg_controller.stop_all_outputs()
                if success:
                    logger.info("Function generator OFF")
                else:
                    logger.error("Failed to disable function generator")
            
            self.fg_toggle_button.setText("OFF")
        
        # Emit signal
        self.function_generator_toggled.emit(checked and self._ensure_connection())
    
    def _on_settings_changed(self):
        """Handle frequency or amplitude changes."""
        # If output is enabled, update immediately
        if self.fg_toggle_button.isChecked() and self._ensure_connection():
            try:
                frequency = float(self.frequency_edit.text())
                amplitude = float(self.amplitude_edit.text())
            except ValueError:
                frequency = self.default_frequency
                amplitude = self.default_amplitude
            
            success = self.fg_controller.output_sine_wave(amplitude, frequency, channel=1)
            if success:
                logger.info(f"Settings: {frequency:.3f} mhz, {amplitude:.2f} vpp")
        
        # Emit signal
        try:
            frequency = float(self.frequency_edit.text())
            amplitude = float(self.amplitude_edit.text())
        except ValueError:
            frequency = self.default_frequency
            amplitude = self.default_amplitude
        self.function_generator_settings_changed.emit(frequency, amplitude)
    
    def _on_frequency_changed(self):
        """Handle text changes in frequency field"""
        pass  # No real-time validation needed
    
    def _on_frequency_enter(self):
        """Handle Enter key press in frequency field."""
        if self.fg_toggle_button.isChecked() and self._ensure_connection():
            try:
                frequency = float(self.frequency_edit.text())
                amplitude = float(self.amplitude_edit.text())
                
                # Validate range
                if not (0.1 <= frequency <= 10.0):
                    frequency = max(0.1, min(10.0, frequency))
                    self.frequency_edit.setText(str(frequency))
                
                success = self.fg_controller.output_sine_wave(amplitude, frequency, channel=1)
                if success:
                    logger.info(f"Frequency updated: {frequency:.3f} mhz")
            except ValueError:
                self.frequency_edit.setText(str(self.default_frequency))
    
    def _on_amplitude_changed(self):
        """Handle text changes in amplitude field"""
        pass  # No real-time validation needed
    
    def _on_amplitude_enter(self):
        """Handle Enter key press in amplitude field."""
        if self.fg_toggle_button.isChecked() and self._ensure_connection():
            try:
                frequency = float(self.frequency_edit.text())
                amplitude = float(self.amplitude_edit.text())
                
                # Validate range
                if not (0.1 <= amplitude <= 20.0):
                    amplitude = max(0.1, min(20.0, amplitude))
                    self.amplitude_edit.setText(str(amplitude))
                
                success = self.fg_controller.output_sine_wave(amplitude, frequency, channel=1)
                if success:
                    logger.info(f"Amplitude updated: {amplitude:.2f} vpp")
            except ValueError:
                self.amplitude_edit.setText(str(self.default_amplitude))

    # Public methods for external control
    
    def get_function_generator_controller(self):
        """Get the function generator controller instance."""
        return self.fg_controller
    
    def is_function_generator_enabled(self) -> bool:
        """Check if function generator output is enabled."""
        return self.fg_toggle_button.isChecked()
    
    def get_frequency(self) -> float:
        """Get current frequency setting in MHz."""
        try:
            return float(self.frequency_edit.text())
        except ValueError:
            return self.default_frequency
    
    def get_amplitude(self) -> float:
        """Get current amplitude setting in Vpp."""
        try:
            return float(self.amplitude_edit.text())
        except ValueError:
            return self.default_amplitude
    
    def set_frequency(self, frequency_mhz: float):
        """Set frequency in MHz."""
        self.frequency_spinbox.setValue(frequency_mhz)
    
    def set_amplitude(self, amplitude_vpp: float):
        """Set amplitude in Vpp."""
        self.amplitude_spinbox.setValue(amplitude_vpp)
    
    def enable_function_generator(self, enable: bool = True):
        """Enable or disable function generator output."""
        self.fg_toggle_button.setChecked(enable)
    
    def get_function_generator_status(self) -> dict:
        """Get function generator status information."""
        # Auto-check connection status
        connected = self._ensure_connection()
        
        return {
            'enabled': self.is_function_generator_enabled(),
            'frequency_mhz': self.get_frequency(),
            'amplitude_vpp': self.get_amplitude(),
            'connected': connected
        }

    def cleanup(self):
        """Clean up resources when widget is destroyed."""
        if self.fg_controller:
            # Turn off outputs first
            if self.fg_toggle_button.isChecked():
                self.fg_controller.stop_all_outputs()
            # Then disconnect
            self.fg_controller.disconnect()
            logger.info("Function generator cleanup completed")