"""
Measurement Controls Widget for the AFS Tracking System.
Provides controls for function generator and other measurement hardware.
"""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFrame, QWidget, QCheckBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from src.utils.logger import get_logger
from src.controllers.function_generator_controller import FunctionGeneratorController
from src.utils.status_display import StatusDisplay

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
        self.default_frequency = 14.0  # MHz (optimal frequency)
        self.default_amplitude = 4.0   # Vpp
        
        # Smooth control improvements
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._apply_settings_debounced)
        self._debounce_delay = 300  # ms - prevents excessive VISA commands
        
        # Settings cache to prevent unnecessary hardware updates
        self._cached_frequency = self.default_frequency
        self._cached_amplitude = self.default_amplitude
        self._output_enabled = False
        self._pending_update = False
        
        self._init_ui()
        self._initialize_function_generator()
        
        # Set initial status
        if hasattr(self, 'fg_status_display'):
            self.fg_status_display.set_status("Ready")

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
        layout.setSpacing(8)  # Consistent spacing between rows
        
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
        self.fg_toggle_button.setToolTip("Toggle function generator output ON/OFF")
        self.fg_toggle_button.clicked.connect(self._on_fg_toggle)
        
        # Add status display with circle indicator
        self.fg_status_display = StatusDisplay()
        self.fg_status_display.set_status("OFF")
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.fg_toggle_button)
        row_layout.addWidget(self.fg_status_display, 1)
        
        layout.addLayout(row_layout)
    
    def _add_frequency_row(self, layout):
        """Add frequency control row."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Frequency (MHz):")
        label.setMinimumWidth(150)
        
        self.frequency_edit = QLineEdit()
        self.frequency_edit.setText(str(self.default_frequency))
        self.frequency_edit.setFixedWidth(80)
        self.frequency_edit.setPlaceholderText("0.1-30.0")
        self.frequency_edit.setToolTip("Enter frequency in MHz (0.1 to 30.0)\nChanges apply automatically with 300ms debouncing")
        self.frequency_edit.textChanged.connect(self._on_frequency_changed)
        self.frequency_edit.editingFinished.connect(self._on_frequency_enter)
        
        spacer = QLabel("")
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.frequency_edit)
        row_layout.addWidget(spacer, 1)
        
        layout.addLayout(row_layout)
    
    def _add_amplitude_row(self, layout):
        """Add amplitude control row."""
        row_layout = QHBoxLayout()
        
        label = QLabel("Amplitude (Vpp):")
        label.setMinimumWidth(150)
        
        self.amplitude_edit = QLineEdit()
        self.amplitude_edit.setText(str(self.default_amplitude))
        self.amplitude_edit.setFixedWidth(80)
        self.amplitude_edit.setPlaceholderText("0.1-20.0")
        self.amplitude_edit.setToolTip("Enter amplitude in Vpp (0.1 to 20.0)\nChanges apply automatically with 300ms debouncing")
        self.amplitude_edit.textChanged.connect(self._on_amplitude_changed)
        self.amplitude_edit.editingFinished.connect(self._on_amplitude_enter)
        
        spacer = QLabel("")
        
        row_layout.addWidget(label)
        row_layout.addWidget(self.amplitude_edit)
        row_layout.addWidget(spacer, 1)
        
        layout.addLayout(row_layout)

    def _initialize_function_generator(self):
        """Initialize the function generator controller."""
        try:
            self.fg_controller = FunctionGeneratorController()
            
            # Attempt auto-connection silently
            if self.fg_controller.connect():
                pass  # Connected successfully
            else:
                pass  # Not available - will try when needed
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
                self.fg_status_display.set_status("Connection Failed")
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
                self.fg_status_display.set_status("Error")
                self._output_enabled = False
            else:
                logger.info(f"Function generator ON: {frequency:.3f} MHz, {amplitude:.2f} Vpp")
                self.fg_toggle_button.setText("ON")
                self.fg_status_display.set_status(f"ON @ {frequency:.1f} MHz")
                self._output_enabled = True
                self._cached_frequency = frequency
                self._cached_amplitude = amplitude
        else:
            # Turn off
            if self.fg_controller and self.fg_controller.is_connected():
                success = self.fg_controller.stop_all_outputs()
                if success:
                    logger.info("Function generator OFF")
                    self.fg_status_display.set_status("OFF")
                else:
                    logger.error("Failed to disable function generator")
                    self.fg_status_display.set_status("Error")
            else:
                self.fg_status_display.set_status("OFF")
            
            self.fg_toggle_button.setText("OFF")
            self._output_enabled = False
            self._update_timer.stop()  # Stop any pending updates
        
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
                # Update status display with new frequency
                if self._output_enabled:
                    self.fg_status_display.set_status(f"ON @ {frequency:.1f} MHz")
        
        # Emit signal
        try:
            frequency = float(self.frequency_edit.text())
            amplitude = float(self.amplitude_edit.text())
        except ValueError:
            frequency = self.default_frequency
            amplitude = self.default_amplitude
        self.function_generator_settings_changed.emit(frequency, amplitude)
    
    def _apply_settings_debounced(self):
        """Apply settings after debounce delay to reduce VISA command frequency."""
        if not self._pending_update or not self._output_enabled:
            return
            
        try:
            frequency = float(self.frequency_edit.text())
            amplitude = float(self.amplitude_edit.text())
            
            # Validate ranges with user-friendly corrections
            frequency = max(13.0, min(15.0, frequency))  # 13-15 MHz range
            amplitude = max(0.1, min(20.0, amplitude))
            
            # Only update hardware if values actually changed
            if (abs(frequency - self._cached_frequency) > 0.001 or 
                abs(amplitude - self._cached_amplitude) > 0.01):
                
                if self._ensure_connection():
                    success = self.fg_controller.output_sine_wave(amplitude, frequency, channel=1)
                    if success:
                        self._cached_frequency = frequency
                        self._cached_amplitude = amplitude
                        logger.info(f"Settings applied: {frequency:.3f} MHz, {amplitude:.2f} Vpp")
                        
                        # Emit signal for timeline logging
                        self.function_generator_settings_changed.emit(frequency, amplitude)
                    else:
                        logger.error("Failed to apply function generator settings")
                        
        except ValueError as e:
            logger.warning(f"Invalid input values: {e}")
            self._reset_to_defaults()
        finally:
            self._pending_update = False
    
    def _reset_to_defaults(self):
        """Reset input fields to default values."""
        self.frequency_edit.setText(str(self.default_frequency))
        self.amplitude_edit.setText(str(self.default_amplitude))
    
    def _set_input_valid(self, input_field, is_valid=True):
        """Set visual feedback for input field validation."""
        if is_valid:
            input_field.setStyleSheet("")  # Default style
        else:
            input_field.setStyleSheet("QLineEdit { border: 2px solid red; }")
    
    def _validate_frequency_input(self, text):
        """Validate frequency input and provide visual feedback."""
        try:
            freq = float(text)
            is_valid = 13.0 <= freq <= 15.0  # 13-15 MHz range
            self._set_input_valid(self.frequency_edit, is_valid)
            return is_valid
        except ValueError:
            self._set_input_valid(self.frequency_edit, False)
            return False
    
    def _validate_amplitude_input(self, text):
        """Validate amplitude input and provide visual feedback."""
        try:
            amp = float(text)
            is_valid = 0.1 <= amp <= 20.0
            self._set_input_valid(self.amplitude_edit, is_valid)
            return is_valid
        except ValueError:
            self._set_input_valid(self.amplitude_edit, False)
            return False
    
    def _on_frequency_changed(self):
        """Handle real-time text changes in frequency field with debouncing."""
        text = self.frequency_edit.text()
        self._validate_frequency_input(text)
        
        if self._output_enabled:
            self._pending_update = True
            self._update_timer.start(self._debounce_delay)
    
    def _on_frequency_enter(self):
        """Handle Enter key press in frequency field for immediate update."""
        self._update_timer.stop()  # Cancel any pending debounced update
        self._pending_update = True
        self._apply_settings_debounced()  # Apply immediately
    
    def _on_amplitude_changed(self):
        """Handle real-time text changes in amplitude field with debouncing."""
        text = self.amplitude_edit.text()
        self._validate_amplitude_input(text)
        
        if self._output_enabled:
            self._pending_update = True
            self._update_timer.start(self._debounce_delay)
    
    def _on_amplitude_enter(self):
        """Handle Enter key press in amplitude field for immediate update."""
        self._update_timer.stop()  # Cancel any pending debounced update
        self._pending_update = True
        self._apply_settings_debounced()  # Apply immediately

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