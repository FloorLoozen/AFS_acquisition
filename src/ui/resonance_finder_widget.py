"""
Resonance Finder Widget for AFS Acquisition.
PyQt5-based resonance frequency analysis tool with frequency sweep and plotting.
"""

import numpy as np
import re
import threading
import time
from typing import Optional, Tuple, List
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QDoubleSpinBox,
    QFrame, QSizePolicy, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.utils.logger import get_logger
from src.controllers.function_generator_controller import FunctionGeneratorController, get_function_generator_controller
from src.controllers.oscilloscope_controller import OscilloscopeController, get_oscilloscope_controller

logger = get_logger("resonance_finder")


class SweepWorker(QThread):
    """Worker thread for performing frequency sweeps."""

    # Signal: times, voltages, scope_limits dict with keys: y_min, y_max, screen_time
    sweep_completed = pyqtSignal(object, object, object)
    sweep_error = pyqtSignal(str)
    
    def __init__(self, funcgen, oscilloscope, amplitude, freq_start, freq_stop, sweep_time):
        super().__init__()
        self.funcgen = funcgen
        self.oscilloscope = oscilloscope
        self.amplitude = amplitude
        self.freq_start = freq_start
        self.freq_stop = freq_stop
        self.sweep_time = sweep_time
    
    def _parse_siglent_value(self, response: str) -> float:
        """Parse a Siglent value response like 'TDIV 1.00E-02S' or 'C1:VDIV 3.20E-02V'."""
        import re
        # Remove any prefix like "C1:VDIV " or "TDIV " to get just the value
        # Look for scientific notation: digits, optional decimal, E, optional sign, digits
        # Followed by optional unit (V, S, Sa/s, etc.)
        match = re.search(r'([+-]?\d+\.?\d*[Ee][+-]?\d+)\s*([A-Za-z/]*)', response)
        if not match:
            # Try simple decimal number
            match = re.search(r'([+-]?\d+\.?\d*)\s*([A-Za-z/]*)', response)
            if not match:
                logger.warning(f"Could not parse value from: {response}")
                return 1.0
        
        value_str = match.group(1)
        unit = match.group(2).upper() if match.group(2) else ''
        
        try:
            value = float(value_str)
        except ValueError:
            logger.warning(f"Could not convert to float: {value_str}")
            return 1.0
        
        # The scientific notation already includes the scale (e.g., 3.20E-02 = 0.032)
        # Units like V, S are base units - no additional multiplier needed
        # Only apply multiplier for explicit unit prefixes like mV, uV, etc.
        # But Siglent uses scientific notation with base units, so usually no multiplier needed
        
        logger.debug(f"Parsed '{response}' -> value={value}, unit='{unit}'")
        return value
    
    def run(self):
        """Execute the frequency sweep in the background thread.
        
        Sequence:
        1. Start function generator sweep (runs continuously)
        2. Wait 1.5 seconds for sweep to run
        3. Read waveform data from oscilloscope (while it's still running)
        4. Stop function generator
        """
        try:
            generator_started = False
            scope = None
            
            if self.oscilloscope and self.oscilloscope.is_connected:
                scope = self.oscilloscope.scope
            
            # Step 1: Start the hardware sweep on the function generator
            if self.funcgen and self.funcgen.is_connected:
                try:
                    success = self.funcgen.sine_frequency_sweep(
                        amplitude=self.amplitude,
                        freq_start=self.freq_start,
                        freq_end=self.freq_stop,
                        sweep_time=self.sweep_time,
                        channel=1,
                    )
                    if not success:
                        self.sweep_error.emit("Failed to start hardware sweep")
                        return
                    generator_started = True
                    logger.info(f"Sweep started: {self.freq_start:.3f} - {self.freq_stop:.3f} MHz, "
                               f"amplitude={self.amplitude:.2f} Vpp, time={self.sweep_time:.2f}s")
                except Exception as e:
                    logger.error(f"Hardware sweep failed to start: {e}")
                    self.sweep_error.emit(f"Hardware sweep error: {e}")
                    return
            else:
                logger.warning("Function generator not connected - using demo data")
            
            # Step 2: Wait 1.5 seconds for sweep to run
            wait_time = 1.5
            logger.info(f"Waiting {wait_time:.1f}s for sweep...")
            time.sleep(wait_time)
            
            # Step 3: Capture waveform data from oscilloscope (don't stop it, just read)
            times = None
            voltages = None
            scope_limits = None
            
            if scope is not None:
                try:
                    # Ensure scope is in remote mode
                    scope.write(":SYSTem:REMote")
                    
                    # Set trigger mode to AUTO so scope continuously triggers
                    scope.write("TRMD AUTO")
                    time.sleep(0.1)
                    
                    # Start acquisition and wait for memory to fill
                    scope.write(":RUN")
                    time.sleep(1.0)  # Wait for scope to start acquiring
                    
                    # Force trigger to ensure we capture data
                    scope.write("FRTR")
                    time.sleep(1.0)  # Wait for acquisition memory to fill
                    
                    # Check trigger status
                    status = scope.query(":TRIGger:STATus?").strip()
                    logger.info(f"Trigger status after RUN+FRTR: {status}")
                    
                    # Query scope settings
                    vdiv_response = scope.query("C1:VDIV?").strip()
                    vdiv_volts = self._parse_siglent_value(vdiv_response)
                    ofst_response = scope.query("C1:OFST?").strip()
                    voffset_volts = self._parse_siglent_value(ofst_response)
                    tdiv_response = scope.query("TDIV?").strip()
                    tdiv_seconds = self._parse_siglent_value(tdiv_response)
                    
                    logger.info(f"Scope settings: VDIV={vdiv_volts*1000:.1f}mV, OFST={voffset_volts*1000:.1f}mV, TDIV={tdiv_seconds*1000:.2f}ms")
                    
                    # Query sample rate to calculate screen samples
                    try:
                        sara_response = scope.query("SARA?").strip()
                        # Parse "SARA 5.00E+07Sa/s" format
                        actual_sample_rate = self._parse_siglent_value(sara_response)
                        logger.info(f"Sample rate: {actual_sample_rate/1e6:.2f} MSa/s")
                    except Exception:
                        actual_sample_rate = 50e6  # Default fallback
                        logger.warning("Could not query sample rate, using 50 MSa/s default")
                    
                    # Calculate screen time and expected screen samples
                    screen_time = 10.0 * tdiv_seconds  # 10 divisions
                    screen_samples = int(screen_time * actual_sample_rate)
                    logger.info(f"Screen: {screen_time*1000:.1f}ms = {screen_samples} samples at {actual_sample_rate/1e6:.1f} MSa/s")
                    
                    # Now stop to freeze the display for reading
                    scope.write(":STOP")
                    time.sleep(0.5)
                    
                    # Increase timeout for large data transfer
                    old_timeout = scope.timeout
                    scope.timeout = 60000  # 60 seconds for large transfers
                    old_chunk = scope.chunk_size
                    scope.chunk_size = 1024 * 1024  # 1MB chunks
                    
                    # Siglent SDS800X HD has a 5M sample transfer limit per request
                    # Read in chunks of 5M to get all screen data
                    chunk_size = 5000000  # 5M samples per transfer
                    all_samples = []
                    
                    try:
                        # Read data in 5M chunks using native Siglent WFSU command
                        samples_read = 0
                        while samples_read < screen_samples:
                            remaining = screen_samples - samples_read
                            points_to_read = min(chunk_size, remaining)
                            
                            # Set up chunk read: SP=sparsing(1=all), NP=num points, FP=first point
                            scope.write(f"WFSU SP,1,NP,{points_to_read},FP,{samples_read}")
                            time.sleep(0.2)
                            
                            logger.info(f"Reading chunk: FP={samples_read}, NP={points_to_read}")
                            
                            # Read this chunk using query_binary_values for reliable transfer
                            chunk_samples = scope.query_binary_values(
                                "C1:WF? DAT2",
                                datatype='B',
                                is_big_endian=False,
                                container=np.array,
                                header_fmt='ieee'
                            )
                            chunk_samples = np.array(chunk_samples, dtype=np.uint8)
                            
                            if len(chunk_samples) > 0:
                                all_samples.append(chunk_samples)
                                samples_read += len(chunk_samples)
                                logger.info(f"Got {len(chunk_samples)} samples in chunk, total: {samples_read}")
                            else:
                                logger.warning("No data received in chunk")
                                break
                            
                            # Safety check - if we got less than requested, we've reached the end
                            if len(chunk_samples) < points_to_read:
                                logger.info(f"Received less than requested ({len(chunk_samples)} < {points_to_read}), done")
                                break
                        
                        # Combine all chunks
                        if all_samples:
                            samples = np.concatenate(all_samples)
                            logger.info(f"Captured {len(samples)} total samples via chunked read")
                        else:
                            samples = np.array([])
                            
                    except Exception as chunk_err:
                        logger.error(f"Chunked read failed: {chunk_err}")
                        samples = np.array([])
                    finally:
                        scope.timeout = old_timeout
                        scope.chunk_size = old_chunk
                    
                    if samples.size > 0:
                        logger.info(f"Got {samples.size} total samples, raw range: {samples.min()} to {samples.max()}")
                        
                        # Store original sample min/max BEFORE downsampling
                        original_sample_min = samples.min()
                        original_sample_max = samples.max()
                        
                        # Screen time = 10 divisions × TDIV (what you see on scope screen)
                        screen_time = 10.0 * tdiv_seconds
                        
                        # Calculate total time of captured data based on sample rate
                        total_captured_time = samples.size / actual_sample_rate
                        logger.info(f"Captured {samples.size} samples = {total_captured_time*1000:.1f}ms at {actual_sample_rate/1e6:.1f} MSa/s")
                        logger.info(f"Screen time: {screen_time*1000:.1f}ms")
                        
                        # If we captured more than screen time, trim to screen time
                        if total_captured_time > screen_time:
                            # Calculate how many samples correspond to screen time
                            samples_for_screen = int(screen_time * actual_sample_rate)
                            logger.info(f"Trimming {samples.size} samples to {samples_for_screen} for screen time")
                            samples = samples[:samples_for_screen]
                            total_captured_time = screen_time
                        
                        logger.info(f"Using {samples.size} samples for {total_captured_time*1000:.1f}ms")
                        
                        # Downsample for display using min-max decimation to preserve peaks
                        max_display = 10000
                        if samples.size > max_display:
                            # Use min-max decimation: for each segment, keep both min and max
                            # This preserves peak amplitude information
                            n_segments = max_display // 2  # Each segment produces 2 points (min, max)
                            segment_size = samples.size // n_segments
                            
                            display_samples = []
                            for i in range(n_segments):
                                start_idx = i * segment_size
                                end_idx = start_idx + segment_size
                                segment = samples[start_idx:end_idx]
                                if len(segment) > 0:
                                    display_samples.append(segment.min())
                                    display_samples.append(segment.max())
                            
                            display_samples = np.array(display_samples, dtype=np.uint8)
                            logger.info(f"Downsampled to {display_samples.size} points using min-max decimation")
                        else:
                            display_samples = samples
                        
                        # Convert to voltages using Siglent SDS800X HD formula
                        # Sample 128 = offset (center of screen)
                        # Sample 0 = offset - 4*VDIV, Sample 255 = offset + 4*VDIV
                        # 8 divisions total, 256 ADC values
                        volts_per_adc = (8.0 * vdiv_volts) / 256.0
                        voltages = (display_samples.astype(np.float64) - 128.0) * volts_per_adc + voffset_volts
                        
                        # Calculate voltage range from ORIGINAL samples for accurate Y-limits
                        voltage_min = (original_sample_min - 128.0) * volts_per_adc + voffset_volts
                        voltage_max = (original_sample_max - 128.0) * volts_per_adc + voffset_volts
                        
                        # Generate time axis based on ACTUAL captured time (not screen time)
                        # This ensures data is plotted at correct time positions
                        times = np.linspace(0, total_captured_time, len(voltages))
                        
                        # Y-axis limits: offset at bottom (fixed), data above offset shown with margin
                        # Data below offset will be clipped
                        actual_voltage_max = voltages.max()
                        
                        scope_y_min = voffset_volts  # Fixed at offset
                        data_above = actual_voltage_max - voffset_volts
                        scope_y_max = actual_voltage_max + data_above * 0.25  # 25% margin above data max
                        
                        logger.info(f"Y-axis: offset={voffset_volts*1000:.1f}mV, data_max={actual_voltage_max*1000:.1f}mV, y_max={scope_y_max*1000:.1f}mV")
                        
                        # X-axis should match screen time (even if data doesn't fill it all)
                        scope_limits = {
                            'y_min': scope_y_min,
                            'y_max': scope_y_max,
                            'time_start': 0,
                            'time_end': total_captured_time,  # Use actual data time, not screen time
                        }
                        
                        logger.info(f"Displaying {len(voltages)} points")
                        logger.info(f"Data time: {times[0]*1000:.3f}ms to {times[-1]*1000:.3f}ms")
                        logger.info(f"Voltage: {voltages.min()*1000:.2f}mV to {voltages.max()*1000:.2f}mV")
                        logger.info(f"Y-axis limits: {scope_y_min*1000:.1f}mV to {scope_y_max*1000:.1f}mV")
                    else:
                        logger.warning("Oscilloscope returned no samples")
                    
                    # Restore normal trigger mode and return to local control
                    try:
                        scope.write("TRMD NORM")
                        scope.write(":RUN")
                        scope.write(":SYSTem:LOCal")
                    except Exception:
                        pass
                        
                except Exception as e:
                    logger.error(f"Failed to capture waveform: {e}")
            
            # If no real data, generate demo data
            if times is None or voltages is None:
                logger.info("Using demo data for plotting")
                # Generate synthetic resonance response
                num_points = 1000
                times = np.linspace(0, self.sweep_time, num_points)
                # Simulate a resonance peak somewhere in the middle
                center_time = self.sweep_time / 2
                peak_width = self.sweep_time * 0.1
                base_voltage = 0.1 + 0.05 * np.random.random()
                peak_voltage = 2.0 + 0.3 * np.random.random()
                voltages = base_voltage + peak_voltage * np.exp(-((times - center_time) / peak_width) ** 2)
                # Add some noise
                voltages += np.random.normal(0, 0.02, len(voltages))
                # Demo scope limits - auto scale to data
                scope_limits = None
            
            # Validate values
            bad_mask = ~np.isfinite(voltages) | (np.abs(voltages) > 1e6)
            if np.any(bad_mask):
                logger.warning("Detected invalid voltage values; replacing with 0.0")
                voltages = voltages.copy()
                voltages[bad_mask] = 0.0
            
            # Emit the captured data along with scope display limits
            self.sweep_completed.emit(times, voltages, scope_limits)
            
        except Exception as e:
            logger.error(f"Sweep failed: {e}")
            self.sweep_error.emit(str(e))
        finally:
            # Stop function generator output after sweep
            if generator_started and self.funcgen and self.funcgen.is_connected:
                try:
                    self.funcgen.stop_all_outputs()
                    logger.info("Function generator outputs stopped")
                except Exception as e:
                    logger.debug(f"Failed to stop function generator outputs: {e}")


class ResonanceFinderWidget(QWidget):
    """Main resonance finder widget with frequency sweep and plotting capabilities."""
    
    def __init__(self, funcgen: Optional[FunctionGeneratorController] = None,
                 oscilloscope: Optional[OscilloscopeController] = None):
        super().__init__()

        # Controllers (allow injection of shared instances)
        if funcgen is not None:
            self.funcgen = funcgen
        else:
            # Prefer DeviceManager if available
            try:
                from src.controllers.device_manager import DeviceManager
                self.funcgen = DeviceManager.get_instance().get_function_generator()
            except Exception:
                try:
                    self.funcgen = get_function_generator_controller()
                except Exception:
                    self.funcgen = FunctionGeneratorController()

        if oscilloscope is not None:
            self.oscilloscope = oscilloscope
        else:
            try:
                from src.controllers.device_manager import DeviceManager
                self.oscilloscope = DeviceManager.get_instance().get_oscilloscope()
            except Exception:
                try:
                    self.oscilloscope = get_oscilloscope_controller()
                except Exception:
                    self.oscilloscope = OscilloscopeController()
        
        # Data storage
        self.clicked_frequencies = []
        self.point_markers = []
        self.previous_frequency_lists = []
        self.data_loaded = False
        
        # Current sweep data for saving
        self.current_frequencies = None
        self.current_voltages = None
        self.sweep_parameters = {}
        
        # Worker thread
        self.sweep_worker = None
        
        self._setup_ui()
        self._initialize_instruments()
    
    def _initialize_instruments(self):
        """Check hardware connection status and inform user."""
        funcgen_connected = False
        osc_connected = False
        
        try:
            # Check if function generator is already connected (via DeviceManager)
            funcgen_connected = self.funcgen.is_connected if self.funcgen else False
            if not funcgen_connected:
                logger.info("Function generator not connected - will attempt background connection")
                # Try to connect if not already connected
                try:
                    funcgen_connected = self.funcgen.connect(fast_fail=True)
                except Exception as conn_e:
                    logger.debug(f"Function generator connection attempt failed: {conn_e}")
        except Exception as e:
            logger.error(f"Function generator check failed: {e}")
        
        try:
            # Check if oscilloscope is already connected (via DeviceManager)
            osc_connected = self.oscilloscope.is_connected if self.oscilloscope else False
            if not osc_connected:
                logger.info("Oscilloscope not connected - will attempt background connection")
                # Try to connect if not already connected
                try:
                    osc_connected = self.oscilloscope.connect(fast_fail=True)
                except Exception as conn_e:
                    logger.debug(f"Oscilloscope connection attempt failed: {conn_e}")
        except Exception as e:
            logger.error(f"Oscilloscope check failed: {e}")
        
        # Only show error if both devices fail
        if not funcgen_connected and not osc_connected:
            QMessageBox.warning(self, "Hardware Error", 
                              "No hardware devices connected. The interface will work in demo mode.")
        elif not funcgen_connected:
            QMessageBox.information(self, "Hardware Status", 
                                  "Function generator not connected. Some features may be limited.")
        elif not osc_connected:
            QMessageBox.information(self, "Hardware Status", 
                                  "Oscilloscope not connected. Demo data will be used for sweeps.")
    
    def _setup_ui(self):
        """Setup the main user interface."""
        main_layout = QHBoxLayout(self)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Sweep parameters
        sweep_group = QGroupBox("Sweep Parameters")
        sweep_layout = QGridLayout(sweep_group)
        sweep_layout.setHorizontalSpacing(10)
        sweep_layout.setVerticalSpacing(8)
        sweep_layout.setContentsMargins(15, 15, 15, 15)
        
        # Set fixed widths for consistent layout
        spinbox_width = 110
        
        # Amplitude - label LEFT aligned, spinbox with suffix
        amp_label = QLabel("Amplitude:")
        amp_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        sweep_layout.addWidget(amp_label, 0, 0)
        self.sweep_amp_spinbox = QDoubleSpinBox()
        self.sweep_amp_spinbox.setRange(0.1, 10.0)
        self.sweep_amp_spinbox.setSingleStep(0.1)
        self.sweep_amp_spinbox.setDecimals(1)
        self.sweep_amp_spinbox.setValue(4.0)
        self.sweep_amp_spinbox.setSuffix(" Vpp")
        self.sweep_amp_spinbox.setFixedWidth(spinbox_width)
        self.sweep_amp_spinbox.setAlignment(Qt.AlignRight)
        sweep_layout.addWidget(self.sweep_amp_spinbox, 0, 1)
        
        # Start frequency - label LEFT aligned, spinbox with suffix
        start_label = QLabel("Start Freq:")
        start_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        sweep_layout.addWidget(start_label, 1, 0)
        self.freq_start_spinbox = QDoubleSpinBox()
        self.freq_start_spinbox.setRange(1.0, 100.0)
        self.freq_start_spinbox.setSingleStep(0.1)
        self.freq_start_spinbox.setDecimals(1)
        self.freq_start_spinbox.setValue(13.0)
        self.freq_start_spinbox.setSuffix(" MHz")
        self.freq_start_spinbox.setFixedWidth(spinbox_width)
        self.freq_start_spinbox.setAlignment(Qt.AlignRight)
        sweep_layout.addWidget(self.freq_start_spinbox, 1, 1)
        
        # Stop frequency - label LEFT aligned, spinbox with suffix
        stop_label = QLabel("Stop Freq:")
        stop_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        sweep_layout.addWidget(stop_label, 2, 0)
        self.freq_stop_spinbox = QDoubleSpinBox()
        self.freq_stop_spinbox.setRange(1.0, 100.0)
        self.freq_stop_spinbox.setSingleStep(0.1)
        self.freq_stop_spinbox.setDecimals(1)
        self.freq_stop_spinbox.setValue(15.0)
        self.freq_stop_spinbox.setSuffix(" MHz")
        self.freq_stop_spinbox.setFixedWidth(spinbox_width)
        self.freq_stop_spinbox.setAlignment(Qt.AlignRight)
        sweep_layout.addWidget(self.freq_stop_spinbox, 2, 1)
        
        # Sweep time - label LEFT aligned, spinbox with suffix
        time_label = QLabel("Sweep Time:")
        time_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        sweep_layout.addWidget(time_label, 3, 0)
        self.sweep_time_spinbox = QDoubleSpinBox()
        self.sweep_time_spinbox.setRange(0.1, 10.0)
        self.sweep_time_spinbox.setSingleStep(0.1)
        self.sweep_time_spinbox.setDecimals(1)
        self.sweep_time_spinbox.setValue(0.1)
        self.sweep_time_spinbox.setSuffix(" s")
        self.sweep_time_spinbox.setFixedWidth(spinbox_width)
        self.sweep_time_spinbox.setAlignment(Qt.AlignRight)
        self.sweep_time_spinbox.setEnabled(True)
        sweep_layout.addWidget(self.sweep_time_spinbox, 3, 1)
        
        # Add stretch to push spinboxes to right
        sweep_layout.setColumnStretch(1, 1)
        
        left_layout.addWidget(sweep_group)
        
        # Start sweep button
        self.start_button = QPushButton("Start Sweep")
        self.start_button.clicked.connect(self._start_sweep)
        self.start_button.setStyleSheet("QPushButton { padding: 8px; font-size: 12px; }")
        left_layout.addWidget(self.start_button)
        
        # Loading label
        self.loading_label = QLabel("Loading...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("color: gray; font-style: italic;")
        self.loading_label.hide()
        left_layout.addWidget(self.loading_label)
        
        # Selected frequencies display
        freq_group = QGroupBox("Selected Points")
        freq_layout = QVBoxLayout(freq_group)
        
        self.freq_display_label = QLabel("No points selected")
        self.freq_display_label.setWordWrap(True)
        self.freq_display_label.setStyleSheet("padding: 5px;")
        freq_layout.addWidget(self.freq_display_label)
        
        left_layout.addWidget(freq_group)
        left_layout.addStretch()
        
        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        plot_group = QGroupBox("Oscilloscope Capture")
        plot_layout = QVBoxLayout(plot_group)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (V)')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        # Connect click events
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        
        plot_layout.addWidget(self.canvas)
        right_layout.addWidget(plot_group)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Initial plot
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _start_sweep(self):
        """Start the frequency sweep measurement."""
        if self.sweep_worker and self.sweep_worker.isRunning():
            logger.warning("Sweep already in progress")
            return
        
        # Get parameters
        amplitude = self.sweep_amp_spinbox.value()
        freq_start = self.freq_start_spinbox.value()
        freq_stop = self.freq_stop_spinbox.value()
        sweep_time = self.sweep_time_spinbox.value()
        
        # Validate parameters
        if freq_start >= freq_stop:
            QMessageBox.warning(self, "Invalid Parameters", "Start frequency must be less than stop frequency")
            return
        
        # Check if at least one device is connected (allow demo mode)
        if not self.funcgen.is_connected and not self.oscilloscope.is_connected:
            QMessageBox.information(self, "Demo Mode", 
                                  "No hardware connected. Running in demo mode with simulated data.")
        
        # Show loading state
        self.start_button.setEnabled(False)
        self.loading_label.show()
        
        # Start sweep worker (single fixed 5s sweep; sweep_time UI is disabled)
        self.sweep_worker = SweepWorker(
            self.funcgen, self.oscilloscope,
            amplitude, freq_start, freq_stop, sweep_time
        )
        self.sweep_worker.sweep_completed.connect(self._on_sweep_completed)
        self.sweep_worker.sweep_error.connect(self._on_sweep_error)
        self.sweep_worker.start()
    
    def _on_sweep_completed(self, times, voltages, scope_limits):
        """Handle completed frequency sweep and oscilloscope capture.
        
        Args:
            times: Time values from oscilloscope (seconds)
            voltages: Voltage values from oscilloscope (volts)
            scope_limits: Dict with y_min, y_max, screen_time from scope settings, or None for auto
        """
        try:
            # Store sweep parameters for saving
            self.sweep_parameters = {
                'amplitude_vpp': self.sweep_amp_spinbox.value(),
                'start_frequency_mhz': self.freq_start_spinbox.value(),
                'stop_frequency_mhz': self.freq_stop_spinbox.value(),
                'sweep_time_s': self.sweep_time_spinbox.value(),
                'timestamp': datetime.now().isoformat()
            }
            # Plot the captured oscilloscope data with scope limits
            self._plot_sweep_data(times, voltages, scope_limits)
            logger.info(f"Oscilloscope capture completed: {len(times)} points")
        except Exception as e:
            logger.error(f"Plot update failed: {e}")
        finally:
            # Always re-enable the UI controls after the sweep finishes
            try:
                self.start_button.setEnabled(True)
                self.loading_label.hide()
            except Exception:
                pass
    
    def _on_sweep_error(self, error_msg):
        """Handle sweep error."""
        logger.error(f"Sweep error: {error_msg}")
        QMessageBox.critical(self, "Sweep Error", f"Frequency sweep failed: {error_msg}")
        self.start_button.setEnabled(True)
        self.loading_label.hide()
    
    def _plot_sweep_data(self, times, voltages, scope_limits=None):
        """Plot the oscilloscope screen capture data.
        
        Args:
            times: Time values from oscilloscope (seconds)
            voltages: Voltage values from oscilloscope (volts)
            scope_limits: Optional dict with y_min, y_max, time_start, time_end to match scope display
        """
        # Save previous frequencies if any exist
        if self.clicked_frequencies:
            self.previous_frequency_lists.append(self.clicked_frequencies.copy())
        
        # Clear previous data
        self.clicked_frequencies.clear()
        for marker in self.point_markers:
            marker.remove()
        self.point_markers.clear()
        
        # Clear and setup plot
        self.ax.clear()
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (V)')
        
        # Plot data
        times = np.array(times)
        voltages = np.array(voltages)
        
        logger.info(f"Plot function received: voltages min={voltages.min()*1000:.1f}mV, max={voltages.max()*1000:.1f}mV")
        
        self.ax.plot(times, voltages, 'b-', linewidth=1.0)
        
        # X-axis: 0 to 100ms (full screen time)
        self.ax.set_xlim(0, times.max())
        
        # Y-axis: offset at bottom (fixed), auto-scale max to show ALL data
        y_min = scope_limits['y_min'] if scope_limits else voltages.min()  # offset
        y_max = voltages.max()  # actual data max
        y_range = y_max - y_min
        y_max_with_margin = y_max + y_range * 0.15  # 15% margin above data
        
        self.ax.set_ylim(y_min, y_max_with_margin)
        
        logger.info(f"Plot Y-axis: y_min(offset)={y_min*1000:.1f}mV, data_max={y_max*1000:.1f}mV, y_max(axis)={y_max_with_margin*1000:.1f}mV")
        
        # Store data for click detection
        self.current_frequencies = times  # Store times as x-axis data
        self.current_voltages = voltages
        self.data_loaded = True
        
        self.figure.tight_layout()
        self.canvas.draw()
        self._update_frequency_display()
    
    def _on_plot_click(self, event):
        """Handle mouse clicks on the plot."""
        if not self.data_loaded or event.inaxes != self.ax:
            return
        
        click_x = event.xdata
        click_y = event.ydata
        
        if click_x is None or click_y is None:
            return
        
        # Round to reasonable precision
        click_x = round(click_x, 6)
        
        # Calculate tolerance based on data range (1% of visible range)
        if self.current_frequencies is not None and len(self.current_frequencies) > 1:
            data_range = np.max(self.current_frequencies) - np.min(self.current_frequencies)
            tolerance = data_range * 0.01
        else:
            tolerance = 0.00001  # Default for small time values
        
        # Check if clicking near existing point (remove it)
        for i, freq in enumerate(self.clicked_frequencies):
            if abs(freq - click_x) < tolerance:
                self.point_markers[i].remove()
                del self.clicked_frequencies[i]
                del self.point_markers[i]
                self.canvas.draw()
                self._update_frequency_display()
                return
        
        # Add new marker
        self._add_frequency_marker(click_x, click_y)
    
    def _add_frequency_marker(self, x, y):
        """Add a frequency marker to the plot."""
        self.clicked_frequencies.append(x)
        marker, = self.ax.plot(x, y, 'ro', markersize=8, markerfacecolor='none', markeredgewidth=2)
        self.point_markers.append(marker)
        self.canvas.draw()
        self._update_frequency_display()
    
    def _update_frequency_display(self):
        """Update the time/position display label."""
        if not self.clicked_frequencies and not self.previous_frequency_lists:
            self.freq_display_label.setText("No points selected")
            return
        
        display_text = ""
        
        # Current selected points (times)
        if self.clicked_frequencies:
            sorted_current = sorted(self.clicked_frequencies)
            current_text = "Selected points (time in s):\n" + ", ".join(f"{t:.6f}" for t in sorted_current)
            display_text += current_text
        
        # Previous points
        if self.previous_frequency_lists:
            if display_text:
                display_text += "\n\n"
            
            prev_text = "Previous selections:\n"
            for i, prev_list in enumerate(reversed(self.previous_frequency_lists[-3:])):  # Show last 3
                if prev_list:
                    sorted_prev = sorted(prev_list)
                    prev_text += f"• {', '.join(f'{t:.6f}' for t in sorted_prev)}\n"
            display_text += prev_text.rstrip()
        
        if not display_text:
            display_text = "No points selected"
        
        self.freq_display_label.setText(display_text)
    
    def _save_resonance_data_to_hdf5(self):
        """Save resonance data to the current HDF5 recording file."""
        try:
            # Get the main window instance
            app = QApplication.instance()
            main_windows = [widget for widget in app.topLevelWidgets() 
                          if hasattr(widget, 'camera_widget') and widget.camera_widget]
            
            if not main_windows:
                logger.warning("No main window found - cannot save resonance data")
                return False
            
            main_window = main_windows[0]
            camera_widget = main_window.camera_widget
            
            # Check if recording is active
            if not hasattr(camera_widget, 'hdf5_recorder') or not camera_widget.hdf5_recorder:
                logger.warning("No active HDF5 recording - resonance data not saved")
                return False
            
            if not camera_widget.hdf5_recorder.is_recording:
                logger.warning("HDF5 recording not active - resonance data not saved")
                return False
            
            # Prepare resonance data for saving
            resonance_data = {
                'sweep_parameters': self.sweep_parameters,
                'selected_frequencies_mhz': self.clicked_frequencies.copy() if self.clicked_frequencies else [],
                'previous_frequencies_lists': [freq_list.copy() for freq_list in self.previous_frequency_lists] if self.previous_frequency_lists else [],
                'data_loaded': self.data_loaded
            }
            
            # Add sweep data if available
            if self.current_frequencies is not None and self.current_voltages is not None:
                resonance_data['sweep_data'] = {
                    'frequencies_mhz': self.current_frequencies.tolist() if hasattr(self.current_frequencies, 'tolist') else list(self.current_frequencies),
                    'voltages_v': self.current_voltages.tolist() if hasattr(self.current_voltages, 'tolist') else list(self.current_voltages)
                }
            
            # Log to HDF5 file using the execution data method
            success = camera_widget.hdf5_recorder.log_execution_data('resonance_finder_data', resonance_data)
            
            if success:
                logger.info("Resonance finder data saved to HDF5 file successfully")
                logger.info(f"Saved: {len(self.clicked_frequencies)} selected frequencies, "
                          f"{len(self.previous_frequency_lists)} previous frequency lists")
                if self.current_frequencies is not None:
                    logger.info(f"Saved: sweep data with {len(self.current_frequencies)} data points")
                
                # Show a brief notification to the user
                QMessageBox.information(self, "Data Saved", 
                                      f"Resonance data saved to HDF5 file:\n"
                                      f"• {len(self.clicked_frequencies)} selected frequencies\n"
                                      f"• {len(self.previous_frequency_lists)} previous measurements\n"
                                      f"• Sweep data: {len(self.current_frequencies) if self.current_frequencies is not None else 0} points")
                return True
            else:
                logger.error("Failed to save resonance finder data to HDF5 file")
                QMessageBox.warning(self, "Save Failed", "Failed to save resonance data to HDF5 file")
                return False
                
        except Exception as e:
            logger.error(f"Error saving resonance data to HDF5: {e}")
            return False
    
    def closeEvent(self, event):
        """Handle widget close event."""
        try:
            # Save resonance data to HDF5 file if there's any data to save
            if (self.data_loaded or self.clicked_frequencies or 
                self.previous_frequency_lists or self.current_frequencies is not None):
                logger.info("Saving resonance finder data before closing...")
                self._save_resonance_data_to_hdf5()
            
            # Stop any running sweep
            if self.sweep_worker and self.sweep_worker.isRunning():
                self.sweep_worker.quit()
                self.sweep_worker.wait(1000)  # Wait up to 1 second
            
            # Stop function generator output (but don't disconnect - managed by DeviceManager)
            if self.funcgen and self.funcgen.is_connected:
                try:
                    self.funcgen.stop_all_outputs()
                    logger.info("Stopped function generator outputs")
                except Exception as fg_e:
                    logger.warning(f"Error stopping function generator: {fg_e}")
            
            # Reset oscilloscope to normal viewing mode (turn off persistence)
            if self.oscilloscope and self.oscilloscope.is_connected:
                try:
                    self.oscilloscope.reset_to_normal_mode()
                    logger.info("Reset oscilloscope to normal viewing mode")
                except Exception as osc_e:
                    logger.warning(f"Error resetting oscilloscope: {osc_e}")
            
            # Note: Don't disconnect devices as they are managed by DeviceManager
            # and may be used by other parts of the application
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        event.accept()