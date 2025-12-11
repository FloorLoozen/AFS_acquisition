"""
Resonance Finder Widget for AFS Acquisition.
PyQt5-based resonance frequency analysis tool with frequency sweep and plotting.
"""

import numpy as np
import re
import threading
import time
import h5py
import io
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QDoubleSpinBox,
    QFrame, QSizePolicy, QMessageBox, QApplication, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.utils.logger import get_logger
from src.utils.status_display import StatusDisplay
from src.controllers.function_generator_controller import FunctionGeneratorController, get_function_generator_controller
from src.controllers.oscilloscope_controller import OscilloscopeController, get_oscilloscope_controller

logger = get_logger("resonance_finder")


class SweepWorker(QThread):
    """Worker thread for performing frequency sweeps."""

    # Signal: times, voltages, scope_limits dict with keys: y_min, y_max, screen_time
    sweep_completed = pyqtSignal(object, object, object)
    sweep_error = pyqtSignal(str)
    status_update = pyqtSignal(str)  # New signal for status updates
    
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
            
            # Emit status update: now retrieving data
            self.status_update.emit("Retrieving Data")
            
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
                    
                    # Query memory depth to get ALL available samples (not just screen)
                    try:
                        mdepth_response = scope.query("ACQuire:MDEPth?").strip()
                        # Parse memory depth (e.g., "ACQUIRE_MDEPTH 10M" or just "10000000")
                        import re
                        match = re.search(r'([0-9.]+)\s*([KMG])?', mdepth_response, re.IGNORECASE)
                        if match:
                            value = float(match.group(1))
                            unit = match.group(2).upper() if match.group(2) else ''
                            multiplier = {'K': 1e3, 'M': 1e6, 'G': 1e9}.get(unit, 1)
                            total_memory_samples = int(value * multiplier)
                        else:
                            total_memory_samples = 10000000  # Default 10M
                        logger.info(f"Memory depth: {total_memory_samples} samples")
                    except Exception as mdepth_e:
                        logger.warning(f"Could not query memory depth: {mdepth_e}, using 10M default")
                        total_memory_samples = 10000000
                    
                    # Calculate screen time for reference
                    screen_time = 10.0 * tdiv_seconds  # 10 divisions
                    screen_samples = int(screen_time * actual_sample_rate)
                    logger.info(f"Screen: {screen_time*1000:.1f}ms = {screen_samples} samples at {actual_sample_rate/1e6:.1f} MSa/s")
                    logger.info(f"Will capture ALL memory: {total_memory_samples} samples")
                    
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
                        # Read ALL memory data in 5M chunks using native Siglent WFSU command
                        samples_read = 0
                        while samples_read < total_memory_samples:
                            remaining = total_memory_samples - samples_read
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
                        
                        # Check for ADC clipping (signal hitting the rails)
                        clipped_high = np.sum(samples == 255)
                        clipped_low = np.sum(samples == 0)
                        if clipped_high > 0 or clipped_low > 0:
                            logger.warning(f"ADC CLIPPING DETECTED: {clipped_high} samples at max (255), {clipped_low} samples at min (0)")
                            logger.warning(f"Signal is being CLIPPED by oscilloscope! Adjust VDIV or OFFSET!")
                        
                        # Store original sample min/max BEFORE downsampling
                        original_sample_min = samples.min()
                        original_sample_max = samples.max()
                        
                        # Screen time = 10 divisions × TDIV (what you see on scope screen)
                        screen_time = 10.0 * tdiv_seconds
                        
                        # Calculate total time of captured data based on sample rate
                        total_captured_time = samples.size / actual_sample_rate
                        logger.info(f"Captured {samples.size} samples = {total_captured_time*1000:.1f}ms at {actual_sample_rate/1e6:.1f} MSa/s")
                        logger.info(f"Screen time: {screen_time*1000:.1f}ms (reference only - showing ALL memory)")
                        logger.info(f"Using ALL {samples.size} samples for {total_captured_time*1000:.1f}ms")
                        
                        # Filter out samples below offset BEFORE downsampling to avoid distortion
                        # ADC value 128 corresponds to the offset
                        above_offset_mask = samples >= 128
                        samples_above = samples[above_offset_mask]
                        indices_above = np.where(above_offset_mask)[0]
                        logger.info(f"Filtered to {samples_above.size} samples above offset (from {samples.size} total)")
                        
                        # Use the filtered samples for downsampling
                        samples_to_process = samples_above
                        indices_to_process = indices_above
                        
                        # Downsample for display using min-max decimation to preserve peaks
                        max_display = 10000
                        if samples_to_process.size > max_display:
                            # Use min-max decimation: for each segment, keep both min and max
                            # This preserves peak amplitude information
                            n_segments = max_display // 2  # Each segment produces 2 points (min, max)
                            segment_size = samples_to_process.size // n_segments
                            
                            display_samples = []
                            display_indices = []
                            for i in range(n_segments):
                                start_idx = i * segment_size
                                end_idx = start_idx + segment_size
                                segment = samples_to_process[start_idx:end_idx]
                                segment_orig_indices = indices_to_process[start_idx:end_idx]
                                if len(segment) > 0:
                                    min_pos = segment.argmin()
                                    max_pos = segment.argmax()
                                    display_samples.append(segment[min_pos])
                                    display_samples.append(segment[max_pos])
                                    display_indices.append(segment_orig_indices[min_pos])
                                    display_indices.append(segment_orig_indices[max_pos])
                            
                            display_samples = np.array(display_samples, dtype=np.uint8)
                            display_indices = np.array(display_indices, dtype=np.int64)
                            logger.info(f"Downsampled to {display_samples.size} points using min-max decimation")
                        else:
                            display_samples = samples_to_process
                            display_indices = indices_to_process
                        
                        # Convert to voltages using Siglent SDS800X HD formula
                        # Sample 128 = offset (center of screen)
                        # Sample 0 = offset - 4*VDIV, Sample 255 = offset + 4*VDIV
                        # 8 divisions total, 256 ADC values
                        volts_per_adc = (8.0 * vdiv_volts) / 256.0
                        voltages = (display_samples.astype(np.float64) - 128.0) * volts_per_adc + voffset_volts
                        
                        # Calculate voltage range from ORIGINAL samples for accurate Y-limits
                        voltage_min = (original_sample_min - 128.0) * volts_per_adc + voffset_volts
                        voltage_max = (original_sample_max - 128.0) * volts_per_adc + voffset_volts
                        
                        # Generate time axis using original indices so points map to their true times
                        times = display_indices.astype(np.float64) / actual_sample_rate
                        
                        # Y-axis limits: offset at bottom (fixed), data above offset shown with margin
                        # Data below offset will be clipped
                        actual_voltage_max = voltages.max()
                        
                        scope_y_min = voffset_volts  # Fixed at offset
                        data_above = actual_voltage_max - voffset_volts
                        scope_y_max = actual_voltage_max + data_above * 0.10  # 10% margin above data max
                        
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
        self.current_times = None
        self.sweep_parameters = {}
        
        # Worker thread
        self.sweep_worker = None
        
        self._setup_ui()
        self._initialize_instruments()
    
    def _initialize_instruments(self):
        """Check hardware connection status and inform user."""
        # Set initial status
        self.status_display.set_status("Initializing")
        
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
        
        # Update status display based on connection
        if funcgen_connected and osc_connected:
            self.status_display.set_status("Ready")
        elif funcgen_connected or osc_connected:
            self.status_display.set_status("Partially Connected")
        else:
            self.status_display.set_status("Disconnected")
        
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
        
        # Status display at the top
        self.status_display = StatusDisplay()
        self.status_display.setContentsMargins(0, 0, 0, 10)  # Add 10px bottom margin
        left_layout.addWidget(self.status_display)
        
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
        start_label = QLabel("Start Frequency:")
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
        stop_label = QLabel("Stop Frequency:")
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
        self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel('Voltage (V)')
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(-2, 0)
        
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
        
        # Show sweeping state
        self.start_button.setEnabled(False)
        self.status_display.set_status("Sweeping")
        
        # Store sweep parameters for frequency axis conversion
        self.start_freq = freq_start
        self.stop_freq = freq_stop
        self.sweep_time = sweep_time
        
        # Log sweep start to audit trail
        self._log_sweep_to_audit_trail('sweep_started', 
                                      f'Resonance sweep started: {freq_start:.2f}-{freq_stop:.2f} MHz, {amplitude:.2f} Vpp',
                                      {'start_frequency_mhz': freq_start, 'stop_frequency_mhz': freq_stop, 
                                       'amplitude_vpp': amplitude, 'sweep_time_s': sweep_time})
        
        # Start sweep worker (single fixed 5s sweep; sweep_time UI is disabled)
        self.sweep_worker = SweepWorker(
            self.funcgen, self.oscilloscope,
            amplitude, freq_start, freq_stop, sweep_time
        )
        self.sweep_worker.sweep_completed.connect(self._on_sweep_completed)
        self.sweep_worker.sweep_error.connect(self._on_sweep_error)
        self.sweep_worker.status_update.connect(self._on_status_update)
        self.sweep_worker.start()
    
    def _on_status_update(self, status: str):
        """Handle status updates from sweep worker."""
        self.status_display.set_status(status)
    
    def _on_sweep_completed(self, times, voltages, scope_limits):
        """Handle completed frequency sweep and oscilloscope capture.
        
        Args:
            times: Time values from oscilloscope (seconds)
            voltages: Voltage values from oscilloscope (volts)
            scope_limits: Dict with y_min, y_max, screen_time from scope settings, or None for auto
        """
        try:
            # Save previous sweep data to HDF5 if exists (before starting new sweep)
            if self.current_times is not None and self.current_voltages is not None:
                logger.info("Saving previous sweep data before starting new sweep...")
                self._save_sweep_to_hdf5(self.current_times, self.current_voltages)
            
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
            
            # Log sweep completion to audit trail
            self._log_sweep_to_audit_trail('sweep_completed',
                                          f'Resonance sweep completed: {len(times)} data points captured',
                                          {'data_points': len(times), 'sweep_parameters': self.sweep_parameters})
            
            # Store the raw data for later saving (when user closes widget after selecting peaks)
            self.current_times = times
            self.current_voltages = voltages
        except Exception as e:
            logger.error(f"Plot update failed: {e}")
        finally:
            # Always re-enable the UI controls after the sweep finishes
            try:
                self.start_button.setEnabled(True)
                self.status_display.set_status("Ready")
            except Exception:
                pass
    
    def _on_sweep_error(self, error_msg):
        """Handle sweep error."""
        logger.error(f"Sweep error: {error_msg}")
        QMessageBox.critical(self, "Sweep Error", f"Frequency sweep failed: {error_msg}")
        self.start_button.setEnabled(True)
        self.status_display.set_status("Error")
    
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
        self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel('Voltage (V)')
        
        # Plot data
        times = np.array(times)
        voltages = np.array(voltages)
        
        # Convert time axis to frequency axis
        # Linear frequency sweep: freq = start_freq + (stop_freq - start_freq) * (time / sweep_time)
        frequencies = self.start_freq + (self.stop_freq - self.start_freq) * (times / self.sweep_time)
        
        logger.info(f"Plot function received: voltages min={voltages.min()*1000:.1f}mV, max={voltages.max()*1000:.1f}mV")
        
        self.ax.plot(frequencies, voltages, 'b-', linewidth=1.0)
        
        # X-axis: frequency range
        self.ax.set_xlim(self.start_freq, self.stop_freq)
        
        # Y-axis: use scope_limits calculated by worker thread if available
        if scope_limits:
            # Use pre-calculated limits from oscilloscope settings
            y_min = scope_limits['y_min']
            y_max = scope_limits['y_max']
            logger.info(f"Plot Y-axis: Using scope limits: {y_min*1000:.1f}mV to {y_max*1000:.1f}mV")
        else:
            # Fallback: auto-scale from data with margins
            offset = voltages.min()
            data_max = voltages.max()
            data_range = data_max - offset
            y_min = offset
            y_max = data_max + data_range * 0.15
            logger.info(f"Plot Y-axis: Auto-scaled: offset={y_min*1000:.1f}mV, data_max={data_max*1000:.1f}mV, y_max={y_max*1000:.1f}mV")
        
        self.ax.set_ylim(y_min, y_max)
        
        # Store data for click detection
        self.current_frequencies = frequencies  # Store frequencies as x-axis data
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
        
        # Round to reasonable precision (3 decimal places for MHz)
        click_x = round(click_x, 3)
        
        # Calculate tolerance based on data range (1% of visible range)
        if self.current_frequencies is not None and len(self.current_frequencies) > 1:
            data_range = np.max(self.current_frequencies) - np.min(self.current_frequencies)
            tolerance = data_range * 0.01
        else:
            tolerance = 0.01  # Default tolerance in MHz
        
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
        
        # Log frequency selection to audit trail
        self._log_sweep_to_audit_trail('resonance_frequency_selected',
                                      f'Potential resonance frequency selected: {x:.3f} MHz',
                                      {'frequency_mhz': x, 'total_selected': len(self.clicked_frequencies)})
    
    def _update_frequency_display(self):
        """Update the frequency display label."""
        if not self.clicked_frequencies and not self.previous_frequency_lists:
            self.freq_display_label.setText("No points selected")
            return
        
        # Current selected points (frequencies in MHz) - just list them
        if self.clicked_frequencies:
            sorted_current = sorted(self.clicked_frequencies)
            display_text = ", ".join(f"{f:.3f} MHz" for f in sorted_current)
        else:
            display_text = "No points selected"
        
        self.freq_display_label.setText(display_text)
    
    def _log_sweep_to_audit_trail(self, event_type: str, description: str, metadata: dict):
        """Log resonance sweep event to HDF5 audit trail."""
        try:
            app = QApplication.instance()
            main_windows = [widget for widget in app.topLevelWidgets() 
                          if hasattr(widget, 'camera_widget') and widget.camera_widget]
            
            if main_windows:
                main_window = main_windows[0]
                camera_widget = main_window.camera_widget
                
                if hasattr(camera_widget, 'hdf5_recorder') and camera_widget.hdf5_recorder:
                    recorder = camera_widget.hdf5_recorder
                    if recorder.is_recording or hasattr(recorder, 'audit_trail'):
                        recorder.log_hardware_event(event_type, description, metadata)
                        logger.debug(f"Resonance sweep event logged to audit trail: {event_type}")
        except Exception as e:
            logger.debug(f"Could not log sweep event to audit trail: {e}")
    
    def _save_sweep_to_hdf5(self, times, voltages):
        """Save sweep data and plot to HDF5 file using session file.
        
        Uses session HDF5 file from main window.
        
        Args:
            times: Time values from oscilloscope
            voltages: Voltage values from oscilloscope
        """
        try:
            # Get session HDF5 file from main window
            hdf5_file_path = None
            app = QApplication.instance()
            main_windows = [widget for widget in app.topLevelWidgets() 
                          if hasattr(widget, 'get_session_hdf5_file')]
            
            if main_windows:
                main_window = main_windows[0]
                hdf5_file_path = main_window.get_session_hdf5_file()
                if hdf5_file_path:
                    logger.info(f"Using session HDF5 file for resonance: {hdf5_file_path}")
                    
                    # Check if resonance already exists in session - ask user what to do
                    if main_window.session_has_resonance:
                        msg_box = QMessageBox(self)
                        msg_box.setIcon(QMessageBox.Question)
                        msg_box.setWindowTitle("Resonance Already Exists")
                        msg_box.setText("The current session already contains resonance data.")
                        msg_box.setInformativeText("What would you like to do?")
                        
                        overwrite_btn = msg_box.addButton("Overwrite", QMessageBox.AcceptRole)
                        new_session_btn = msg_box.addButton("New Session", QMessageBox.DestructiveRole)
                        cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
                        msg_box.setDefaultButton(cancel_btn)
                        
                        msg_box.exec_()
                        clicked = msg_box.clickedButton()
                        
                        if clicked == cancel_btn:
                            logger.info("Resonance save cancelled by user")
                            return
                        elif clicked == new_session_btn:
                            # Ask main window to create new session
                            if hasattr(main_window, '_new_session_file'):
                                main_window._new_session_file()
                                # Get the new session file
                                hdf5_file_path = main_window.get_session_hdf5_file()
                                logger.info(f"Using new session file: {hdf5_file_path}")
                        # else: overwrite_btn clicked = Overwrite (continue with current session)
                    
                    # Mark that session has resonance data
                    main_window.mark_session_has_resonance()
                
                # Get file path from frequency settings widget
                if hasattr(main_window, 'frequency_settings_widget') and main_window.frequency_settings_widget:
                    try:
                        save_path = main_window.frequency_settings_widget.get_save_path()
                        filename = main_window.frequency_settings_widget.get_filename()
                        if save_path and filename:
                            import os
                            hdf5_file_path = os.path.join(save_path, filename)
                            if not hdf5_file_path.lower().endswith('.hdf5'):
                                hdf5_file_path += '.hdf5'
                            logger.info(f"Using HDF5 file from measurement settings: {hdf5_file_path}")
                    except Exception as e:
                        logger.debug(f"Could not get file from measurement settings: {e}")
                
                # Fallback: check if recording is active
                if not hdf5_file_path and hasattr(main_window, 'camera_widget') and main_window.camera_widget:
                    if hasattr(main_window.camera_widget, 'hdf5_recorder') and main_window.camera_widget.hdf5_recorder:
                        if hasattr(main_window.camera_widget.hdf5_recorder, 'file_path'):
                            hdf5_file_path = str(main_window.camera_widget.hdf5_recorder.file_path)
                            logger.info(f"Using active recording HDF5 file: {hdf5_file_path}")
            
            # If no file from settings or recording, try config
            if not hdf5_file_path:
                try:
                    from src.utils.config_manager import ConfigManager
                    config = ConfigManager()
                    if config.files.last_hdf5_file:
                        hdf5_file_path = config.files.last_hdf5_file
                        logger.info(f"Using last HDF5 file from config: {hdf5_file_path}")
                except Exception as e:
                    logger.debug(f"Could not get last HDF5 file from config: {e}")
            
            # If still no file, create a new one
            if not hdf5_file_path:
                # datetime and Path already imported at module level
                # Use default save path from config
                from src.utils.config_manager import get_config
                cfg = get_config()
                default_dir = Path(cfg.files.default_save_path)
                default_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                hdf5_file_path = str(default_dir / f"resonance_data_{timestamp}.hdf5")
                logger.info(f"Creating new HDF5 file: {hdf5_file_path}")
            
            # Convert times to frequencies using sweep parameters
            start_freq = self.sweep_parameters.get('start_frequency_mhz', self.freq_start_spinbox.value())
            stop_freq = self.sweep_parameters.get('stop_frequency_mhz', self.freq_stop_spinbox.value())
            sweep_time = self.sweep_parameters.get('sweep_time_s', self.sweep_time_spinbox.value())
            frequencies = start_freq + (stop_freq - start_freq) * (times / sweep_time)
            
            # Log current state of clicked frequencies
            logger.info(f"Saving sweep with {len(self.clicked_frequencies)} clicked frequencies: {self.clicked_frequencies}")
            
            # Save plot as PNG in memory buffer (without manual click markers)
            # Temporarily hide all click markers
            hidden_markers = []
            for marker in self.point_markers:
                marker.set_visible(False)
                hidden_markers.append(marker)
            
            buf = io.BytesIO()
            self.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plot_png_bytes = buf.read()  # Keep as bytes, not numpy array
            buf.close()
            
            # Restore click markers visibility
            for marker in hidden_markers:
                marker.set_visible(True)
            self.canvas.draw()  # Redraw to show markers again
            
            # Save to HDF5 file directly
            with h5py.File(hdf5_file_path, 'a') as hf:
                # Create meta_data group if it doesn't exist
                if 'meta_data' not in hf:
                    meta_group = hf.create_group('meta_data')
                    meta_group.attrs['description'] = b'Metadata and analysis results'
                else:
                    meta_group = hf['meta_data']
                
                # Create or get frequency_sweep group
                if 'frequency_sweep' in meta_group:
                    del meta_group['frequency_sweep']
                
                sweep_group = meta_group.create_group('frequency_sweep')
                sweep_group.attrs['description'] = b'Frequency sweep measurement data'
                sweep_group.attrs['amplitude_vpp'] = self.sweep_parameters.get('amplitude_vpp', 0)
                sweep_group.attrs['start_frequency_mhz'] = self.sweep_parameters.get('start_frequency_mhz', 0)
                sweep_group.attrs['stop_frequency_mhz'] = self.sweep_parameters.get('stop_frequency_mhz', 0)
                sweep_group.attrs['sweep_time_s'] = self.sweep_parameters.get('sweep_time_s', 0)
                sweep_group.attrs['timestamp'] = self.sweep_parameters.get('timestamp', '')
                
                # Create compound dataset for sweep data (time, frequency, voltage)
                sweep_dtype = np.dtype([
                    ('time_s', np.float64),
                    ('frequency_mhz', np.float64),
                    ('voltage_mv', np.float64)
                ])
                sweep_data = np.empty(len(times), dtype=sweep_dtype)
                sweep_data['time_s'] = times
                sweep_data['frequency_mhz'] = frequencies
                sweep_data['voltage_mv'] = voltages * 1000  # Convert V to mV
                
                # Save data table inside frequency_sweep group
                data_dataset = sweep_group.create_dataset('data', data=sweep_data, compression='gzip')
                data_dataset.attrs['description'] = b'Sweep measurement data: time, frequency, and voltage'
                data_dataset.attrs['columns'] = b'time_s, frequency_mhz, voltage_mv'
                
                # Save plot image inside frequency_sweep group
                plot_dataset = sweep_group.create_dataset('plot', data=np.void(plot_png_bytes))
                plot_dataset.attrs['description'] = b'Frequency sweep plot (PNG image)'
                plot_dataset.attrs['format'] = b'png'
                plot_dataset.attrs['dpi'] = 150
                plot_dataset.attrs['note'] = b'Binary PNG data - save to .png file to view'
                
                # Save potential resonance frequencies only if any were selected
                if self.clicked_frequencies:
                    # Find corresponding voltages for each clicked frequency using interpolation
                    clicked_voltages = []
                    for freq in self.clicked_frequencies:
                        # Find closest frequency in the data
                        idx = np.argmin(np.abs(frequencies - freq))
                        clicked_voltages.append(voltages[idx] * 1000)  # Convert to mV
                    
                    freq_dtype = np.dtype([
                        ('frequency_mhz', np.float64),
                        ('voltage_mv', np.float64)
                    ])
                    freq_data = np.empty(len(self.clicked_frequencies), dtype=freq_dtype)
                    freq_data['frequency_mhz'] = self.clicked_frequencies
                    freq_data['voltage_mv'] = clicked_voltages
                    
                    potential_freq_dataset = sweep_group.create_dataset('potential_frequencies', data=freq_data, compression='gzip')
                    potential_freq_dataset.attrs['description'] = b'Manually selected potential resonance frequencies with corresponding voltages'
                    potential_freq_dataset.attrs['columns'] = b'frequency_mhz, voltage_mv'
                    potential_freq_dataset.attrs['count'] = len(self.clicked_frequencies)
                    logger.info(f"  Saved {len(self.clicked_frequencies)} potential frequencies with voltages")
                else:
                    logger.info(f"  No potential frequencies selected - table not created")
                
                logger.info(f"Saved resonance sweep to HDF5: /meta_data/frequency_sweep/ in {hdf5_file_path}")
                logger.info(f"  Data points: {len(frequencies)}, Selected frequencies: {len(self.clicked_frequencies)}")
            
            # Update config with this file path
            try:
                from src.utils.config_manager import ConfigManager
                config = ConfigManager()
                config.files.last_hdf5_file = hdf5_file_path
                config.save_config()
            except Exception as e:
                logger.debug(f"Could not save HDF5 file to config: {e}")
                
        except Exception as e:
            logger.error(f"Error saving sweep to HDF5: {e}")
            QMessageBox.warning(self, "Save Error", f"Failed to save resonance data:\n{e}")
    
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
            # Save sweep data to HDF5 file if there's a completed sweep
            if self.current_times is not None and self.current_voltages is not None:
                logger.info("Saving resonance sweep data before closing...")
                self._save_sweep_to_hdf5(self.current_times, self.current_voltages)
            
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