"""
Resonance Finder Widget for AFS Acquisition.
PyQt5-based resonance frequency analysis tool with frequency sweep and plotting.
"""

import numpy as np
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

    sweep_completed = pyqtSignal(object, object)  # frequencies, voltages
    sweep_error = pyqtSignal(str)
    
    def __init__(self, funcgen, oscilloscope, amplitude, freq_start, freq_stop, sweep_time):
        super().__init__()
        self.funcgen = funcgen
        self.oscilloscope = oscilloscope
        self.amplitude = amplitude
        self.freq_start = freq_start
        self.freq_stop = freq_stop
        self.sweep_time = sweep_time
        # sweep_time parameter is present for compatibility but a fixed
        # sweep duration of 5.0 s will be used during execution.
    
    def run(self):
        """Execute the frequency sweep in the background thread."""
        try:
            # Do NOT change oscilloscope settings here per user request.
            # We will only query measurements from the scope during sweeps.
            
            # Single-run behavior: start the generator sweep using the user
            # selectable sweep_time (default 0.1 s), but keep the generator
            # output ON for a fixed 5.0 seconds while we sample the
            # oscilloscope. DO NOT modify oscilloscope settings.
            sweep_cmd_duration = float(self.sweep_time)
            keep_on_duration = 5.0
            num_points = 100
            frequencies = np.linspace(self.freq_start, self.freq_stop, num_points)

            # Start the hardware sweep (duration = sweep_cmd_duration)
            generator_started = False
            try:
                if self.funcgen and self.funcgen.is_connected:
                    try:
                        success = self.funcgen.sine_frequency_sweep(
                            amplitude=self.amplitude,
                            freq_start=self.freq_start,
                            freq_end=self.freq_stop,
                            sweep_time=sweep_cmd_duration,
                            channel=1,
                        )
                        if not success:
                            self.sweep_error.emit("Failed to start hardware sweep")
                            return
                        generator_started = True
                    except Exception as e:
                        logger.error(f"Hardware sweep failed to start: {e}")
                        self.sweep_error.emit(f"Hardware sweep error: {e}")
                        return
                else:
                    logger.warning("Function generator not connected - using demo data")

                # Sample the oscilloscope once per second for 5 seconds while
                # the generator is performing short 0.1s sweeps. We'll collect
                # one 'screen' per second (5 samples total) and emit a time
                # series (seconds) and the measured amplitude (pk-pk).
                samples = []
                sample_times = []
                sample_count = 5
                for sec in range(sample_count):
                    # wait one second between captures (first capture immediate)
                    if sec > 0:
                        time.sleep(1.0)

                    if self.oscilloscope and self.oscilloscope.is_connected:
                        try:
                            # Prefer acquiring the full waveform and compute pk-pk
                            wf = self.oscilloscope.acquire_waveform(channel=1, require_trigger=False)
                            if wf is not None:
                                pkpk = float(np.max(wf.voltages) - np.min(wf.voltages))
                                samples.append(pkpk)
                            else:
                                samples.append(0.0)
                        except Exception as e:
                            logger.debug(f"Waveform capture error at second {sec}: {e}")
                            samples.append(0.0)
                    else:
                        # Demo sample
                        center_freq = (self.freq_start + self.freq_stop) / 2
                        peak_width = (self.freq_stop - self.freq_start) * 0.1
                        base_voltage = 0.1 + 0.05 * np.random.random()
                        peak_voltage = 2.0 + 0.3 * np.random.random()
                        resonance = peak_voltage * np.exp(-((center_freq - center_freq) / peak_width) ** 2)
                        samples.append(base_voltage + resonance)

                    sample_times.append(float(sec))

                voltages = np.array(samples)
                times = np.array(sample_times)

                # Validate values: discard non-finite or absurd values
                bad_mask = ~np.isfinite(voltages) | (np.abs(voltages) > 1e6)
                if np.any(bad_mask):
                    logger.warning(
                        "Detected invalid amplitude samples; replacing with 0.0 for plotting"
                    )
                    voltages = voltages.copy()
                    voltages[bad_mask] = 0.0

                logger.info(
                    f"Sweep sampling completed: {len(times)} samples acquired, values={voltages.tolist()}"
                )
                # Emit seconds array and measured amplitudes
                self.sweep_completed.emit(times, voltages)
            finally:
                # Ensure generator output is turned off if we started it
                if generator_started and self.funcgen and self.funcgen.is_connected:
                    try:
                        self.funcgen.stop_all_outputs()
                    except Exception as e:
                        logger.debug(f"Failed to stop function generator outputs: {e}")
                # Try to return oscilloscope to normal/local mode after sweep
                try:
                    if self.oscilloscope and self.oscilloscope.is_connected:
                        try:
                            # Best-effort restore of display/remote state
                            self.oscilloscope.reset_to_normal_mode()
                            # Also attempt to return the instrument to local
                            try:
                                self.oscilloscope._send_command(":SYSTem:LOCal")
                            except Exception:
                                # Some instruments accept SYST:LOC or SYSTem:LOCal
                                try:
                                    self.oscilloscope._send_command("SYSTem:LOCal")
                                except Exception:
                                    logger.debug("Could not send local command to oscilloscope")
                        except Exception as e:
                            logger.debug(f"Failed to reset oscilloscope after sweep: {e}")
                except Exception:
                    pass
            
                # (No additional emissions here; we already emitted the sampled
                # time-series above. Clean up and continue.)
            
        except Exception as e:
            logger.error(f"Sweep failed: {e}")
            self.sweep_error.emit(str(e))
            
        except Exception as e:
            logger.error(f"Sweep failed: {e}")
            self.sweep_error.emit(str(e))


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
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # Sweep parameters
        sweep_group = QGroupBox("Sweep Parameters")
        sweep_layout = QGridLayout(sweep_group)
        
        # Amplitude
        sweep_layout.addWidget(QLabel("Amplitude (Vₚₚ):"), 0, 0)
        self.sweep_amp_spinbox = QDoubleSpinBox()
        self.sweep_amp_spinbox.setRange(0.1, 10.0)
        self.sweep_amp_spinbox.setSingleStep(0.1)
        self.sweep_amp_spinbox.setDecimals(1)
        self.sweep_amp_spinbox.setValue(4.0)
        sweep_layout.addWidget(self.sweep_amp_spinbox, 0, 1)
        
        # Start frequency
        sweep_layout.addWidget(QLabel("Start Frequency (MHz):"), 1, 0)
        self.freq_start_spinbox = QDoubleSpinBox()
        self.freq_start_spinbox.setRange(1.0, 100.0)
        self.freq_start_spinbox.setSingleStep(0.1)
        self.freq_start_spinbox.setDecimals(1)
        self.freq_start_spinbox.setValue(13.0)
        sweep_layout.addWidget(self.freq_start_spinbox, 1, 1)
        
        # Stop frequency
        sweep_layout.addWidget(QLabel("Stop Frequency (MHz):"), 2, 0)
        self.freq_stop_spinbox = QDoubleSpinBox()
        self.freq_stop_spinbox.setRange(1.0, 100.0)
        self.freq_stop_spinbox.setSingleStep(0.1)
        self.freq_stop_spinbox.setDecimals(1)
        self.freq_stop_spinbox.setValue(15.0)
        sweep_layout.addWidget(self.freq_stop_spinbox, 2, 1)
        
        # Sweep time
        sweep_layout.addWidget(QLabel("Sweep Time (s):"), 3, 0)
        self.sweep_time_spinbox = QDoubleSpinBox()
        self.sweep_time_spinbox.setRange(0.1, 10.0)
        self.sweep_time_spinbox.setSingleStep(0.1)
        self.sweep_time_spinbox.setDecimals(1)
        self.sweep_time_spinbox.setValue(0.1)
        # Sweep time (used for generator sweep duration) is user-changeable
        self.sweep_time_spinbox.setEnabled(True)
        sweep_layout.addWidget(self.sweep_time_spinbox, 3, 1)
        
        
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
        freq_group = QGroupBox("Selected Frequencies")
        freq_layout = QVBoxLayout(freq_group)
        
        self.freq_display_label = QLabel("No frequencies selected")
        self.freq_display_label.setWordWrap(True)
        self.freq_display_label.setStyleSheet("padding: 5px;")
        freq_layout.addWidget(self.freq_display_label)
        
        left_layout.addWidget(freq_group)
        left_layout.addStretch()
        
        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        plot_group = QGroupBox("Sweep Response")
        plot_layout = QVBoxLayout(plot_group)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Frequency (MHz)')
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
    
    def _on_sweep_completed(self, frequencies, voltages):
        """Handle completed frequency sweep."""
        try:
            # Store sweep parameters for saving
            self.sweep_parameters = {
                'amplitude_vpp': self.sweep_amp_spinbox.value(),
                'start_frequency_mhz': self.freq_start_spinbox.value(),
                'stop_frequency_mhz': self.freq_stop_spinbox.value(),
                'sweep_time_s': self.sweep_time_spinbox.value(),
                'timestamp': datetime.now().isoformat()
            }
            # Single-run: just plot the returned voltages
            self._plot_sweep_data(frequencies, voltages)
            logger.info(f"Sweep completed: {len(frequencies)} points")
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
    
    def _plot_sweep_data(self, frequencies, voltages):
        """Plot the frequency sweep data."""
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
        # Auto-label axis: if x-values look like small seconds (0..10) use Time
        try:
            if np.nanmax(frequencies) <= 10 and np.nanmin(frequencies) >= 0:
                self.ax.set_xlabel('Time (s)')
            else:
                self.ax.set_xlabel('Frequency (MHz)')
        except Exception:
            self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel('Voltage (V)')
        
        # Plot data
        frequencies = np.array(frequencies)
        voltages = np.array(voltages)
        
        self.ax.plot(frequencies, voltages, 'b-', linewidth=1.0)
        
        # Set reasonable limits
        freq_margin = (frequencies.max() - frequencies.min()) * 0.05
        self.ax.set_xlim(frequencies.min() - freq_margin, frequencies.max() + freq_margin)
        
        volt_margin = (voltages.max() - voltages.min()) * 0.1
        if volt_margin == 0:
            volt_margin = 0.1
        self.ax.set_ylim(voltages.min() - volt_margin, voltages.max() + volt_margin)
        
        # Store data for click detection
        self.current_frequencies = frequencies
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
        click_x = round(click_x, 3)
        
        # Check if clicking near existing point (remove it)
        for i, freq in enumerate(self.clicked_frequencies):
            if abs(freq - click_x) < 0.05:  # 50 kHz tolerance
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
        """Update the frequency display label."""
        if not self.clicked_frequencies and not self.previous_frequency_lists:
            self.freq_display_label.setText("No frequencies selected")
            return
        
        display_text = ""
        
        # Current frequencies
        if self.clicked_frequencies:
            sorted_current = sorted(self.clicked_frequencies)
            current_text = "Selected frequencies (MHz):\n" + ", ".join(f"{f:.3f}" for f in sorted_current)
            display_text += current_text
        
        # Previous frequencies
        if self.previous_frequency_lists:
            if display_text:
                display_text += "\n\n"
            
            prev_text = "Previous frequencies (MHz):\n"
            for i, prev_list in enumerate(reversed(self.previous_frequency_lists[-3:])):  # Show last 3
                if prev_list:
                    sorted_prev = sorted(prev_list)
                    prev_text += f"• {', '.join(f'{f:.3f}' for f in sorted_prev)}\n"
            display_text += prev_text.rstrip()
        
        if not display_text:
            display_text = "No frequencies selected"
        
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