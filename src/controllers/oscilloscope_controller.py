"""
Minimal Oscilloscope Controller for the AFS Tracking System.
Simple and efficient oscilloscope communication.
"""

import pyvisa
import numpy as np
import time
from typing import Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger("oscilloscope_controller")


class OscilloscopeController:
    """Minimal oscilloscope controller with essential functionality."""

    def __init__(self, resource_name: str = 'ASRL4::INSTR'):
        self.resource_name = resource_name
        self.oscilloscope = None
        self.is_connected = False

    def connect(self) -> bool:
        """Connect to oscilloscope."""
        try:
            rm = pyvisa.ResourceManager()
            self.oscilloscope = rm.open_resource(self.resource_name)
            self.oscilloscope.timeout = 5000
            self.oscilloscope.read_termination = '\n'
            self.oscilloscope.write_termination = '\n'
            
            osc_id = self.oscilloscope.query('*IDN?').strip()
            self.is_connected = True
            logger.info(f"Oscilloscope connected: {osc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from oscilloscope."""
        if self.oscilloscope:
            try:
                self.oscilloscope.close()
            except:
                pass
            finally:
                self.oscilloscope = None
                self.is_connected = False
                logger.info("Oscilloscope disconnected")

    def configure_acquisition(self) -> bool:
        """Configure oscilloscope for optimal data acquisition."""
        if not self.is_connected or not self.oscilloscope:
            return False
            
        try:
            # Basic acquisition setup
            self.oscilloscope.write('*CLS')  # Clear status
            self.oscilloscope.write('DATA:SOURCE CH1')
            self.oscilloscope.write('DATA:ENC ASCII')
            self.oscilloscope.write('HEADER OFF')
            
            # Set appropriate time and voltage scales
            self.oscilloscope.write('CH1:COUPLING DC')
            self.oscilloscope.write('CH1:SCALE 0.5')  # 0.5V/div
            self.oscilloscope.write('HORIZONTAL:SCALE 100E-6')  # 100μs/div
            
            logger.info("Oscilloscope configured for acquisition")
            return True
            
        except Exception as e:
            logger.error(f"Oscilloscope configuration failed: {e}")
            return False

    def acquire_single_waveform(self) -> Optional[np.ndarray]:
        """Acquire a single waveform."""
        if not self.is_connected or not self.oscilloscope:
            return None
            
        try:
            # Basic configuration
            self.oscilloscope.write('DATA:SOURCE CH1')
            self.oscilloscope.write('DATA:ENC ASCII')
            self.oscilloscope.write('HEADER OFF')
            
            # Get waveform parameters
            yoff = float(self.oscilloscope.query('WFMPRE:YOFF?'))
            ymult = float(self.oscilloscope.query('WFMPRE:YMULT?'))
            
            # Get waveform data
            raw_data = self.oscilloscope.query('CURVE?')
            voltage_data = np.array([float(x) for x in raw_data.split(',')])
            
            # Convert to voltage
            voltage_array = (voltage_data - yoff) * ymult
            
            logger.debug(f"Acquired {len(voltage_array)} data points")
            return voltage_array
            
        except Exception as e:
            logger.warning(f"Waveform acquisition failed: {e}")
            return None

    def acquire_sweep_data(self, duration: float, freq_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acquire data during a frequency sweep.
        
        Args:
            duration: Sweep duration in seconds
            freq_range: Tuple of (start_freq, stop_freq) in MHz
            
        Returns:
            Tuple of (frequencies, voltages) arrays
        """
        if not self.is_connected or not self.oscilloscope:
            logger.error("Oscilloscope not connected")
            return np.array([]), np.array([])
        
        try:
            start_freq, stop_freq = freq_range
            num_points = 100  # Number of data points to collect
            
            # Generate frequency array
            frequencies = np.linspace(start_freq, stop_freq, num_points)
            voltages = []
            
            # Calculate time between measurements
            step_time = duration / num_points
            
            logger.info(f"Starting sweep data acquisition: {start_freq:.2f} - {stop_freq:.2f} MHz, {duration:.1f}s")
            
            start_time = time.time()
            
            for i, freq in enumerate(frequencies):
                # Calculate expected time for this measurement
                expected_time = start_time + i * step_time
                current_time = time.time()
                
                # Wait if we're ahead of schedule
                if current_time < expected_time:
                    time.sleep(expected_time - current_time)
                
                # Acquire waveform
                waveform = self.acquire_single_waveform()
                
                if waveform is not None and len(waveform) > 0:
                    # Calculate RMS voltage
                    voltage_rms = np.sqrt(np.mean(waveform**2))
                    voltages.append(voltage_rms)
                else:
                    # Use previous value or zero if no previous data
                    if voltages:
                        voltages.append(voltages[-1])
                    else:
                        voltages.append(0.0)
            
            voltages = np.array(voltages)
            
            logger.info(f"Sweep data acquisition completed: {len(frequencies)} points")
            return frequencies, voltages
            
        except Exception as e:
            logger.error(f"Sweep data acquisition failed: {e}")
            return np.array([]), np.array([])

    def get_connection_status(self) -> dict:
        """Get current connection status."""
        return {
            'connected': self.is_connected,
            'resource_name': self.resource_name
        }

    def __del__(self):
        """Cleanup on destruction."""
        self.disconnect()


# Singleton instance management
_oscilloscope_instance = None

def get_oscilloscope_controller() -> OscilloscopeController:
    """Get the singleton oscilloscope controller instance."""
    global _oscilloscope_instance
    if _oscilloscope_instance is None:
        _oscilloscope_instance = OscilloscopeController()
    return _oscilloscope_instance