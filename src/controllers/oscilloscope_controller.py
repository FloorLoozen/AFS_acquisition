"""
Oscilloscope Controller for the AFS Tracking System.
Handles oscilloscope communication and data acquisition.
"""

import pyvisa
import numpy as np
import time
from typing import Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger("oscilloscope_controller")


class OscilloscopeController:
    """
    Controls oscilloscope communication and data acquisition.
    
    Features:
    - VISA communication with configurable settings
    - Binary and ASCII data acquisition modes
    - Waveform parameter extraction
    - Frequency sweep data collection
    - Robust error handling and logging
    """

    def __init__(self, resource_name: str = 'ASRL4::INSTR', use_binary: bool = True):
        """
        Initialize oscilloscope controller.
        
        Args:
            resource_name: VISA resource identifier for the oscilloscope
            use_binary: Whether to use binary data transfer (faster) or ASCII
        """
        self.resource_name = resource_name
        self.oscilloscope: Optional[pyvisa.Resource] = None
        self.use_binary = use_binary
        self.is_connected = False
        
        # Connection parameters
        self.timeout = 30000  # 30 second timeout
        self.baud_rate = 9600
        self.data_bits = 8
        
        logger.debug(f"Oscilloscope controller initialized for {resource_name}")

    def connect(self) -> bool:
        """
        Establish connection to oscilloscope.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            rm = pyvisa.ResourceManager()
            
            self.oscilloscope = rm.open_resource(self.resource_name)
            self.oscilloscope.timeout = self.timeout
            self.oscilloscope.read_termination = '\n'
            self.oscilloscope.write_termination = '\n'
            self.oscilloscope.baud_rate = self.baud_rate
            self.oscilloscope.data_bits = self.data_bits
            self.oscilloscope.parity = pyvisa.constants.Parity.none
            self.oscilloscope.stop_bits = pyvisa.constants.StopBits.one

            # Query device identification
            osc_id = self.oscilloscope.query('*IDN?').strip()
            self.is_connected = True
            
            logger.info(f"Oscilloscope connected: {osc_id}")
            return True

        except Exception as e:
            logger.error(f"Oscilloscope connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        """Close oscilloscope connection and clean up resources."""
        if self.oscilloscope:
            try:
                self.oscilloscope.close()
                logger.info("Oscilloscope disconnected")
            except Exception as e:
                logger.warning(f"Error during oscilloscope disconnect: {e}")
            finally:
                self.oscilloscope = None
                self.is_connected = False

    def configure_acquisition(self, channel: str = 'CH1', 
                            start_point: int = 1, 
                            stop_point: int = 10000) -> bool:
        """
        Configure oscilloscope for data acquisition.
        
        Args:
            channel: Data source channel (e.g., 'CH1', 'CH2')
            start_point: Starting data point for acquisition
            stop_point: Ending data point for acquisition
            
        Returns:
            bool: True if configuration successful, False otherwise
        """
        if not self.is_connected or not self.oscilloscope:
            logger.error("Oscilloscope not connected")
            return False
            
        try:
            encoding = 'RIBINARY' if self.use_binary else 'ASCII'
            
            commands = [
                '*CLS',                    # Clear status
                'ACQUIRE:STATE STOP',      # Stop acquisition
                'ACQUIRE:STATE RUN',       # Start acquisition
                f'DATA:SOURCE {channel}',  # Set data source
                f'DATA:ENC {encoding}',    # Set encoding format
                'HEADER OFF',              # Disable headers in responses
                f'DATA:START {start_point}',  # Set start point
                f'DATA:STOP {stop_point}'     # Set stop point
            ]

            # Add width command only for binary mode (2 bytes per sample for 16-bit)
            if self.use_binary:
                commands.insert(-2, 'DATA:WIDTH 2')

            # Send all configuration commands
            for cmd in commands:
                self.oscilloscope.write(cmd)
                
            # Allow time for configuration to take effect
            time.sleep(1)
            
            logger.debug(f"Oscilloscope configured: {channel}, {encoding} mode")
            return True

        except Exception as e:
            logger.error(f"Oscilloscope configuration failed: {e}")
            return False

    def get_waveform_parameters(self) -> Tuple[float, float, float]:
        """
        Get waveform scaling parameters from oscilloscope.
        
        Returns:
            Tuple[float, float, float]: (y_multiplier, y_zero, y_offset)
            
        Raises:
            RuntimeError: If oscilloscope not connected or query fails
        """
        if not self.is_connected or not self.oscilloscope:
            raise RuntimeError("Oscilloscope not connected")
            
        try:
            y_mult = float(self.oscilloscope.query('WFMPRE:YMULT?'))
            y_zero = float(self.oscilloscope.query('WFMPRE:YZERO?'))
            y_off = float(self.oscilloscope.query('WFMPRE:YOFF?'))
            
            logger.debug(f"Waveform parameters: mult={y_mult}, zero={y_zero}, offset={y_off}")
            return y_mult, y_zero, y_off
            
        except Exception as e:
            logger.error(f"Failed to get waveform parameters: {e}")
            raise RuntimeError(f"Waveform parameter query failed: {e}")

    def acquire_single_waveform(self) -> Optional[np.ndarray]:
        """
        Acquire a single waveform from the oscilloscope.
        
        Returns:
            Optional[np.ndarray]: Voltage data array, None if acquisition fails
        """
        if not self.is_connected or not self.oscilloscope:
            logger.error("Oscilloscope not connected")
            return None
            
        try:
            # Get scaling parameters
            y_mult, y_zero, y_off = self.get_waveform_parameters()
            
            if self.use_binary:
                # Binary mode - much faster for large datasets
                raw_binary = self.oscilloscope.query_binary_values(
                    'CURVE?',
                    datatype='h',        # signed 16-bit integers
                    is_big_endian=True
                )
                raw_data = np.array(raw_binary)
            else:
                # ASCII mode - slower but more compatible
                raw_data_str = self.oscilloscope.query('CURVE?')
                raw_data = np.array([float(x) for x in raw_data_str.split(',')])
            
            # Convert to voltage using scaling parameters
            voltage = (raw_data - y_off) * y_mult + y_zero
            
            logger.debug(f"Acquired waveform: {len(voltage)} points")
            return voltage
            
        except Exception as e:
            logger.error(f"Waveform acquisition failed: {e}")
            return None

    def acquire_sweep_data(self, duration: float = 1.0,
                          freq_range: Tuple[float, float] = (13, 15)) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Acquire data during a frequency sweep.
        
        Args:
            duration: Acquisition duration in seconds
            freq_range: Frequency range tuple (start_MHz, end_MHz)
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (frequency_array, voltage_array)
            Returns (None, None) if acquisition fails
        """
        if not self.is_connected or not self.oscilloscope:
            logger.error("Oscilloscope not connected")
            return None, None
            
        try:
            # Get scaling parameters once
            y_mult, y_zero, y_off = self.get_waveform_parameters()
            
            all_voltage = []
            start_time = time.time()
            sample_count = 0

            logger.info(f"Starting sweep data acquisition: {duration}s, {freq_range[0]}-{freq_range[1]} MHz")

            while time.time() - start_time < duration:
                try:
                    if self.use_binary:
                        raw_binary = self.oscilloscope.query_binary_values(
                            'CURVE?',
                            datatype='h',
                            is_big_endian=True
                        )
                        raw_data = np.array(raw_binary)
                    else:
                        raw_data_str = self.oscilloscope.query('CURVE?')
                        raw_data = np.array([float(x) for x in raw_data_str.split(',')])

                    # Convert to voltage
                    voltage = (raw_data - y_off) * y_mult + y_zero
                    all_voltage.append(voltage)
                    sample_count += 1
                    
                except Exception as e:
                    logger.warning(f"Sample acquisition error (continuing): {e}")
                    continue

            if not all_voltage:
                logger.error("No data acquired during sweep")
                return None, None

            # Concatenate all voltage samples
            all_voltage = np.concatenate(all_voltage)
            
            # Create frequency array linearly distributed over the range
            frequency_range = np.linspace(freq_range[0], freq_range[1], len(all_voltage))
            
            logger.info(f"Sweep acquisition complete: {sample_count} samples, {len(all_voltage)} total points")
            return frequency_range, all_voltage
            
        except Exception as e:
            logger.error(f"Sweep data acquisition failed: {e}")
            return None, None

    def get_connection_status(self) -> dict:
        """
        Get current connection status and device information.
        
        Returns:
            dict: Status information including connection state and device ID
        """
        status = {
            'connected': self.is_connected,
            'resource_name': self.resource_name,
            'use_binary': self.use_binary,
            'device_id': None
        }
        
        if self.is_connected and self.oscilloscope:
            try:
                status['device_id'] = self.oscilloscope.query('*IDN?').strip()
            except Exception as e:
                logger.warning(f"Failed to query device ID: {e}")
                
        return status

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        if self.is_connected:
            self.disconnect()


# Singleton instance management (similar to other controllers in the project)
_oscilloscope_instance = None

def get_oscilloscope_controller() -> OscilloscopeController:
    """
    Get the singleton oscilloscope controller instance.
    
    Returns:
        OscilloscopeController: The singleton controller instance
    """
    global _oscilloscope_instance
    if _oscilloscope_instance is None:
        _oscilloscope_instance = OscilloscopeController()
    return _oscilloscope_instance


if __name__ == "__main__":
    # Test the oscilloscope controller
    controller = OscilloscopeController()
    
    logger.info("Testing oscilloscope controller...")
    
    if controller.connect():
        logger.info("Connection successful!")
        
        if controller.configure_acquisition():
            logger.info("Configuration successful!")
            
            # Test single waveform acquisition
            waveform = controller.acquire_single_waveform()
            if waveform is not None:
                logger.info(f"Single waveform acquired: {len(waveform)} points")
            
        controller.disconnect()
    else:
        logger.error("Connection failed!")