"""
Oscilloscope Controller for AFS Acquisition.

Provides robust, production-ready control of Tektronix oscilloscopes
with comprehensive error handling, automatic recovery, and data validation.
Designed for scientific instrumentation with reliability and precision as primary goals.
"""

import pyvisa
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum
import threading

from src.utils.logger import get_logger
from src.utils.exceptions import OscilloscopeError, HardwareError
from src.utils.validation import validate_positive_number, validate_range

logger = get_logger("oscilloscope_controller")


class ConnectionState(Enum):
    """Connection state enumeration for clear state tracking."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class OscilloscopeController:
    """
    Robust controller for Tektronix oscilloscopes.
    
    Features:
    - Automatic device discovery and connection
    - Retry logic with exponential backoff
    - Parameter validation and bounds checking
    - State tracking and recovery
    - Comprehensive error reporting
    - Waveform acquisition with metadata
    """
    
    # Class constants for validation and limits
    MAX_RETRY_ATTEMPTS = 3
    BASE_RETRY_DELAY = 0.5
    DEFAULT_TIMEOUT = 5000  # milliseconds
    MIN_CHANNEL = 1
    MAX_CHANNEL = 4
    
    def __init__(self, resource_name: Optional[str] = None):
        """
        Initialize oscilloscope controller.
        
        Args:
            resource_name: Specific VISA resource name. If None, auto-detect Tektronix device.
        """
        self.resource_name = resource_name
        self.oscilloscope: Optional[pyvisa.Resource] = None
        self._connection_state = ConnectionState.DISCONNECTED
        self._device_info: Optional[Dict[str, str]] = None
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._channel_states: Dict[int, bool] = {}  # Track channel enable states
        # Thread-safety lock for VISA operations
        self._lock = threading.RLock()

    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state
    
    @property
    def is_connected(self) -> bool:
        """Check if oscilloscope is connected and responsive."""
        return (self._connection_state == ConnectionState.CONNECTED and 
                self.oscilloscope is not None)
    
    @property
    def device_info(self) -> Dict[str, str]:
        """Get cached device information."""
        if self._device_info is None:
            return {"model": "Unknown", "serial": "Unknown", "firmware": "Unknown"}
        return self._device_info.copy()
    
    def _execute_with_retry(self, operation_name: str, operation, *args, **kwargs) -> Any:
        """
        Execute an operation with retry logic and exponential backoff.
        
        Args:
            operation_name: Human-readable operation description
            operation: Function to execute
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Result of successful operation
            
        Raises:
            OscilloscopeError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                if attempt > 0:
                    delay = self.BASE_RETRY_DELAY * (2 ** (attempt - 1))
                    logger.debug(f"Retrying {operation_name}, attempt {attempt + 1}, delay: {delay:.2f}s")
                    time.sleep(delay)
                
                result = operation(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"{operation_name} failed on attempt {attempt + 1}: {e}")
                
                # For certain errors, don't retry
                if isinstance(e, (pyvisa.VisaIOError, pyvisa.InvalidSession)):
                    self._connection_state = ConnectionState.ERROR
                    
                if attempt == self.MAX_RETRY_ATTEMPTS - 1:
                    break
        
        self._error_count += 1
        self._last_error = str(last_exception)
        error_msg = f"{operation_name} failed after {self.MAX_RETRY_ATTEMPTS} attempts: {last_exception}"
        logger.error(error_msg)
        raise OscilloscopeError(error_msg) from last_exception
    
    def _send_command(self, command: str, read_response: bool = False) -> Optional[str]:
        """
        Send SCPI command to oscilloscope with error handling.
        
        Args:
            command: SCPI command string
            read_response: Whether to read and return response
            
        Returns:
            Response string if read_response=True, None otherwise
            
        Raises:
            OscilloscopeError: If command fails
        """
        if not self.is_connected:
            raise OscilloscopeError("Oscilloscope not connected")

        try:
            logger.debug(f"Sending command: {command}")
            with self._lock:
                if read_response:
                    response = self.oscilloscope.query(command).strip()
                    logger.debug(f"Received response: {response}")
                    return response
                else:
                    self.oscilloscope.write(command)
                    # Check for errors after write commands
                    # Note: _check_device_errors expects lock to already be held
                    self._check_device_errors()
                    return None

        except Exception as e:
            error_msg = f"Command '{command}' failed: {e}"
            logger.error(error_msg)
            self._connection_state = ConnectionState.ERROR
            raise OscilloscopeError(error_msg) from e
    
    def _check_device_errors(self) -> None:
        """Check device for errors and clear error queue.
        
        Note: This method expects the caller to already hold self._lock.
        """
        try:
            # Lock is already held by caller (_send_command)
            max_iterations = 10  # Prevent infinite loops
            iterations = 0
            while iterations < max_iterations:
                error_response = self.oscilloscope.query("SYSTEM:ERROR?").strip()
                # Check if no error: response can be "0,..." or "+0,..."
                if error_response.startswith("0,") or error_response.startswith("+0,"):
                    break  # No more errors
                logger.warning(f"Device error: {error_response}")
                iterations += 1
        except Exception as e:
            logger.warning(f"Failed to check device errors: {e}")

    def connect(self, fast_fail: bool = False) -> bool:
        """
        Connect to oscilloscope with automatic discovery and validation.
        
        Args:
            fast_fail: If True, only try once with short timeout (for startup)
        
        Returns:
            bool: True if successfully connected
            
        Raises:
            OscilloscopeError: If connection fails (only when fast_fail=False)
        """
        if self.is_connected:
            logger.info("Oscilloscope already connected")
            return True
        
        self._connection_state = ConnectionState.CONNECTING
        
        try:
            if fast_fail:
                # Single attempt with short timeout for fast startup
                return self._connect_impl_fast()
            else:
                # Normal connection with retries
                return self._execute_with_retry("Connection", self._connect_impl)
        except OscilloscopeError:
            self._connection_state = ConnectionState.ERROR
            if not fast_fail:
                raise
            return False
    
    def _connect_impl_fast(self) -> bool:
        """Fast connection attempt with short timeout (for startup)."""
        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            logger.info(f"Oscilloscope: Found VISA resources: {list(resources)}")
            
            if not resources:
                logger.info("Oscilloscope: No VISA resources found")
                return False
            
            target_resource = None
            
            if self.resource_name:
                if self.resource_name in resources:
                    target_resource = self.resource_name
            else:
                # Auto-detect Tektronix device
                for resource in resources:
                    resource_upper = resource.upper()
                    if any(keyword in resource_upper for keyword in ['TEKTRONIX', 'TEK', 'ASRL']):
                        target_resource = resource
                        break
            
            if not target_resource:
                logger.info("Oscilloscope: No suitable resource found")
                return False
            
            logger.info(f"Oscilloscope: Attempting to connect to {target_resource}")
            
            # Open with reasonable timeout for fast startup (but still generous for serial)
            self.oscilloscope = rm.open_resource(target_resource)
            self.oscilloscope.timeout = 10000  # 10 seconds - serial ports can be slow on startup
            self.oscilloscope.read_termination = '\n'
            self.oscilloscope.write_termination = '\n'
            
            # Configure serial port settings if it's a serial connection (RS-232)
            if 'ASRL' in target_resource:
                # Based on your working script: 9600 baud, 8N1, LF termination
                self.oscilloscope.baud_rate = 9600
                self.oscilloscope.data_bits = 8
                self.oscilloscope.parity = pyvisa.constants.Parity.none
                self.oscilloscope.stop_bits = pyvisa.constants.StopBits.one
                self.oscilloscope.flow_control = pyvisa.constants.VI_ASRL_FLOW_NONE
                logger.info(f"Oscilloscope: Configured RS-232 with 9600 baud, 8N1")
                
                # Clear input buffer before querying
                try:
                    self.oscilloscope.clear()
                except Exception as clear_error:
                    logger.debug(f"Could not clear oscilloscope buffer: {clear_error}")
                
            # Query IDN to verify connection
            idn = self.oscilloscope.query("*IDN?").strip()
            
            # Parse device info
            idn_parts = idn.split(',')
            self._device_info = {
                "manufacturer": idn_parts[0] if len(idn_parts) > 0 else "Unknown",
                "model": idn_parts[1] if len(idn_parts) > 1 else "Unknown",
                "serial": idn_parts[2] if len(idn_parts) > 2 else "Unknown", 
                "firmware": idn_parts[3] if len(idn_parts) > 3 else "Unknown"
            }
            
            # Restore normal timeout after connection
            self.oscilloscope.timeout = self.DEFAULT_TIMEOUT
            
            self._connection_state = ConnectionState.CONNECTED
            self._error_count = 0
            self.resource_name = target_resource
            
            logger.info(f"Oscilloscope connected: {idn_parts[1] if len(idn_parts) > 1 else target_resource}")
            return True
            
        except Exception as e:
            # Log failure reason for debugging
            logger.info(f"Oscilloscope fast connection failed: {type(e).__name__}: {e}")
            if hasattr(self, 'oscilloscope') and self.oscilloscope:
                try:
                    self.oscilloscope.close()
                except Exception as close_error:
                    logger.debug(f"Error closing VISA resource during error recovery: {close_error}")
                self.oscilloscope = None
            return False
    
    def _connect_impl(self) -> bool:
        """Implementation of connection logic."""
        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            logger.debug(f"Available VISA resources: {list(resources)}")
            
            if not resources:
                raise OscilloscopeError("No VISA resources found")
            
            target_resource = None
            
            if self.resource_name:
                if self.resource_name in resources:
                    target_resource = self.resource_name
                else:
                    raise OscilloscopeError(f"Specified resource '{self.resource_name}' not found")
            else:
                # Auto-detect Tektronix device
                for resource in resources:
                    resource_upper = resource.upper()
                    if any(keyword in resource_upper for keyword in ['TEKTRONIX', 'TEK', 'ASRL']):
                        target_resource = resource
                        break
                
                if not target_resource:
                    # Try first serial or TCP resource as fallback
                    for resource in resources:
                        if any(protocol in resource.upper() for protocol in ['ASRL', 'TCPIP']):
                            target_resource = resource
                            break
            
            if not target_resource:
                raise OscilloscopeError("No suitable oscilloscope found")
            
            logger.info(f"Attempting to connect to: {target_resource}")
            
            # Open connection with optimal settings
            self.oscilloscope = rm.open_resource(target_resource)
            self.oscilloscope.timeout = self.DEFAULT_TIMEOUT  # Full timeout for normal connection
            self.oscilloscope.read_termination = '\n'
            self.oscilloscope.write_termination = '\n'
            
            # Configure serial port settings if it's a serial connection
            if 'ASRL' in target_resource:
                # Based on your working script: 9600 baud, 8N1, LF termination
                self.oscilloscope.baud_rate = 9600
                self.oscilloscope.data_bits = 8
                self.oscilloscope.parity = pyvisa.constants.Parity.none
                self.oscilloscope.stop_bits = pyvisa.constants.StopBits.one
                self.oscilloscope.flow_control = pyvisa.constants.VI_ASRL_FLOW_NONE
                logger.info(f"Configured serial port with 9600 baud, 8N1")
                
                # Clear input buffer before querying
                try:
                    self.oscilloscope.clear()
                except Exception as clear_error:
                    logger.debug(f"Could not clear oscilloscope buffer: {clear_error}")
            
            # Brief stabilization delay for serial ports
            if 'ASRL' in target_resource:
                time.sleep(0.3)
                
            # Verify connection with device identification
            idn = self.oscilloscope.query("*IDN?").strip()
            logger.info(f"Connected to: {idn}")
            
            # Parse device information
            idn_parts = idn.split(',')
            self._device_info = {
                "manufacturer": idn_parts[0] if len(idn_parts) > 0 else "Unknown",
                "model": idn_parts[1] if len(idn_parts) > 1 else "Unknown",
                "serial": idn_parts[2] if len(idn_parts) > 2 else "Unknown", 
                "firmware": idn_parts[3] if len(idn_parts) > 3 else "Unknown"
            }
            
            # Clear any existing errors and reset
            self._send_command("*CLS")
            self._send_command("*RST")
            time.sleep(0.5)  # Allow reset to complete
            
            self._connection_state = ConnectionState.CONNECTED
            self._error_count = 0
            self._last_error = None
            self.resource_name = target_resource
            
            logger.info("Oscilloscope connected successfully")
            return True
            
        except Exception as e:
            error_msg = f"Connection failed: {e}"
            logger.error(error_msg)
            if hasattr(self, 'oscilloscope') and self.oscilloscope:
                try:
                    self.oscilloscope.close()
                except Exception as close_error:
                    logger.debug(f"Error closing VISA resource during error recovery: {close_error}")
                self.oscilloscope = None
            raise OscilloscopeError(error_msg) from e

    def disconnect(self) -> None:
        """
        Disconnect from oscilloscope with proper cleanup.
        
        Raises:
            OscilloscopeError: If cleanup operations fail
        """
        if not self.oscilloscope:
            logger.info("Oscilloscope already disconnected")
            return

        try:
            logger.info("Beginning oscilloscope disconnect")

            # Small delay to ensure any pending operations complete
            time.sleep(0.1)

            # Close the VISA connection under lock to prevent concurrent access
            logger.info("Closing VISA connection")
            with self._lock:
                try:
                    self.oscilloscope.close()
                    logger.info("VISA connection closed successfully")
                except Exception as e:
                    logger.debug(f"Error closing oscilloscope during disconnect: {e}")

        except Exception as e:
            error_msg = f"Error during disconnect: {e}"
            logger.error(error_msg)
            # Don't raise exception during cleanup, but log the error
        finally:
            # Always clean up references regardless of errors
            self.oscilloscope = None
            self._connection_state = ConnectionState.DISCONNECTED
            self._device_info = None
            self._channel_states.clear()
            logger.info("Oscilloscope disconnect completed")

    def configure_acquisition(self) -> bool:
        """
        Configure oscilloscope for optimal data acquisition.
        
        Returns:
            True if successful
            
        Raises:
            OscilloscopeError: If configuration fails
        """
        if not self.is_connected:
            raise OscilloscopeError("Oscilloscope not connected")
        
        try:
            logger.info("Configuring oscilloscope for acquisition")
            
            # Clear any errors and set up basic configuration
            self._send_command("*CLS")
            self._send_command("DATA:SOURCE CH1")
            self._send_command("DATA:ENC ASCII")
            self._send_command("HEADER OFF")
            
            # Set appropriate time and voltage scales
            self._send_command("CH1:COUPLING DC")
            self._send_command("CH1:SCALE 0.5")  # 0.5V/div
            self._send_command("HORIZONTAL:SCALE 100E-6")  # 100μs/div
            
            # Update channel state tracking
            self._channel_states[1] = True
            
            logger.info("Oscilloscope configured successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to configure oscilloscope: {e}"
            logger.error(error_msg)
            raise OscilloscopeError(error_msg) from e

    def acquire_single_waveform(self, channel: int = 1) -> Optional[np.ndarray]:
        """
        Acquire a single waveform with comprehensive error handling.
        
        Args:
            channel: Channel number to acquire from (1-4)
            
        Returns:
            Voltage array or None if acquisition fails
            
        Raises:
            OscilloscopeError: If acquisition fails
        """
        # Input validation
        validate_range(channel, self.MIN_CHANNEL, self.MAX_CHANNEL, "channel")
        
        if not self.is_connected:
            raise OscilloscopeError("Oscilloscope not connected")
        
        try:
            logger.debug(f"Acquiring waveform from channel {channel}")
            
            # Configure data source and format
            self._send_command(f"DATA:SOURCE CH{channel}")
            self._send_command("DATA:ENC ASCII")
            self._send_command("HEADER OFF")
            
            # Get waveform parameters with validation
            yoff_str = self._send_command("WFMPRE:YOFF?", read_response=True)
            ymult_str = self._send_command("WFMPRE:YMULT?", read_response=True)
            
            if not yoff_str or not ymult_str:
                raise OscilloscopeError("Failed to retrieve waveform parameters")
            
            try:
                yoff = float(yoff_str)
                ymult = float(ymult_str)
            except ValueError as e:
                raise OscilloscopeError(f"Invalid waveform parameters: yoff={yoff_str}, ymult={ymult_str}") from e
            
            # Get waveform data
            raw_data = self._send_command("CURVE?", read_response=True)
            if not raw_data:
                raise OscilloscopeError("No waveform data received")
            
            # Parse and convert data
            try:
                voltage_data = np.array([float(x) for x in raw_data.split(',')])
            except ValueError as e:
                raise OscilloscopeError(f"Failed to parse waveform data: {e}") from e
            
            if len(voltage_data) == 0:
                raise OscilloscopeError("Empty waveform data received")
            
            # Convert to voltage
            voltage_array = (voltage_data - yoff) * ymult
            
            logger.debug(f"Successfully acquired {len(voltage_array)} data points")
            return voltage_array
            
        except Exception as e:
            error_msg = f"Waveform acquisition failed: {e}"
            logger.error(error_msg)
            raise OscilloscopeError(error_msg) from e

    def acquire_sweep_data(self, duration: float, freq_range: Tuple[float, float], 
                          channel: int = 1, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acquire data during a frequency sweep with comprehensive validation.
        
        Args:
            duration: Sweep duration in seconds
            freq_range: Tuple of (start_freq, stop_freq) in MHz
            channel: Channel number to acquire from (1-4)
            num_points: Number of data points to collect
            
        Returns:
            Tuple of (frequencies, voltages) arrays
            
        Raises:
            OscilloscopeError: If acquisition fails
        """
        # Input validation
        validate_positive_number(duration, "duration")
        validate_range(channel, self.MIN_CHANNEL, self.MAX_CHANNEL, "channel")
        validate_positive_number(num_points, "num_points")
        
        if len(freq_range) != 2:
            raise ValueError("freq_range must be a tuple of (start_freq, stop_freq)")
        
        start_freq, stop_freq = freq_range
        validate_positive_number(start_freq, "start frequency")
        validate_positive_number(stop_freq, "stop frequency")
        
        if start_freq >= stop_freq:
            raise ValueError("Start frequency must be less than stop frequency")
        
        if not self.is_connected:
            raise OscilloscopeError("Oscilloscope not connected")
        
        try:
            # Generate frequency array
            frequencies = np.linspace(start_freq, stop_freq, num_points)
            voltages = []
            
            # Calculate time between measurements
            step_time = duration / num_points
            # Add overall timeout (2x expected duration)
            overall_timeout = duration * 2.0
            overall_start_time = time.time()
            
            logger.info(f"Starting sweep data acquisition: {start_freq:.2f} - {stop_freq:.2f} MHz, "
                       f"{duration:.1f}s, {num_points} points")
            
            start_time = time.time()
            successful_acquisitions = 0
            
            for i, freq in enumerate(frequencies):
                # Check overall timeout
                if time.time() - overall_start_time > overall_timeout:
                    logger.error(f"Sweep acquisition timeout after {overall_timeout:.1f}s")
                    raise OscilloscopeError(f"Sweep acquisition timed out after {overall_timeout:.1f}s")
                
                # Calculate expected time for this measurement
                expected_time = start_time + i * step_time
                current_time = time.time()
                
                # Wait if we're ahead of schedule
                if current_time < expected_time:
                    time.sleep(expected_time - current_time)
                
                try:
                    # Acquire waveform with timeout
                    waveform = self.acquire_single_waveform(channel)
                    
                    if waveform is not None and len(waveform) > 0:
                        # Calculate RMS voltage
                        voltage_rms = np.sqrt(np.mean(waveform**2))
                        voltages.append(voltage_rms)
                        successful_acquisitions += 1
                    else:
                        # Use previous value or zero if no previous data
                        if voltages:
                            voltages.append(voltages[-1])
                        else:
                            voltages.append(0.0)
                        logger.warning(f"Failed to acquire data at {freq:.3f} MHz")
                
                except Exception as e:
                    logger.warning(f"Acquisition error at {freq:.3f} MHz: {e}")
                    # Use previous value or zero if no previous data
                    if voltages:
                        voltages.append(voltages[-1])
                    else:
                        voltages.append(0.0)
                
                # Progress logging
                if (i + 1) % (num_points // 10) == 0:
                    progress = (i + 1) / num_points * 100
                    logger.debug(f"Sweep progress: {progress:.1f}%")
            
            voltages = np.array(voltages)
            
            success_rate = successful_acquisitions / num_points * 100
            logger.info(f"Sweep data acquisition completed: {len(frequencies)} points, "
                       f"{success_rate:.1f}% success rate")
            
            if successful_acquisitions == 0:
                raise OscilloscopeError("No successful data acquisitions during sweep")
            
            return frequencies, voltages
            
        except Exception as e:
            error_msg = f"Sweep data acquisition failed: {e}"
            logger.error(error_msg)
            raise OscilloscopeError(error_msg) from e

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get comprehensive connection status information.
        
        Returns:
            Dictionary with current status information
        """
        return {
            'connected': self.is_connected,
            'connection_state': self.connection_state.value,
            'resource_name': self.resource_name,
            'device_info': self.device_info,
            'channel_states': self._channel_states.copy(),
            'error_count': self._error_count,
            'last_error': self._last_error
        }

    def __del__(self):
        """Cleanup on destruction - suppress all errors during garbage collection."""
        try:
            self.disconnect()
        except Exception:
            # Suppress errors during garbage collection to avoid warnings
            pass


# Singleton instance management
_oscilloscope_instance = None

def get_oscilloscope_controller() -> OscilloscopeController:
    """Get the singleton oscilloscope controller instance."""
    global _oscilloscope_instance
    if _oscilloscope_instance is None:
        _oscilloscope_instance = OscilloscopeController()
    return _oscilloscope_instance