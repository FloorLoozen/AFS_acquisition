"""
Function Generator Controller for AFS Tracking System.

Provides robust, production-ready control of Siglent SDG series function generators
with comprehensive error handling, automatic recovery, and performance optimization.
Designed for scientific instrumentation with reliability and precision as primary goals.
"""

import pyvisa
import time
from typing import Optional, Tuple, Dict, Any
from enum import Enum

from src.utils.logger import get_logger
from src.utils.exceptions import FunctionGeneratorError, HardwareError
from src.utils.validation import validate_positive_number, validate_range

logger = get_logger("function_generator")


class ConnectionState(Enum):
    """Connection state enumeration for clear state tracking."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class FunctionGeneratorController:
    """
    Robust controller for Siglent SDG series function generators.
    
    Features:
    - Automatic device discovery and connection
    - Retry logic with exponential backoff
    - Parameter validation and bounds checking
    - State tracking and recovery
    - Comprehensive error reporting
    - Performance optimization with caching
    """
    
    # Class constants for validation and limits
    MIN_FREQUENCY_MHZ = 0.000001  # 1 µHz
    MAX_FREQUENCY_MHZ = 200.0     # Typical max for SDG series
    MIN_AMPLITUDE_VPP = 0.001     # 1 mV
    MAX_AMPLITUDE_VPP = 20.0      # Typical max for SDG series
    MAX_RETRY_ATTEMPTS = 3
    BASE_RETRY_DELAY = 0.5
    DEFAULT_TIMEOUT = 5000  # milliseconds
    
    def __init__(self, resource_name: Optional[str] = None):
        """
        Initialize function generator controller.
        
        Args:
            resource_name: Specific VISA resource name. If None, auto-detect Siglent device.
        """
        self.resource_name = resource_name
        self.function_generator: Optional[pyvisa.Resource] = None
        self._connection_state = ConnectionState.DISCONNECTED
        self._output_on = False
        self._last_sine: Optional[Tuple[float, float, int]] = None  # (freq_mhz, amplitude, channel)
        self._device_info: Optional[Dict[str, str]] = None
        self._error_count = 0
        self._last_error: Optional[str] = None

    def connect(self) -> bool:
        """
        Connect to function generator with automatic discovery and validation.
        
        Returns:
            bool: True if successfully connected
            
        Raises:
            FunctionGeneratorError: If connection fails
        """
        if self.is_connected:
            logger.info("Function generator already connected")
            return True
        
        self._connection_state = ConnectionState.CONNECTING
        
        try:
            return self._execute_with_retry("Connection", self._connect_impl)
        except FunctionGeneratorError:
            self._connection_state = ConnectionState.ERROR
            raise
    
    def _connect_impl(self) -> bool:
        """Implementation of connection logic."""
        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            logger.debug(f"Available VISA resources: {list(resources)}")
            
            if not resources:
                raise FunctionGeneratorError("No VISA resources found")
            
            target_resource = None
            
            if self.resource_name:
                if self.resource_name in resources:
                    target_resource = self.resource_name
                else:
                    raise FunctionGeneratorError(f"Specified resource '{self.resource_name}' not found")
            else:
                # Auto-detect Siglent device with multiple patterns
                for resource in resources:
                    resource_upper = resource.upper()
                    if any(keyword in resource_upper for keyword in ['SIGLENT', 'SDG', 'F4EC']):
                        target_resource = resource
                        break
                
                if not target_resource:
                    # Try first USB or TCP resource as fallback
                    for resource in resources:
                        if any(protocol in resource.upper() for protocol in ['USB', 'TCPIP']):
                            target_resource = resource
                            break
            
            if not target_resource:
                raise FunctionGeneratorError("No suitable function generator found")
            
            logger.info(f"Attempting to connect to: {target_resource}")
            
            # Open connection with optimal settings
            self.function_generator = rm.open_resource(target_resource)
            self.function_generator.timeout = self.DEFAULT_TIMEOUT
            self.function_generator.read_termination = '\n'
            self.function_generator.write_termination = '\n'
            
            # Brief stabilization delay
            time.sleep(0.2)
            
            # Verify connection with device identification
            idn = self.function_generator.query("*IDN?").strip()
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
            time.sleep(0.1)  # Allow reset to complete
            
            self._connection_state = ConnectionState.CONNECTED
            self._error_count = 0
            self._last_error = None
            self.resource_name = target_resource
            
            logger.info("Function generator connected successfully")
            return True
            
        except Exception as e:
            error_msg = f"Connection failed: {e}"
            logger.error(error_msg)
            if hasattr(self, 'function_generator') and self.function_generator:
                try:
                    self.function_generator.close()
                except:
                    pass
                self.function_generator = None
            # Propagate exception to allow retry logic and callers to handle failure
            raise FunctionGeneratorError(error_msg) from e
    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state
    
    @property
    def is_connected(self) -> bool:
        """Check if function generator is connected and responsive."""
        return (self._connection_state == ConnectionState.CONNECTED and 
                self.function_generator is not None)
    
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
            FunctionGeneratorError: If all retry attempts fail
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
        raise FunctionGeneratorError(error_msg) from last_exception
    
    def _send_command(self, command: str, read_response: bool = False) -> Optional[str]:
        """
        Send SCPI command to function generator with error handling.
        
        Args:
            command: SCPI command string
            read_response: Whether to read and return response
            
        Returns:
            Response string if read_response=True, None otherwise
            
        Raises:
            FunctionGeneratorError: If command fails
        """
        if not self.is_connected:
            raise FunctionGeneratorError("Function generator not connected")
        
        try:
            logger.debug(f"Sending command: {command}")
            
            if read_response:
                response = self.function_generator.query(command).strip()
                logger.debug(f"Received response: {response}")
                return response
            else:
                self.function_generator.write(command)
                # Check for errors after write commands
                self._check_device_errors()
                return None
                
        except Exception as e:
            error_msg = f"Command '{command}' failed: {e}"
            logger.error(error_msg)
            self._connection_state = ConnectionState.ERROR
            raise FunctionGeneratorError(error_msg) from e
    
    def _check_device_errors(self) -> None:
        """Check device for errors and clear error queue."""
        try:
            while True:
                error_response = self.function_generator.query("SYST:ERR?").strip()
                if error_response.startswith("0,"):
                    break  # No more errors
                logger.warning(f"Device error: {error_response}")
        except Exception as e:
            logger.warning(f"Failed to check device errors: {e}")

    def output_sine_wave(self, amplitude: float, frequency_mhz: float, channel: int = 1) -> bool:
        """
        Output sine wave with comprehensive validation and error handling.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            frequency_mhz: Frequency in MHz
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful
            
        Raises:
            FunctionGeneratorError: If operation fails
        """
        # Input validation
        validate_positive_number(amplitude, "amplitude")
        validate_positive_number(frequency_mhz, "frequency")
        validate_range(amplitude, self.MIN_AMPLITUDE_VPP, self.MAX_AMPLITUDE_VPP, "amplitude")
        validate_range(frequency_mhz, self.MIN_FREQUENCY_MHZ, self.MAX_FREQUENCY_MHZ, "frequency")
        
        if channel not in [1, 2]:
            raise ValueError(f"Channel must be 1 or 2, got {channel}")
        
        if not self.is_connected:
            raise FunctionGeneratorError("Function generator not connected")
        
        # Smart caching to avoid redundant operations
        current = (round(frequency_mhz, 6), round(amplitude, 6), int(channel))
        if self._output_on and self._last_sine == current:
            logger.debug("Parameters unchanged, skipping update")
            return True
        
        try:
            logger.info(f"Setting sine wave: {frequency_mhz:.3f} MHz @ {amplitude:.2f} Vpp on channel {channel}")
            
            channel_str = f"C{channel}"
            frequency_hz = frequency_mhz * 1_000_000
            
            # Configure waveform parameters
            self._send_command(f"{channel_str}:BSWV SHAPE,SINE")
            self._send_command(f"{channel_str}:BSWV FRQ,{frequency_hz}")
            self._send_command(f"{channel_str}:BSWV AMP,{amplitude}")
            
            # Only turn output ON if it's not already on
            if not self._output_on:
                self._send_command(f"{channel_str}:OUTP ON")
                self._output_on = True
            
            self._last_sine = current
            logger.info("Sine wave configured successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to output sine wave: {e}"
            logger.error(error_msg)
            raise FunctionGeneratorError(error_msg) from e

    def update_parameters_only(self, amplitude: float, frequency_mhz: float, channel: int = 1) -> bool:
        """
        Update frequency and amplitude with optimized performance for real-time control.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            frequency_mhz: Frequency in MHz
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful
            
        Raises:
            FunctionGeneratorError: If operation fails
        """
        # Input validation
        validate_positive_number(amplitude, "amplitude")
        validate_positive_number(frequency_mhz, "frequency")
        validate_range(amplitude, self.MIN_AMPLITUDE_VPP, self.MAX_AMPLITUDE_VPP, "amplitude")
        validate_range(frequency_mhz, self.MIN_FREQUENCY_MHZ, self.MAX_FREQUENCY_MHZ, "frequency")
        
        if not self.is_connected:
            raise FunctionGeneratorError("Function generator not connected")
        
        # Enhanced caching with tighter tolerance for fast updates
        current = (round(frequency_mhz, 4), round(amplitude, 3), int(channel))
        if self._last_sine == current:
            return True
        
        try:
            # Log only significant changes to reduce overhead
            freq_change = abs(frequency_mhz - (self._last_sine[0] if self._last_sine else 0)) > 0.01
            amp_change = abs(amplitude - (self._last_sine[1] if self._last_sine else 0)) > 0.01
            
            if freq_change or amp_change:
                logger.debug(f"Updating parameters: {frequency_mhz:.3f} MHz @ {amplitude:.2f} Vpp")
            
            # Fast parameter updates with reduced timeout
            channel_str = f"C{channel}"
            frequency_hz = frequency_mhz * 1_000_000
            
            # Temporarily reduce timeout for faster response
            original_timeout = self.function_generator.timeout
            self.function_generator.timeout = 1000  # 1 second for fast updates
            
            try:
                # Send commands without individual error checking for speed
                self.function_generator.write(f"{channel_str}:BSWV FRQ,{frequency_hz}")
                self.function_generator.write(f"{channel_str}:BSWV AMP,{amplitude}")
                
                self._last_sine = current
                return True
                
            finally:
                # Always restore original timeout
                self.function_generator.timeout = original_timeout
            
        except Exception as e:
            error_msg = f"Failed to update parameters: {e}"
            logger.error(error_msg)
            raise FunctionGeneratorError(error_msg) from e

    def stop_all_outputs(self) -> bool:
        """
        Turn off all outputs with comprehensive error handling.
        
        Returns:
            True if successful
            
        Raises:
            FunctionGeneratorError: If operation fails
        """
        if not self.is_connected:
            raise FunctionGeneratorError("Function generator not connected")
        
        try:
            logger.info("Stopping all function generator outputs")
            
            # Set reasonable timeout for shutdown commands
            original_timeout = self.function_generator.timeout
            self.function_generator.timeout = 2000  # 2 seconds for shutdown
            
            try:
                # Turn off both channels
                self._send_command("C1:OUTP OFF")
                self._send_command("C2:OUTP OFF")
                
                self._output_on = False
                logger.info("All outputs turned off successfully")
                return True
                
            finally:
                # Always restore original timeout
                self.function_generator.timeout = original_timeout
            
        except Exception as e:
            # Mark as off anyway since we're shutting down
            self._output_on = False
            error_msg = f"Failed to stop outputs: {e}"
            logger.error(error_msg)
            raise FunctionGeneratorError(error_msg) from e

    def update_parameters_batch(self, amplitude: float, frequency_mhz: float, channel: int = 1) -> bool:
        """
        Ultra-fast parameter update using batch commands for maximum performance.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            frequency_mhz: Frequency in MHz
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful, falls back to regular method on failure
        """
        if not self.is_connected:
            return self.update_parameters_only(amplitude, frequency_mhz, channel)
        
        try:
            # Enhanced caching
            current = (round(frequency_mhz, 4), round(amplitude, 3), int(channel))
            if self._last_sine == current:
                return True
            
            # Batch command approach - send multiple commands in one transaction
            channel_str = f"C{channel}"
            frequency_hz = frequency_mhz * 1_000_000
            
            # Combine commands with semicolon separator (SCPI standard)
            batch_command = f"{channel_str}:BSWV FRQ,{frequency_hz};{channel_str}:BSWV AMP,{amplitude}"
            
            # Set very short timeout for maximum speed
            original_timeout = self.function_generator.timeout
            self.function_generator.timeout = 500  # 0.5 seconds
            
            try:
                self.function_generator.write(batch_command)
                self._last_sine = current
                return True
                
            finally:
                self.function_generator.timeout = original_timeout
                
        except Exception:
            # Silently fail for speed - fallback to regular method
            return self.update_parameters_only(amplitude, frequency_mhz, channel)

    def sine_frequency_sweep(self, amplitude: float, freq_start: float, freq_end: float, 
                           sweep_time: float, channel: int = 1) -> bool:
        """
        Configure and start a frequency sweep with validation.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            freq_start: Start frequency in MHz
            freq_end: End frequency in MHz
            sweep_time: Sweep duration in seconds
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful
            
        Raises:
            FunctionGeneratorError: If operation fails
        """
        # Input validation
        validate_positive_number(amplitude, "amplitude")
        validate_positive_number(freq_start, "start frequency")
        validate_positive_number(freq_end, "end frequency")
        validate_positive_number(sweep_time, "sweep time")
        
        validate_range(amplitude, self.MIN_AMPLITUDE_VPP, self.MAX_AMPLITUDE_VPP, "amplitude")
        validate_range(freq_start, self.MIN_FREQUENCY_MHZ, self.MAX_FREQUENCY_MHZ, "start frequency")
        validate_range(freq_end, self.MIN_FREQUENCY_MHZ, self.MAX_FREQUENCY_MHZ, "end frequency")
        
        if channel not in [1, 2]:
            raise ValueError(f"Channel must be 1 or 2, got {channel}")
        
        if not self.is_connected:
            raise FunctionGeneratorError("Function generator not connected")
        
        try:
            logger.info(f"Starting frequency sweep: {freq_start:.3f} - {freq_end:.3f} MHz over {sweep_time:.1f}s")
            
            channel_str = f"C{channel}"
            freq_start_hz = freq_start * 1_000_000
            freq_end_hz = freq_end * 1_000_000
            
            # Configure basic waveform parameters
            self._send_command(f"{channel_str}:BSWV SHAPE,SINE")
            self._send_command(f"{channel_str}:BSWV AMP,{amplitude}")
            
            # Configure sweep parameters
            self._send_command(f"{channel_str}:SWWV STATE,ON")
            self._send_command(f"{channel_str}:SWWV TIME,{sweep_time}")
            self._send_command(f"{channel_str}:SWWV START,{freq_start_hz}")
            self._send_command(f"{channel_str}:SWWV STOP,{freq_end_hz}")
            self._send_command(f"{channel_str}:SWWV SOURCE,TIME")
            self._send_command(f"{channel_str}:SWWV SWMD,LINEAR")
            
            # Turn on output
            self._send_command(f"{channel_str}:OUTP ON")
            self._output_on = True
            
            # Update cached values
            self._last_sine = (freq_start, amplitude, channel)
            
            logger.info("Frequency sweep started successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start frequency sweep: {e}"
            logger.error(error_msg)
            raise FunctionGeneratorError(error_msg) from e

    def get_output_status(self) -> Dict[str, Any]:
        """
        Get comprehensive output status information.
        
        Returns:
            Dictionary with current status information
        """
        return {
            'connected': self.is_connected,
            'connection_state': self.connection_state.value,
            'output_on': self._output_on,
            'last_sine': self._last_sine,
            'device_info': self.device_info,
            'error_count': self._error_count,
            'last_error': self._last_error
        }

    def disconnect(self) -> None:
        """
        Disconnect from function generator with proper cleanup.
        
        Raises:
            FunctionGeneratorError: If cleanup operations fail
        """
        if not self.function_generator:
            logger.info("Function generator already disconnected")
            return
        
        try:
            logger.info("Beginning function generator disconnect")
            
            # First, try to stop all outputs safely
            try:
                self.stop_all_outputs()
            except Exception as e:
                logger.warning(f"Error stopping outputs during disconnect: {e}")
            
            # Small delay to ensure commands are processed
            time.sleep(0.1)
            
            # Close the VISA connection
            logger.info("Closing VISA connection")
            self.function_generator.close()
            logger.info("VISA connection closed successfully")
            
        except Exception as e:
            error_msg = f"Error during disconnect: {e}"
            logger.error(error_msg)
            # Don't raise exception during cleanup, but log the error
        finally:
            # Always clean up references regardless of errors
            self.function_generator = None
            self._connection_state = ConnectionState.DISCONNECTED
            self._output_on = False
            self._last_sine = None
            self._device_info = None
            logger.info("Function generator disconnect completed")