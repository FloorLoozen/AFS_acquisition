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
import struct

from src.utils.logger import get_logger
from src.utils.exceptions import OscilloscopeError, HardwareError
from src.utils.validation import validate_positive_number, validate_range
from src.utils.visa_helper import VISAHelper

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
        self._is_siglent = False  # Track if this is a Siglent oscilloscope
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
            resources = VISAHelper.list_resources()
            logger.info(f"Oscilloscope: Found VISA resources: {resources}")
            
            if not resources:
                logger.info("Oscilloscope: No VISA resources found")
                return False
            
            target_resource = None
            
            if self.resource_name:
                if self.resource_name in resources:
                    target_resource = self.resource_name
            else:
                # Hardcoded oscilloscope address (Siglent SDS804X HD)
                hardcoded_osc = 'USB0::0xF4EC::0x1017::SDS08A0X904388::INSTR'
                if hardcoded_osc in resources:
                    target_resource = hardcoded_osc
                else:
                    # Fallback: Auto-detect Tektronix or Siglent oscilloscope
                    for resource in resources:
                        resource_upper = resource.upper()
                        if any(keyword in resource_upper for keyword in ['TEKTRONIX', 'TEK', 'SDS', '0X1017', 'ASRL']):
                            target_resource = resource
                            break
            
            if not target_resource:
                logger.info("Oscilloscope: No suitable resource found")
                return False
            
            logger.info(f"Oscilloscope: Attempting to connect to {target_resource}")
            
            # Open with reasonable timeout for fast startup (but still generous for serial)
            self.oscilloscope = VISAHelper.open_resource(target_resource)
            if not self.oscilloscope:
                return False
                
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
            
            # Detect if this is a Siglent oscilloscope
            manufacturer = self._device_info.get("manufacturer", "").upper()
            model = self._device_info.get("model", "").upper()
            self._is_siglent = "SIGLENT" in manufacturer or "SDS" in model
            
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
            resources = VISAHelper.list_resources()
            logger.debug(f"Available VISA resources: {resources}")
            
            if not resources:
                raise OscilloscopeError("No VISA resources found")
            
            target_resource = None
            
            if self.resource_name:
                if self.resource_name in resources:
                    target_resource = self.resource_name
                else:
                    raise OscilloscopeError(f"Specified resource '{self.resource_name}' not found")
            else:
                # Hardcoded oscilloscope address (Siglent SDS804X HD)
                hardcoded_osc = 'USB0::0xF4EC::0x1017::SDS08A0X904388::INSTR'
                if hardcoded_osc in resources:
                    target_resource = hardcoded_osc
                else:
                    # Fallback: Auto-detect Tektronix or Siglent oscilloscope with multiple patterns
                    for resource in resources:
                        resource_upper = resource.upper()
                        if any(keyword in resource_upper for keyword in ['TEKTRONIX', 'TEK', 'SDS', '0X1017', 'ASRL']):
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
            self.oscilloscope = VISAHelper.open_resource(target_resource)
            if not self.oscilloscope:
                raise OscilloscopeError(f"Failed to open resource: {target_resource}")
                
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
            
            # Detect if this is a Siglent oscilloscope
            manufacturer = self._device_info.get("manufacturer", "").upper()
            model = self._device_info.get("model", "").upper()
            self._is_siglent = "SIGLENT" in manufacturer or "SDS" in model
            
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

    def configure_acquisition(self, channel: int = 1, trigger_channel: int = 2, use_auto_trigger: bool = True) -> bool:
        """
        Configure oscilloscope for optimal data acquisition.
        Supports both Tektronix and Siglent oscilloscopes.
        
        Args:
            channel: Channel to display and acquire from (default: 1)
            trigger_channel: Channel to use as trigger source (default: 2)
            use_auto_trigger: Use AUTO trigger mode for faster acquisition (default: True)
        
        Returns:
            True if successful
            
        Raises:
            OscilloscopeError: If configuration fails
        """
        if not self.is_connected:
            raise OscilloscopeError("Oscilloscope not connected")
        
        try:
            logger.info(f"Configuring oscilloscope: display CH{channel}, trigger CH{trigger_channel}, auto={use_auto_trigger}")
            
            # Clear any errors
            self._send_command("*CLS")
            
            if self._is_siglent:
                # Siglent-specific configuration
                # Enable both channels (signal channel and trigger channel)
                self._send_command(f"C{channel}:TRA ON")  # Display channel
                self._send_command(f"C{trigger_channel}:TRA ON")  # Trigger channel
                
                # Set vertical scale (volts/div) for signal channel
                self._send_command(f"C{channel}:VDIV 0.5V")
                
                # Set vertical scale for trigger channel (can be different)
                self._send_command(f"C{trigger_channel}:VDIV 1.0V")
                
                # Set coupling to DC for both channels
                self._send_command(f"C{channel}:CPL D1M")  # DC 1MOhm
                self._send_command(f"C{trigger_channel}:CPL D1M")
                
                # Set horizontal time base (time/div)
                self._send_command("TDIV 100US")  # 100 microseconds/div
                
                # Configure trigger
                self._send_command(f"TRSE EDGE,SR,C{trigger_channel},HT,OFF")  # Edge trigger on trigger_channel
                self._send_command(f"C{trigger_channel}:TRLV 0.5V")  # Trigger level
                
                # Set trigger mode
                if use_auto_trigger:
                    self._send_command("TRMD AUTO")  # AUTO mode - much faster, always triggers
                    logger.info("Siglent: AUTO trigger mode enabled for fast acquisition")
                else:
                    self._send_command("TRMD NORM")  # NORMAL mode - waits for trigger
                    logger.info("Siglent: NORMAL trigger mode")
                
                # Enable persistence mode for envelope display
                # Siglent uses PESU (Persistence Setup) command
                self._send_command("PESU ON,INFINITE")  # Turn on persistence with infinite accumulation
                
                # Set acquisition mode to sample (not average)
                self._send_command("ACQW SAMPLING")
                
                # Start/arm the trigger system (ensure scope is running)
                self._send_command("TRIG_MODE SINGLE")  # Prepare for trigger
                self._send_command("ARM")  # Arm the trigger
                
                # Alternative: Set to RUN mode to continuously acquire
                # Some Siglent models need this to start acquisition
                try:
                    self._send_command("TRIG_MODE AUTO")  # Keep acquiring in auto mode
                except Exception as e:
                    logger.debug(f"TRIG_MODE AUTO not supported: {e}")
                
                logger.info(f"Siglent oscilloscope configured: CH{channel} display, CH{trigger_channel} trigger, persistence ON, RUNNING")
            else:
                # Tektronix-specific configuration
                self._send_command("DATA:SOURCE CH1")
                self._send_command("DATA:ENC ASCII")
                self._send_command("HEADER OFF")
                
                # Set appropriate time and voltage scales
                self._send_command(f"CH{channel}:COUPLING DC")
                self._send_command(f"CH{channel}:SCALE 0.5")  # 0.5V/div
                self._send_command(f"CH{trigger_channel}:COUPLING DC")
                self._send_command("HORIZONTAL:SCALE 100E-6")  # 100μs/div
                
                # Configure trigger
                self._send_command(f"TRIGGER:A:EDGE:SOURCE CH{trigger_channel}")
                self._send_command("TRIGGER:A:LEVEL 0.5")
                
                if use_auto_trigger:
                    self._send_command("TRIGGER:A:MODE AUTO")
                else:
                    self._send_command("TRIGGER:A:MODE NORMAL")
                
                # Start acquisition
                self._send_command("ACQUIRE:STATE RUN")
                
                logger.info("Tektronix oscilloscope configured for acquisition")
            
            # Update channel state tracking
            self._channel_states[channel] = True
            self._channel_states[trigger_channel] = True
            
            logger.info("Oscilloscope configured successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to configure oscilloscope: {e}"
            logger.error(error_msg)
            raise OscilloscopeError(error_msg) from e
    
    def clear_persistence(self) -> bool:
        """
        Clear the persistence display (useful before starting a new sweep).
        
        Returns:
            True if successful
        """
        if not self.is_connected:
            logger.warning("Cannot clear persistence - oscilloscope not connected")
            return False
        
        try:
            if self._is_siglent:
                # For Siglent, turn persistence off and back on to clear
                # Siglent uses PESU (Persistence Setup) command
                self._send_command("PESU OFF")  # Turn off persistence (clears display)
                time.sleep(0.1)
                self._send_command("PESU ON,INFINITE")  # Turn back on with infinite accumulation
                logger.info("Siglent persistence display cleared")
            else:
                # Tektronix method
                self._send_command("DISPLAY:PERSISTENCE CLEAR")
                logger.info("Tektronix persistence display cleared")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear persistence: {e}")
            return False
    
    def reset_to_normal_mode(self) -> bool:
        """
        Reset oscilloscope to normal viewing mode (turn off persistence, use NORMAL trigger).
        Call this when finished with special acquisition modes like Resonance Finder.
        
        Returns:
            True if successful
        """
        if not self.is_connected:
            logger.warning("Cannot reset oscilloscope - not connected")
            return False
        
        try:
            if self._is_siglent:
                # Turn off persistence mode
                self._send_command("PESU OFF")
                
                # Set trigger mode back to NORMAL (waits for valid trigger)
                self._send_command("TRMD NORM")
                
                logger.info("Siglent oscilloscope reset to normal mode: persistence OFF, NORMAL trigger")
            else:
                # Tektronix method
                self._send_command("DISPLAY:PERSISTENCE OFF")
                self._send_command("TRIGGER:A:MODE NORMAL")
                logger.info("Tektronix oscilloscope reset to normal mode")
            return True
        except Exception as e:
            logger.warning(f"Failed to reset oscilloscope to normal mode: {e}")
            return False

    def acquire_single_waveform(self, channel: int = 1) -> Optional[np.ndarray]:
        """
        Acquire a single waveform with comprehensive error handling.
        Supports both Tektronix and Siglent oscilloscopes.
        
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
        
        # Use appropriate method based on oscilloscope type
        if self._is_siglent:
            return self._acquire_waveform_siglent(channel)
        else:
            return self._acquire_waveform_tektronix(channel)
    
    def _siglent_read_wavedesc(self, scope: pyvisa.Resource) -> Dict[str, float | int | str]:
        """Decode Siglent WAVEDESC header to extract scaling metadata."""
        header_bytes = scope.query_binary_values(
            ":WAVeform:PREamble?",
            datatype="B",
            container=bytearray,
        )
        header = bytes(header_bytes)
        if len(header) < 200:
            raise OscilloscopeError(f"Unexpected WAVEDESC length: {len(header)} bytes")

        comm_order_big = int.from_bytes(header[34:36], "big", signed=False)
        comm_order_little = int.from_bytes(header[34:36], "little", signed=False)
        if comm_order_big in (0, 1):
            endian = ">" if comm_order_big == 0 else "<"
        elif comm_order_little in (0, 1):
            endian = ">" if comm_order_little == 0 else "<"
        else:
            raise OscilloscopeError("Could not determine WAVEDESC byte order")

        comm_type = struct.unpack_from(endian + "h", header, 32)[0]
        if comm_type not in (0, 1):
            raise OscilloscopeError(f"Unsupported WAVEDESC comm_type: {comm_type}")

        vertical_gain = struct.unpack_from(endian + "f", header, 156)[0]
        vertical_offset = struct.unpack_from(endian + "f", header, 160)[0]
        horiz_interval = struct.unpack_from(endian + "f", header, 176)[0]
        horiz_offset = struct.unpack_from(endian + "d", header, 180)[0]

        return {
            "endian": endian,
            "comm_type": comm_type,
            "vertical_gain": vertical_gain,
            "vertical_offset": vertical_offset,
            "horiz_interval": horiz_interval,
            "horiz_offset": horiz_offset,
        }

    def _siglent_read_waveform_samples(self, scope: pyvisa.Resource, meta: Dict[str, float | int | str]) -> np.ndarray:
        """Retrieve raw waveform samples honoring Siglent metadata."""
        datatype = "b" if meta["comm_type"] == 0 else "h"
        samples = scope.query_binary_values(
            ":WAVeform:DATA?",
            datatype=datatype,
            is_big_endian=(meta["endian"] == ">"),
            container=np.array,
        )
        if datatype == "h":
            return samples.astype(np.int16, copy=False)
        return samples.astype(np.int8, copy=False)

    def _acquire_waveform_siglent(self, channel: int) -> Optional[np.ndarray]:
        """Acquire waveform from Siglent oscilloscope using SCPI waveform transfer."""
        try:
            logger.debug(f"Acquiring waveform data from Siglent channel {channel}")

            with self._lock:
                if not self.oscilloscope:
                    raise OscilloscopeError("Oscilloscope handle unavailable")

                scope = self.oscilloscope
                scope.chunk_size = 262144
                scope.query_delay = max(getattr(scope, "query_delay", 0.0), 0.1)
                scope.clear()
                time.sleep(0.05)

                channel_name = f"C{channel}"
                scope.write(f":WAVeform:SOURce {channel_name}")
                scope.write(":WAVeform:MODE NORMal")
                scope.write(":WAVeform:FORMat BYTE")

                meta = self._siglent_read_wavedesc(scope)
                raw_samples = self._siglent_read_waveform_samples(scope, meta)

            if raw_samples.size == 0:
                raise OscilloscopeError("Siglent returned no waveform samples")

            sample_float = raw_samples.astype(np.float64, copy=False)
            voltages = (sample_float - meta["vertical_offset"]) * meta["vertical_gain"]
            _times = meta["horiz_offset"] + np.arange(raw_samples.size, dtype=np.float64) * meta["horiz_interval"]

            logger.debug(
                "Siglent waveform meta: comm_type=%s gain=%e offset=%e dt=%e",
                meta["comm_type"],
                meta["vertical_gain"],
                meta["vertical_offset"],
                meta["horiz_interval"],
            )

            return voltages

        except Exception as e:
            logger.error(f"Siglent waveform acquisition error: {e}")
            raise OscilloscopeError(str(e)) from e
    
    def _acquire_waveform_tektronix(self, channel: int) -> Optional[np.ndarray]:
        """Acquire waveform from Tektronix oscilloscope."""
        try:
            logger.debug(f"Acquiring waveform from Tektronix channel {channel}")
            
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
            
            logger.debug(f"Successfully acquired {len(voltage_array)} data points from Tektronix")
            return voltage_array
            
        except Exception as e:
            error_msg = f"Tektronix waveform acquisition failed: {e}"
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