"""
Function Generator Controller for AFS Tracking System.
Minimal, robust implementation matching the working script approach.
"""

import pyvisa
from src.utils.logger import get_logger

logger = get_logger("function_generator")


class FunctionGeneratorController:
    """Minimal controller for Siglent SDG series function generators."""
    
    def __init__(self):
        """Initialize function generator controller."""
        self.function_generator = None
        self._output_on = False
        self._last_sine = None  # tuple (freq_mhz, amplitude, channel)
        self._is_connected = False

    def connect(self) -> bool:
        """
        Connect to function generator with enhanced recovery capabilities and retry logic.
        
        Returns:
            True if connection successful, False otherwise
        """
        import time
        
        # Retry parameters for robust connection
        max_attempts = 3
        base_delay = 0.5  # Start with 500ms delay
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    # Exponential backoff: 0.5s, 1.0s, 2.0s
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info(f"Connection attempt {attempt + 1}/{max_attempts} (waiting {delay:.1f}s)...")
                    time.sleep(delay)
                
                # Clean up any existing connection
                if self.function_generator:
                    try:
                        self.function_generator.close()
                    except:
                        pass
                    self.function_generator = None
                    self._is_connected = False
                
                # Small delay to let device reset
                if attempt > 0:
                    time.sleep(0.2)
                
                # Create fresh resource manager
                rm = pyvisa.ResourceManager()
                resources = rm.list_resources()
                
                # Find Siglent USB device
                siglent_resource = None
                for resource in resources:
                    if 'USB' in resource and 'F4EC' in resource:
                        siglent_resource = resource
                        break
                        
                if not siglent_resource:
                    logger.error("Function Generator: No Siglent device found")
                    if attempt == max_attempts - 1:  # Last attempt
                        return False
                    continue
                
                logger.info(f"Attempting to connect to: {siglent_resource}")
                
                # Open the connection with timeout
                self.function_generator = rm.open_resource(siglent_resource)
                
                # Set optimal communication settings
                self.function_generator.read_termination = '\n'
                self.function_generator.write_termination = '\n'
                self.function_generator.timeout = 5000  # Increased timeout for stability
                
                # Brief delay to allow device to stabilize
                time.sleep(0.2)
                
                # Test basic communication with minimal commands
                try:
                    # Simple ID query - most reliable test
                    fg_id = self.function_generator.query('*IDN?').strip()
                    logger.info(f"Function Generator: Connected to {fg_id}")
                    
                    # Clear any error states
                    self.function_generator.write('*CLS')
                    
                    self._is_connected = True
                    return True
                    
                except pyvisa.errors.VisaIOError as visa_error:
                    # Check for specific timeout error
                    if "VI_ERROR_TMO" in str(visa_error):
                        logger.warning(f"Function Generator: Device timeout on attempt {attempt + 1}")
                        if attempt == max_attempts - 1:  # Last attempt
                            logger.error("Function Generator: Device not responding after all attempts")
                            logger.error("Device may need physical reconnection (unplug/replug USB)")
                    else:
                        logger.error(f"Function Generator: VISA error: {visa_error}")
                    
                    # Close and clean up for retry
                    if self.function_generator:
                        try:
                            self.function_generator.close()
                        except:
                            pass
                        self.function_generator = None
                    
                    if attempt == max_attempts - 1:  # Last attempt
                        raise visa_error
                    continue
                
            except Exception as e:
                logger.error(f"Function generator connection attempt {attempt + 1} failed: {e}")
                
                # Clean up for retry
                if self.function_generator:
                    try:
                        self.function_generator.close()
                    except:
                        pass
                    self.function_generator = None
                    self._is_connected = False
                
                if attempt == max_attempts - 1:  # Last attempt
                    return False
                continue
        
        return False

    def is_connected(self) -> bool:
        """Check if function generator is connected."""
        return self._is_connected and self.function_generator is not None

    def output_sine_wave(self, amplitude: float, frequency_mhz: float, channel: int = 1) -> bool:
        """
        Output sine wave using the exact working approach from the original script.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            frequency_mhz: Frequency in MHz
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.function_generator:
            logger.error("Function Generator: not connected")
            return False
            
        try:
            # Simple caching - avoid identical calls
            current = (round(frequency_mhz, 6), round(amplitude, 6), int(channel))
            if self._output_on and self._last_sine == current:
                return True
                
            logger.info(f"Function Generator: sine ({frequency_mhz:.3f} MHz @ {amplitude:.2f} Vpp)")
            
            # Use exact same commands and order as the working script
            channel_str = f"C{channel}"
            frequency_hz = frequency_mhz * 1_000_000
            
            self.function_generator.write(f"{channel_str}:BSWV SHAPE,SINE")
            self.function_generator.write(f"{channel_str}:BSWV FRQ,{frequency_hz}")
            self.function_generator.write(f"{channel_str}:BSWV AMP,{amplitude}")
            
            # Only turn output ON if it's not already on
            if not self._output_on:
                self.function_generator.write(f"{channel_str}:OUTP ON")
                self._output_on = True
            
            self._last_sine = current
            return True
            
        except Exception as e:
            logger.error(f"Function Generator: sine wave failed: {e}")
            return False

    def update_parameters_only(self, amplitude: float, frequency_mhz: float, channel: int = 1) -> bool:
        """
        Update only frequency and amplitude without touching output state.
        Optimized for fast updates with minimal VISA overhead.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            frequency_mhz: Frequency in MHz
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.function_generator:
            logger.error("Function Generator: not connected")
            return False
            
        try:
            # Enhanced caching with tighter tolerance
            current = (round(frequency_mhz, 4), round(amplitude, 3), int(channel))
            if self._last_sine == current:
                return True
                
            # Only log significant changes to reduce overhead
            freq_change = abs(frequency_mhz - (self._last_sine[0] if self._last_sine else 0)) > 0.01
            amp_change = abs(amplitude - (self._last_sine[1] if self._last_sine else 0)) > 0.01
            
            if freq_change or amp_change:
                logger.info(f"Function Generator: update ({frequency_mhz:.3f} MHz @ {amplitude:.2f} Vpp)")
            
            # Fast parameter updates with reduced timeout
            channel_str = f"C{channel}"
            frequency_hz = frequency_mhz * 1_000_000
            
            # Set shorter timeout for faster response
            original_timeout = self.function_generator.timeout
            self.function_generator.timeout = 1000  # 1 second for fast updates
            
            try:
                # Send commands without individual error checking for speed
                self.function_generator.write(f"{channel_str}:BSWV FRQ,{frequency_hz}")
                self.function_generator.write(f"{channel_str}:BSWV AMP,{amplitude}")
                
                self._last_sine = current
                return True
                
            finally:
                # Restore original timeout
                self.function_generator.timeout = original_timeout
            
        except Exception as e:
            logger.error(f"Function Generator: parameter update failed: {e}")
            return False

    def stop_all_outputs(self) -> bool:
        """Turn off all outputs using the exact working approach."""
        if not self.function_generator:
            logger.error("Function Generator: not connected")
            return False
            
        try:
            logger.info("Function Generator: stopping all outputs")
            
            # Set a reasonable timeout for shutdown commands
            original_timeout = self.function_generator.timeout
            self.function_generator.timeout = 2000  # 2 seconds for shutdown
            
            try:
                # Use exact same commands as the working script
                self.function_generator.write("C1:OUTP OFF")
                self.function_generator.write("C2:OUTP OFF")
                
                self._output_on = False
                logger.info("Function Generator: outputs turned off successfully")
                return True
                
            finally:
                # Restore original timeout
                self.function_generator.timeout = original_timeout
            
        except Exception as e:
            logger.error(f"Function Generator: stop outputs failed: {e}")
            # Mark as off anyway since we're shutting down
            self._output_on = False
            return False

    def update_parameters_batch(self, amplitude: float, frequency_mhz: float, channel: int = 1) -> bool:
        """
        Ultra-fast parameter update using batch commands.
        Combines multiple commands into a single VISA transaction.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            frequency_mhz: Frequency in MHz
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.function_generator:
            return False
            
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
        Configure and start a frequency sweep.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            freq_start: Start frequency in MHz
            freq_end: End frequency in MHz
            sweep_time: Sweep duration in seconds
            channel: Output channel (1 or 2)
            
        Returns:
            True if sweep started successfully, False otherwise
        """
        if not self.function_generator:
            logger.error("Function Generator: not connected")
            return False
            
        try:
            logger.info(f"Function Generator: starting frequency sweep {freq_start:.3f} - {freq_end:.3f} MHz over {sweep_time:.1f}s")
            
            channel_str = f"C{channel}"
            freq_start_hz = freq_start * 1_000_000
            freq_end_hz = freq_end * 1_000_000
            
            # Configure basic waveform parameters
            self.function_generator.write(f"{channel_str}:BSWV SHAPE,SINE")
            self.function_generator.write(f"{channel_str}:BSWV AMP,{amplitude}")
            
            # Configure sweep parameters
            self.function_generator.write(f"{channel_str}:SWWV STATE,ON")
            self.function_generator.write(f"{channel_str}:SWWV TIME,{sweep_time}")
            self.function_generator.write(f"{channel_str}:SWWV START,{freq_start_hz}")
            self.function_generator.write(f"{channel_str}:SWWV STOP,{freq_end_hz}")
            self.function_generator.write(f"{channel_str}:SWWV SOURCE,TIME")
            self.function_generator.write(f"{channel_str}:SWWV SWMD,LINEAR")
            
            # Turn on output
            self.function_generator.write(f"{channel_str}:OUTP ON")
            self._output_on = True
            
            # Update cached values
            self._last_sine = (freq_start, amplitude, channel)
            
            logger.info("Function Generator: frequency sweep started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Function Generator: frequency sweep failed: {e}")
            return False

    def get_output_status(self) -> dict:
        """Get current output status."""        
        return {
            'connected': self.is_connected(),
            'output_on': self._output_on,
            'last_sine': self._last_sine
        }

    def disconnect(self) -> None:
        """Disconnect from function generator and clean up resources."""
        if self.function_generator:
            try:
                logger.info("Function Generator: Beginning clean disconnect process")
                
                # First, stop all outputs safely
                try:
                    self.stop_all_outputs()
                except Exception as e:
                    logger.warning(f"Function Generator: stop outputs error during disconnect: {e}")
                
                # Add a small delay to ensure commands are processed
                import time
                time.sleep(0.1)
                
                # Close the VISA connection
                logger.info("Function Generator: Closing VISA connection")
                self.function_generator.close()
                logger.info("Function Generator: VISA connection closed successfully")
                
            except Exception as e:
                logger.error(f"Function Generator: error during disconnect: {e}")
            finally:
                # Always clean up the reference regardless of errors
                self.function_generator = None
                self._is_connected = False
                self._output_on = False
                self._last_sine = None
                logger.info("Function Generator: Disconnect completed - all references cleared")