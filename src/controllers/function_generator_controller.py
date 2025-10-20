"""
Function Generator Controller for AFS Tracking System.
Controls Siglent SDG series function generators via VISA interface.
"""

import pyvisa
import time
from src.utils.logger import get_logger

logger = get_logger("function_generator")


class FunctionGeneratorController:
    """Controller for Siglent SDG series function generators."""
    
    def __init__(self, visa_address: str = 'USB0::0xF4EC::0xEE38::SDG1XCA4161219::INSTR'):
        """
        Initialize function generator controller.
        
        Args:
            visa_address: VISA address of the function generator
        """
        self.visa_address = visa_address
        self.function_generator = None
        
        # Idempotent state tracking for logs/commands
        self._output_on = False
        self._last_sine = None  # tuple (freq_mhz, amplitude, channel)
        self._is_connected = False

    def connect(self) -> bool:
        """
        Connect to function generator efficiently.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            rm = pyvisa.ResourceManager()
            self.function_generator = rm.open_resource(self.visa_address)
            self.function_generator.timeout = 2000  # 2 second timeout
            
            # Test connection and get ID
            fg_id = self.function_generator.query('*IDN?').strip()
            
            # Quick reset and setup
            self.function_generator.write("*RST")
            time.sleep(0.1)
            
            # Enable SYNC and trigger output for external triggering
            self.ensure_sync_enabled(force_redundant=False)
            
            self._is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._is_connected = False
            if hasattr(self, 'function_generator') and self.function_generator:
                try:
                    self.function_generator.close()
                except:
                    pass
                self.function_generator = None
            return False

    def is_connected(self) -> bool:
        """Check if function generator is connected."""
        return self._is_connected and self.function_generator is not None

    def output_sine_wave(self, amplitude: float, frequency_mhz: float, channel: int = 1) -> bool:
        """
        Output sine wave with efficient parameter caching.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            frequency_mhz: Frequency in MHz
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False
            
        try:
            # Smart caching with rounded values for stability
            current = (round(frequency_mhz, 6), round(amplitude, 6), int(channel))
            if self._output_on and self._last_sine == current:
                return True  # No change needed - very efficient
            
            # Determine what needs updating
            needs_freq = not self._last_sine or abs(self._last_sine[0] - current[0]) > 0.001
            needs_amp = not self._last_sine or abs(self._last_sine[1] - current[1]) > 0.01
            needs_channel = not self._last_sine or self._last_sine[2] != channel
            
            # Build minimal command list
            commands = []
            channel_str = f"C{channel}"
            
            if needs_channel:
                commands.append(f"{channel_str}:BSWV SHAPE,SINE")
            if needs_freq:
                commands.append(f"{channel_str}:BSWV FRQ,{frequency_mhz * 1_000_000}")
            if needs_amp:
                commands.append(f"{channel_str}:BSWV AMP,{amplitude}")
            if not self._output_on:
                commands.append(f"{channel_str}:OUTP ON")
            
            # Send commands efficiently
            for cmd in commands:
                self.function_generator.write(cmd)
            
            # Ensure SYNC stays on after any output configuration
            self.ensure_sync_enabled()
            
            # Log only significant changes
            if commands and (needs_freq or needs_amp or not self._output_on):
                logger.info(f"Function Generator: sine {frequency_mhz:.3f} MHz @ {amplitude:.2f} Vpp")
            
            self._output_on = True
            self._last_sine = current
            return True
            
        except Exception as e:
            logger.error(f"Sine wave output failed: {e}")
            return False

    def sine_frequency_sweep(self, amplitude: float, freq_start: float, freq_end: float, 
                           sweep_time: float, channel: int = 1) -> bool:
        """
        Configure and start frequency sweep efficiently.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            freq_start: Start frequency in MHz
            freq_end: End frequency in MHz
            sweep_time: Sweep duration in seconds
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False
            
        try:
            logger.info(f"Sweep: {freq_start:.1f}-{freq_end:.1f} MHz, {sweep_time:.1f}s")
            
            channel_str = f"C{channel}"
            start_freq_hz = freq_start * 1_000_000
            end_freq_hz = freq_end * 1_000_000
            
            # Batch all configuration commands for efficiency
            config_commands = [

                f"{channel_str}:SWWV STATE,OFF",
                f"{channel_str}:BSWV SHAPE,SINE",
                f"{channel_str}:BSWV AMP,{amplitude}",
                f"{channel_str}:BSWV FRQ,{start_freq_hz}",
                f"{channel_str}:BSWV OFST,0",
                f"{channel_str}:OUTP PLRT,NOR",
                f"{channel_str}:OUTP IMPD,HZ",
                "SYNC:OUTP ON",
                "SYNC:PLRT NOR",
                ":TRIGger:OUTPut ON",
                f"{channel_str}:SWWV STATE,ON",
                f"{channel_str}:SWWV TIME,{sweep_time}",
                f"{channel_str}:SWWV START,{start_freq_hz}",
                f"{channel_str}:SWWV STOP,{end_freq_hz}",
                f"{channel_str}:SWWV DIR,UP",
                f"{channel_str}:SWWV SOURCE,MAN",
                f"{channel_str}:SWWV SWMD,LINEAR",
                f"{channel_str}:SWWV DLAY,0"
            ]
            
            # Send all configuration commands
            for cmd in config_commands:
                self.function_generator.write(cmd)
            time.sleep(0.1)  # Minimal stabilization time
            
            # Start sweep
            self.function_generator.write(f"C{channel}:OUTP ON")
            time.sleep(0.1)
            self.function_generator.write(f"{channel_str}:SWWV SWST")
            
            # Ensure SYNC stays on after starting sweep
            self.ensure_sync_enabled(force_redundant=True)
            
            logger.info(f"Sweep started: {freq_start:.1f}-{freq_end:.1f} MHz")
            
            self._output_on = True
            self._last_sine = (freq_start, amplitude, channel)
            return True
            
        except Exception as e:
            logger.error(f"Sweep configuration failed: {e}")
            # Simple cleanup on error
            try:
                self.function_generator.write(f"C{channel}:OUTP OFF")
                self.ensure_sync_enabled(force_redundant=True)
            except:
                pass  # Continue even if cleanup fails
            return False

    def ensure_sync_enabled(self, force_redundant: bool = False) -> bool:
        """
        Ensure SYNC and trigger outputs are enabled for external triggering.
        
        Args:
            force_redundant: If True, applies commands multiple times for critical operations
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False
            
        try:
            commands = [
                "SYNC:OUTP ON",
                "SYNC:PLRT NOR", 
                ":TRIGger:OUTPut ON"
            ]
            
            # Apply commands once or multiple times for critical operations
            iterations = 3 if force_redundant else 1
            for _ in range(iterations):
                for cmd in commands:
                    self.function_generator.write(cmd)
                    
            log_msg = "SYNC/trigger output enabled" + (" (force redundant)" if force_redundant else "")
            logger.info(f"Function Generator: {log_msg}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable SYNC/trigger: {e}")
            return False

    def stop_sweep(self, channel: int = 1) -> bool:
        """
        Stop active frequency sweep but keep SYNC output enabled.
        
        Args:
            channel: Channel to stop sweep on
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False
            
        try:
            channel_str = f"C{channel}"
            self.function_generator.write(f"{channel_str}:SWWV STATE,OFF")
            self.function_generator.write(f"C{channel}:OUTP OFF")
            
            # Ensure SYNC stays ON after stopping sweep
            self.ensure_sync_enabled(force_redundant=True)
            logger.info(f"Function Generator: sweep stopped on channel {channel} (SYNC output kept ON)")
            
            # Reset state
            self._output_on = False
            self._last_sine = None
            return True
            
        except Exception as e:
            logger.error(f"Function Generator: failed to stop sweep: {e}")
            return False

    def stop_all_outputs(self) -> bool:
        """
        Stop all function generator outputs.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False
            
        try:
            # Idempotent: only act/log if outputs were on
            if not self._output_on:
                return True
                
            for channel in [1, 2]:
                self.function_generator.write(f"C{channel}:SWWV STATE,OFF")
                self.function_generator.write(f"C{channel}:OUTP OFF")
                
            # Ensure SYNC stays ON after stopping all outputs
            self.ensure_sync_enabled(force_redundant=True)
            logger.info("Function Generator: all outputs off (sync output kept enabled)")
            self._output_on = False
            self._last_sine = None
            return True
            
        except Exception as e:
            logger.error(f"Function Generator: failed to stop outputs: {e}")
            return False

    def get_output_status(self) -> dict:
        """
        Get current output status and ensure SYNC stays on.
        
        Returns:
            Dictionary with output status information
        """
        # Automatically maintain SYNC whenever status is checked
        self.ensure_sync_enabled()
        
        return {
            'connected': self.is_connected(),
            'output_on': self._output_on,
            'last_sine': self._last_sine,
            'visa_address': self.visa_address
        }

    def disconnect(self) -> None:
        """Disconnect from function generator and clean up resources."""
        if self.function_generator:
            try:
                self.stop_all_outputs()
                self.function_generator.close()
            except Exception as e:
                logger.error(f"Function Generator: error during disconnect: {e}")
            finally:
                self.function_generator = None
                self._is_connected = False


def main():
    """Test function for the function generator controller."""
    controller = FunctionGeneratorController()
    
    try:
        if not controller.connect():
            logger.error("Failed to connect to function generator")
            return
            
        # Test sine wave output
        controller.output_sine_wave(amplitude=4.0, frequency_mhz=14.0, channel=1)
        time.sleep(2)
        
        # Test frequency sweep
        controller.sine_frequency_sweep(
            amplitude=4.0,
            freq_start=13.0,
            freq_end=15.0,
            sweep_time=1.0,
            channel=1,
        )
        
    except KeyboardInterrupt:
        logger.warning("Function Generator: test interrupted")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()