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
        Connect to the function generator.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            rm = pyvisa.ResourceManager()
            self.function_generator = rm.open_resource(self.visa_address)
            fg_id = self.function_generator.query('*IDN?').strip()
            self._is_connected = True
            logger.info(f"Function Generator connected ({fg_id})")
            return True
        except Exception as e:
            logger.error(f"Function Generator connection failed: {e}")
            self._is_connected = False
            return False

    def is_connected(self) -> bool:
        """Check if function generator is connected."""
        return self._is_connected and self.function_generator is not None

    def output_sine_wave(self, amplitude: float, frequency_mhz: float, channel: int = 1) -> bool:
        """
        Output sine wave with specified parameters.
        
        Args:
            amplitude: Peak-to-peak voltage in volts
            frequency_mhz: Frequency in MHz
            channel: Output channel (1 or 2)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Function Generator: not connected")
            return False
            
        try:
            # Deduplicate: skip if no change and already on
            current = (round(frequency_mhz, 6), round(amplitude, 6), int(channel))
            if self._output_on and self._last_sine == current:
                return True
                
            logger.info(f"Function Generator: sine ({frequency_mhz:.3f} MHz @ {amplitude:.2f} Vpp)")
            
            channel_str = f"C{channel}"
            frequency_hz = frequency_mhz * 1_000_000
            
            self.function_generator.write(f"{channel_str}:BSWV SHAPE,SINE")
            self.function_generator.write(f"{channel_str}:BSWV FRQ,{frequency_hz}")
            self.function_generator.write(f"{channel_str}:BSWV AMP,{amplitude}")
            self.function_generator.write(f"{channel_str}:OUTP ON")
            
            self._output_on = True
            self._last_sine = current
            return True
            
        except Exception as e:
            logger.error(f"Function Generator: failed to output sine wave: {e}")
            return False

    def sine_frequency_sweep(self, amplitude: float, freq_start: float, freq_end: float, 
                           sweep_time: float, channel: int = 1) -> bool:
        """
        Perform frequency sweep with sine wave.
        
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
            logger.error("Function Generator: not connected")
            return False
            
        try:
            logger.info(f"Function Generator: sweep ({freq_start:.3f}-{freq_end:.3f} MHz, {sweep_time:.1f}s)")
            
            channel_str = f"C{channel}"
            start_freq_hz = freq_start * 1_000_000
            
            self.function_generator.write(f"{channel_str}:BSWV SHAPE,SINE")
            self.function_generator.write(f"{channel_str}:BSWV AMP,{amplitude}")
            self.function_generator.write(f"{channel_str}:BSWV FRQ,{start_freq_hz}")
            self.function_generator.write(f"{channel_str}:SWWV STATE,ON")
            self.function_generator.write(f"{channel_str}:SWWV TIME,{sweep_time}")
            self.function_generator.write(f"{channel_str}:SWWV START,{start_freq_hz}")
            self.function_generator.write(f"{channel_str}:SWWV STOP,{freq_end * 1_000_000}")
            self.function_generator.write(f"{channel_str}:SWWV DIR,UP")
            self.function_generator.write(f"{channel_str}:SWWV SOURCE,TIME")
            self.function_generator.write(f"{channel_str}:SWWV SWMD,LINEAR")
            self.function_generator.write(f"C{channel}:OUTP ON")
            self.function_generator.write(f"{channel_str}:SWWV SWST")
            
            time.sleep(sweep_time + 2)
            
            self.function_generator.write(f"{channel_str}:SWWV STATE,OFF")
            self.function_generator.write(f"C{channel}:OUTP OFF")
            
            # Reset state after sweep
            self._output_on = False
            self._last_sine = None
            return True
            
        except Exception as e:
            logger.error(f"Function Generator: sweep failed: {e}")
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
                
            logger.info("Function Generator: outputs off")
            self._output_on = False
            self._last_sine = None
            return True
            
        except Exception as e:
            logger.error(f"Function Generator: failed to stop outputs: {e}")
            return False

    def get_output_status(self) -> dict:
        """
        Get current output status.
        
        Returns:
            Dictionary with output status information
        """
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
                logger.info("Function Generator: disconnected")


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