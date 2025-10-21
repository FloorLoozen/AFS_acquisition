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
        Connect to function generator using the exact working approach.
        Auto-detects any available Siglent device.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            
            # Find any Siglent USB device
            siglent_resource = None
            for resource in resources:
                if 'USB' in resource and 'F4EC' in resource:
                    siglent_resource = resource
                    break
                    
            if not siglent_resource:
                logger.error("Function Generator: No Siglent device found")
                return False
                
            # Connect using exact approach from working script
            self.function_generator = rm.open_resource(siglent_resource)
            
            # Test connection with simple ID query
            fg_id = self.function_generator.query('*IDN?').strip()
            logger.info(f"Function Generator: Connected to {fg_id}")
            
            self._is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Function generator connection failed: {e}")
            self.function_generator = None
            self._is_connected = False
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
            self.function_generator.write(f"{channel_str}:OUTP ON")
            
            self._output_on = True
            self._last_sine = current
            return True
            
        except Exception as e:
            logger.error(f"Function Generator: sine wave failed: {e}")
            return False

    def stop_all_outputs(self) -> bool:
        """Turn off all outputs using the exact working approach."""
        if not self.function_generator:
            logger.error("Function Generator: not connected")
            return False
            
        try:
            logger.info("Function Generator: stopping all outputs")
            
            # Use exact same commands as the working script
            self.function_generator.write("C1:OUTP OFF")
            self.function_generator.write("C2:OUTP OFF")
            
            self._output_on = False
            return True
            
        except Exception as e:
            logger.error(f"Function Generator: stop outputs failed: {e}")
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
                self.stop_all_outputs()
                self.function_generator.close()
            except Exception as e:
                logger.error(f"Function Generator: error during disconnect: {e}")
            finally:
                self.function_generator = None
                self._is_connected = False