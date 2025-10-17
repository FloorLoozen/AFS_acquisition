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