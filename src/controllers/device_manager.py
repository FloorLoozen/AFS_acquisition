"""Device Manager - Centralized hardware controller management."""

import threading
from typing import Optional, Dict, Any

from src.utils.logger import get_logger

logger = get_logger("device_manager")


class DeviceManager:
    """Centralized hardware device manager with health monitoring."""
    
    _instance: Optional['DeviceManager'] = None
    _lock = threading.Lock()

    def __init__(self):
        from src.controllers.function_generator_controller import get_function_generator_controller
        from src.controllers.oscilloscope_controller import get_oscilloscope_controller
        
        self._fg = get_function_generator_controller()
        self._osc = get_oscilloscope_controller()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()

    @classmethod
    def get_instance(cls) -> 'DeviceManager':
        """Get or create the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_function_generator(self):
        """Get the function generator controller."""
        return self._fg

    def get_oscilloscope(self):
        """Get the oscilloscope controller."""
        return self._osc

    def connect_all(self, fast_fail: bool = True) -> Dict[str, Any]:
        """Attempt to connect to all managed devices."""
        results = {}
        
        # Function generator
        try:
            if not self._fg.is_connected:
                self._fg.connect(fast_fail=fast_fail)
            results['function_generator'] = {'connected': self._fg.is_connected}
        except Exception as e:
            results['function_generator'] = {'connected': False, 'error': str(e)}

        # Oscilloscope
        try:
            if not self._osc.is_connected:
                self._osc.connect(fast_fail=fast_fail)
            results['oscilloscope'] = {'connected': self._osc.is_connected}
        except Exception as e:
            results['oscilloscope'] = {'connected': False, 'error': str(e)}

        return results

    def disconnect_all(self):
        """Disconnect all managed devices."""
        try:
            if self._fg.is_connected:
                self._fg.disconnect()
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.error(f"Error disconnecting function generator: {e}")
        except Exception as e:
            logger.error(f"Unexpected error disconnecting function generator: {e}")
        
        try:
            if self._osc.is_connected:
                self._osc.disconnect()
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.error(f"Error disconnecting oscilloscope: {e}")
        except Exception as e:
            logger.error(f"Unexpected error disconnecting oscilloscope: {e}")

    def start_health_monitor(self, interval: float = 5.0):
        """Start background health monitor thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_health_monitor(self):
        """Stop the health monitor thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_stop.set()
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Health monitor loop that reconnects disconnected devices."""
        while not self._monitor_stop.is_set():
            try:
                if not self._fg.is_connected:
                    try:
                        self._fg.connect(fast_fail=True)
                    except (OSError, ConnectionError, TimeoutError) as conn_err:
                        # Fast-fail reconnects expected to fail frequently during connection recovery
                        logger.debug(f"FG reconnect attempt in progress: {type(conn_err).__name__}")
                    except Exception as e:
                        logger.debug(f"Unexpected error reconnecting function generator: {e}")
                
                if not self._osc.is_connected:
                    try:
                        self._osc.connect(fast_fail=True)
                    except (OSError, ConnectionError, TimeoutError) as conn_err:
                        # Fast-fail reconnects expected to fail frequently during connection recovery
                        logger.debug(f"OSC reconnect attempt in progress: {type(conn_err).__name__}")
                    except Exception as e:
                        logger.debug(f"Unexpected error reconnecting oscilloscope: {e}")
            except Exception as e:
                logger.error(f"Critical error in health monitor loop: {e}")

            self._monitor_stop.wait(5.0)
