"""
Device Manager

Centralized manager for hardware controllers (function generator).
Provides a single access point for controllers and an optional background health monitor
that can attempt reconnects if devices drop.

This keeps controller instantiation unified and reduces the risk of multiple concurrent
VISA sessions to the same physical instrument.
"""
import threading
from typing import Optional, Dict, Any

from src.utils.logger import get_logger

logger = get_logger("device_manager")


class DeviceManager:
    """Centralized hardware device manager with health monitoring."""
    
    _instance: Optional['DeviceManager'] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        """Initialize device manager with function generator controller."""
        from src.controllers.function_generator_controller import get_function_generator_controller
        
        # Controllers (singletons provided by their modules)
        self._fg = get_function_generator_controller()

        # Health monitor thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._monitor_interval = 5.0  # seconds
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> 'DeviceManager':
        """Get or create the singleton DeviceManager instance.
        
        Returns:
            DeviceManager: The singleton instance
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_function_generator(self):
        """Get the function generator controller.
        
        Returns:
            FunctionGeneratorController: The function generator controller instance
        """
        return self._fg

    def connect_all(self, fast_fail: bool = True) -> Dict[str, Any]:
        """Attempt to connect to all managed devices.

        If a device is already connected it is left alone.
        
        Args:
            fast_fail: If True, use fast connection attempts (default for startup)
            
        Returns:
            dict: Connection results for each device
        """
        results = {}
        with self._lock:
            # Function generator
            try:
                if not self._fg.is_connected:
                    self._fg.connect(fast_fail=fast_fail)
                results['function_generator'] = {'connected': self._fg.is_connected}
            except Exception as e:
                logger.error(f"Failed to connect function generator: {e}")
                results['function_generator'] = {'connected': False, 'error': str(e)}

        return results

    def start_health_monitor(self, interval: float = 5.0):
        """Start a background thread that periodically checks device connections
        and attempts to reconnect if they're disconnected.
        
        Args:
            interval: Check interval in seconds (minimum 1.0)
        """
        with self._lock:
            if self._monitor_thread and self._monitor_thread.is_alive():
                logger.debug("Health monitor already running")
                return

            self._monitor_interval = max(1.0, float(interval))
            self._monitor_stop.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("DeviceManager: health monitor started")

    def stop_health_monitor(self):
        """Stop the health monitor thread."""
        with self._lock:
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_stop.set()
                self._monitor_thread.join(timeout=2.0)
                logger.info("DeviceManager: health monitor stopped")

    def _monitor_loop(self):
        """Health monitor loop that attempts to reconnect disconnected devices."""
        while not self._monitor_stop.is_set():
            try:
                with self._lock:
                    # Try reconnects for devices that are disconnected
                    if not self._fg.is_connected:
                        try:
                            if self._fg.connect(fast_fail=True):
                                logger.info("Function generator reconnected successfully")
                        except Exception as e:
                            logger.debug(f"Failed to reconnect function generator: {e}")
                            
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")

            # Sleep until next check, with early exit possible
            self._monitor_stop.wait(self._monitor_interval)
