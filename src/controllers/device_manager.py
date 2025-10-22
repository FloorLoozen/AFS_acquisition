"""
Device Manager

Centralized manager for hardware controllers (function generator, oscilloscope, etc.).
Provides a single access point for controllers and an optional background health monitor
that can attempt reconnects if devices drop.

This keeps controller instantiation unified and reduces the risk of multiple concurrent
VISA sessions to the same physical instrument.
"""
import threading
import time
from typing import Optional, TYPE_CHECKING

from src.utils.logger import get_logger

# Avoid circular imports by using TYPE_CHECKING
if TYPE_CHECKING:
    from src.controllers.function_generator_controller import FunctionGeneratorController
    from src.controllers.oscilloscope_controller import OscilloscopeController

logger = get_logger("device_manager")


class DeviceManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        # Lazy imports to avoid circular imports at module import time
        from src.controllers.function_generator_controller import get_function_generator_controller
        from src.controllers.oscilloscope_controller import get_oscilloscope_controller
        # Controllers (singletons provided by their modules)
        self._fg = get_function_generator_controller()
        self._osc = get_oscilloscope_controller()

        # Option to disable automatic oscilloscope operations (useful for fast startup)
        # Can be toggled by the main UI to avoid attempting RS-232 connections during fast startup
        self._disable_osc = False

        # Health monitor thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._monitor_interval = 5.0  # seconds
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> 'DeviceManager':
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_function_generator(self):
        return self._fg

    def get_oscilloscope(self):
        return self._osc

    def connect_all(self, fast_fail: bool = True):
        """Attempt to connect to all managed devices.

        If a device is already connected it is left alone. Returns a dict with
        results for each device.
        
        Args:
            fast_fail: If True, use fast connection attempts (default for startup)
        """
        results = {}
        with self._lock:
            # Function generator
            try:
                if not self._fg.is_connected:
                    self._fg.connect(fast_fail=fast_fail)
                results['function_generator'] = {'connected': self._fg.is_connected}
            except Exception as e:
                results['function_generator'] = {'connected': False, 'error': str(e)}

            # Oscilloscope (skip if disabled for fast startup)
            if getattr(self, '_disable_osc', False):
                results['oscilloscope'] = {'connected': False, 'message': 'Oscilloscope connections disabled'}
            else:
                try:
                    if not self._osc.is_connected:
                        self._osc.connect(fast_fail=fast_fail)
                    results['oscilloscope'] = {'connected': self._osc.is_connected}
                except Exception as e:
                    results['oscilloscope'] = {'connected': False, 'error': str(e)}

        return results

    def start_health_monitor(self, interval: float = 5.0):
        """Start a background thread that periodically checks device connections
        and attempts to reconnect if they're disconnected.
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
        with self._lock:
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_stop.set()
                self._monitor_thread.join(timeout=2.0)
                logger.info("DeviceManager: health monitor stopped")

    def _monitor_loop(self):
        while not self._monitor_stop.is_set():
            try:
                with self._lock:
                    # Try reconnects for devices that are disconnected
                    # Use fast_fail for background reconnection attempts
                    try:
                        if not self._fg.is_connected:
                            # Silent reconnect attempt
                            try:
                                if self._fg.connect(fast_fail=True):
                                    logger.info("Function generator reconnected successfully")
                            except Exception:
                                pass  # Silent failure
                    except Exception:
                        pass  # Silent failure

                    try:
                        # Skip osc reconnects if disabled
                        if getattr(self, '_disable_osc', False):
                            pass
                        else:
                            if not self._osc.is_connected:
                                # Silent reconnect attempt
                                try:
                                    if self._osc.connect(fast_fail=True):
                                        logger.info("Oscilloscope reconnected successfully")
                                except Exception:
                                    pass  # Silent failure
                    except Exception:
                        pass  # Silent failure
            except Exception:
                pass  # Silent failure

            # Sleep until next check, with early exit possible
            self._monitor_stop.wait(self._monitor_interval)
