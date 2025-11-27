"""Retry and Circuit Breaker utilities for robust error handling."""

import time
import functools
from typing import Callable, TypeVar
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("retry_utils")
T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern to prevent repeated attempts to failing hardware."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
    
    @property
    def state(self) -> CircuitState:
        """Get current state, transitioning OPEN -> HALF_OPEN if timeout elapsed."""
        if (self._state == CircuitState.OPEN and self._last_failure_time and
            time.time() - self._last_failure_time >= self.recovery_timeout):
            self._state = CircuitState.HALF_OPEN
        return self._state
    
    def call(self, func: Callable[[], T]) -> T:
        """Call function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            raise RuntimeError("Circuit breaker is OPEN - hardware appears to be failing")
        
        try:
            result = func()
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
            return result
        except Exception:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
            raise
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None


def retry_with_backoff(max_attempts: int = 3, base_delay: float = 0.5,
                      max_delay: float = 30.0, exponential: bool = True):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt >= max_attempts:
                        raise
                    
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay) if exponential else base_delay
                    logger.warning(f"{func.__name__}: Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        return wrapper
    return decorator
