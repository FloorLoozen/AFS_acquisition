"""
Retry and Circuit Breaker Utilities

Provides retry logic with exponential backoff and circuit breaker pattern
for robust error handling in hardware communication.
"""
import time
import functools
from typing import Callable, TypeVar, Optional, Tuple, Type
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("retry_utils")

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for hardware failures.
    
    Prevents repeated attempts to access failing hardware by "opening the circuit"
    after a threshold of failures. Periodically allows test attempts to check if
    the hardware has recovered.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 success_threshold: int = 2):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again (OPEN -> HALF_OPEN)
            success_threshold: Consecutive successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, transitioning OPEN -> HALF_OPEN if timeout elapsed."""
        if (self._state == CircuitState.OPEN and 
            self._last_failure_time and
            time.time() - self._last_failure_time >= self.recovery_timeout):
            logger.info("Circuit breaker: Entering HALF_OPEN state for recovery test")
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
        return self._state
    
    def call(self, func: Callable[[], T]) -> T:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            
        Returns:
            Result of function call
            
        Raises:
            RuntimeError: If circuit is OPEN
            Exception: Any exception from the function (if circuit allows call)
        """
        current_state = self.state
        
        if current_state == CircuitState.OPEN:
            raise RuntimeError("Circuit breaker is OPEN - hardware appears to be failing")
        
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self._failure_count = 0
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                logger.info("Circuit breaker: Recovered, moving to CLOSED state")
                self._state = CircuitState.CLOSED
                self._success_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker: Recovery test failed, back to OPEN")
            self._state = CircuitState.OPEN
            self._success_count = 0
        elif self._failure_count >= self.failure_threshold:
            logger.error(f"Circuit breaker: Opening after {self._failure_count} failures")
            self._state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        logger.info("Circuit breaker: Manual reset to CLOSED")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    exponential: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential: If True, use exponential backoff; if False, constant delay
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_attempts=3, base_delay=1.0)
        def connect_to_device():
            # ... connection code ...
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt >= max_attempts:
                        logger.error(f"{func.__name__}: All {max_attempts} attempts failed")
                        raise
                    
                    # Calculate delay
                    if exponential:
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    else:
                        delay = base_delay
                    
                    logger.warning(
                        f"{func.__name__}: Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            # Should never reach here, but for type safety
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__}: Unexpected retry loop exit")
        
        return wrapper
    return decorator


def timeout_handler(timeout_seconds: float) -> Callable:
    """
    Decorator to add timeout to function calls (simple implementation).
    
    Note: This is a simplified version. For production, consider using
    threading.Timer or signal-based timeouts.
    
    Args:
        timeout_seconds: Maximum time to allow function to run
        
    Returns:
        Decorated function with timeout
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import threading
            result = []
            exception = []
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    exception.append(e)
            
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                logger.error(f"{func.__name__}: Timeout after {timeout_seconds}s")
                raise TimeoutError(f"{func.__name__} exceeded timeout of {timeout_seconds}s")
            
            if exception:
                raise exception[0]
            
            if result:
                return result[0]
            
            raise RuntimeError(f"{func.__name__}: No result and no exception")
        
        return wrapper
    return decorator
