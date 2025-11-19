"""Performance monitoring and metrics collection.

Provides decorators and utilities for tracking application performance,
including frame processing, compression times, memory usage, and bottleneck detection.
"""

import time
import psutil
import functools
import threading
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from src.utils.logger import get_logger

logger = get_logger("performance_monitor")


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Frame processing metrics
    frames_captured: int = 0
    frames_dropped: int = 0
    frames_written: int = 0
    avg_capture_fps: float = 0.0
    avg_write_fps: float = 0.0
    
    # Compression metrics
    compression_count: int = 0
    total_compression_time: float = 0.0
    avg_compression_time: float = 0.0
    compression_ratio: float = 0.0
    
    # Memory metrics
    memory_used_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_percent: float = 0.0
    
    # CPU metrics
    cpu_percent: float = 0.0
    thread_count: int = 0
    
    # GPU metrics (if available)
    gpu_frames_processed: int = 0
    gpu_avg_time_ms: float = 0.0
    
    # Timing metrics
    operation_times: Dict[str, List[float]] = field(default_factory=dict)
    
    # Recording session metrics
    session_start_time: Optional[float] = None
    session_duration: float = 0.0
    total_data_written_mb: float = 0.0
    
    def reset(self):
        """Reset all metrics to initial state."""
        self.frames_captured = 0
        self.frames_dropped = 0
        self.frames_written = 0
        self.avg_capture_fps = 0.0
        self.avg_write_fps = 0.0
        self.compression_count = 0
        self.total_compression_time = 0.0
        self.avg_compression_time = 0.0
        self.compression_ratio = 0.0
        self.memory_used_mb = 0.0
        self.memory_peak_mb = 0.0
        self.memory_percent = 0.0
        self.cpu_percent = 0.0
        self.thread_count = 0
        self.gpu_frames_processed = 0
        self.gpu_avg_time_ms = 0.0
        self.operation_times.clear()
        self.session_start_time = None
        self.session_duration = 0.0
        self.total_data_written_mb = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            'frames': {
                'captured': self.frames_captured,
                'dropped': self.frames_dropped,
                'written': self.frames_written,
                'drop_rate': f"{(self.frames_dropped / max(1, self.frames_captured) * 100):.2f}%",
                'capture_fps': f"{self.avg_capture_fps:.1f}",
                'write_fps': f"{self.avg_write_fps:.1f}"
            },
            'compression': {
                'count': self.compression_count,
                'total_time': f"{self.total_compression_time:.2f}s",
                'avg_time': f"{self.avg_compression_time:.2f}s",
                'ratio': f"{self.compression_ratio:.1f}%"
            },
            'memory': {
                'current_mb': f"{self.memory_used_mb:.1f}",
                'peak_mb': f"{self.memory_peak_mb:.1f}",
                'percent': f"{self.memory_percent:.1f}%"
            },
            'cpu': {
                'percent': f"{self.cpu_percent:.1f}%",
                'threads': self.thread_count
            },
            'gpu': {
                'frames_processed': self.gpu_frames_processed,
                'avg_time_ms': f"{self.gpu_avg_time_ms:.2f}"
            },
            'session': {
                'duration': f"{self.session_duration:.1f}s",
                'data_written_mb': f"{self.total_data_written_mb:.1f}"
            }
        }


class PerformanceMonitor:
    """Global performance monitor singleton."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.metrics = PerformanceMetrics()
        self._lock = threading.RLock()
        self._process = psutil.Process()
        
        # Rolling windows for FPS calculation
        self._capture_times = deque(maxlen=100)
        self._write_times = deque(maxlen=100)
        
        # Bottleneck detection
        self._slow_operations: Dict[str, List[float]] = {}
        self._slow_threshold_ms = 50.0  # Operations taking >50ms are logged
        
        self._initialized = True
        logger.info("Performance monitor initialized")
    
    def record_frame_captured(self):
        """Record a frame capture event."""
        with self._lock:
            self.metrics.frames_captured += 1
            current_time = time.time()
            self._capture_times.append(current_time)
            
            # Calculate rolling FPS
            if len(self._capture_times) > 1:
                time_span = self._capture_times[-1] - self._capture_times[0]
                if time_span > 0:
                    self.metrics.avg_capture_fps = len(self._capture_times) / time_span
    
    def record_frame_dropped(self):
        """Record a dropped frame."""
        with self._lock:
            self.metrics.frames_dropped += 1
    
    def record_frame_written(self, data_size_mb: float = 0.0):
        """Record a frame write event."""
        with self._lock:
            self.metrics.frames_written += 1
            self.metrics.total_data_written_mb += data_size_mb
            current_time = time.time()
            self._write_times.append(current_time)
            
            # Calculate rolling FPS
            if len(self._write_times) > 1:
                time_span = self._write_times[-1] - self._write_times[0]
                if time_span > 0:
                    self.metrics.avg_write_fps = len(self._write_times) / time_span
    
    def record_compression(self, duration: float, original_size_mb: float, 
                          compressed_size_mb: float):
        """Record a compression operation."""
        with self._lock:
            self.metrics.compression_count += 1
            self.metrics.total_compression_time += duration
            self.metrics.avg_compression_time = (
                self.metrics.total_compression_time / self.metrics.compression_count
            )
            
            if original_size_mb > 0:
                ratio = ((original_size_mb - compressed_size_mb) / original_size_mb) * 100
                self.metrics.compression_ratio = ratio
    
    def record_gpu_processing(self, frame_count: int, total_time_ms: float):
        """Record GPU processing metrics."""
        with self._lock:
            self.metrics.gpu_frames_processed += frame_count
            if frame_count > 0:
                self.metrics.gpu_avg_time_ms = total_time_ms / frame_count
    
    def record_operation_time(self, operation_name: str, duration: float):
        """Record timing for a named operation."""
        with self._lock:
            if operation_name not in self.metrics.operation_times:
                self.metrics.operation_times[operation_name] = []
            
            self.metrics.operation_times[operation_name].append(duration)
            
            # Detect slow operations (bottlenecks)
            duration_ms = duration * 1000
            if duration_ms > self._slow_threshold_ms:
                if operation_name not in self._slow_operations:
                    self._slow_operations[operation_name] = []
                self._slow_operations[operation_name].append(duration_ms)
                
                # Log warning for very slow operations
                if duration_ms > 200:
                    logger.warning(f"Slow operation detected: {operation_name} took {duration_ms:.1f}ms")
    
    def update_system_metrics(self):
        """Update system-level metrics (CPU, memory, threads)."""
        with self._lock:
            try:
                # Memory metrics
                mem_info = self._process.memory_info()
                self.metrics.memory_used_mb = mem_info.rss / (1024 * 1024)
                self.metrics.memory_peak_mb = max(self.metrics.memory_peak_mb, 
                                                  self.metrics.memory_used_mb)
                self.metrics.memory_percent = self._process.memory_percent()
                
                # CPU metrics
                self.metrics.cpu_percent = self._process.cpu_percent()
                self.metrics.thread_count = self._process.num_threads()
                
            except Exception as e:
                logger.debug(f"Error updating system metrics: {e}")
    
    def start_session(self):
        """Start a recording session."""
        with self._lock:
            self.metrics.session_start_time = time.time()
            logger.debug("Performance monitoring session started")
    
    def end_session(self):
        """End a recording session."""
        with self._lock:
            if self.metrics.session_start_time:
                self.metrics.session_duration = time.time() - self.metrics.session_start_time
                logger.debug(f"Performance monitoring session ended: {self.metrics.session_duration:.1f}s")
    
    def get_bottlenecks(self, top_n: int = 5) -> List[tuple]:
        """Get top N slowest operations.
        
        Returns:
            List of (operation_name, avg_time_ms, call_count) tuples
        """
        with self._lock:
            bottlenecks = []
            for op_name, times in self.metrics.operation_times.items():
                if times:
                    avg_time = sum(times) / len(times) * 1000  # Convert to ms
                    bottlenecks.append((op_name, avg_time, len(times)))
            
            # Sort by average time, descending
            bottlenecks.sort(key=lambda x: x[1], reverse=True)
            return bottlenecks[:top_n]
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            return self.metrics
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics.reset()
            self._capture_times.clear()
            self._write_times.clear()
            self._slow_operations.clear()
            logger.debug("Performance metrics reset")
    
    def print_summary(self):
        """Print performance summary to console."""
        summary = self.metrics.get_summary()
        logger.info("=== Performance Summary ===")
        logger.info(f"Frames: {summary['frames']}")
        logger.info(f"Compression: {summary['compression']}")
        logger.info(f"Memory: {summary['memory']}")
        logger.info(f"CPU: {summary['cpu']}")
        logger.info(f"GPU: {summary['gpu']}")
        logger.info(f"Session: {summary['session']}")
        
        # Print bottlenecks
        bottlenecks = self.get_bottlenecks(5)
        if bottlenecks:
            logger.info("Top 5 slowest operations:")
            for op_name, avg_time, count in bottlenecks:
                logger.info(f"  {op_name}: {avg_time:.2f}ms avg ({count} calls)")


# Singleton instance
_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _monitor


# Decorators for automatic performance tracking

def profile_performance(operation_name: Optional[str] = None):
    """Decorator to profile function execution time.
    
    Args:
        operation_name: Optional name for the operation. If not provided,
                       uses the function name.
    
    Usage:
        @profile_performance("my_operation")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                _monitor.record_operation_time(op_name, duration)
        
        return wrapper
    return decorator


def track_memory(func: Callable) -> Callable:
    """Decorator to track memory usage before and after function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _monitor.update_system_metrics()
        mem_before = _monitor.metrics.memory_used_mb
        
        result = func(*args, **kwargs)
        
        _monitor.update_system_metrics()
        mem_after = _monitor.metrics.memory_used_mb
        mem_delta = mem_after - mem_before
        
        if abs(mem_delta) > 10:  # Log if >10MB change
            logger.debug(f"{func.__name__} memory delta: {mem_delta:+.1f}MB")
        
        return result
    
    return wrapper
