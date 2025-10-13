"""Performance monitoring utilities for the AFS Tracking System.

Provides lightweight performance measurement, system monitoring,
and optimization guidance for the application.
"""

import time
import threading
import psutil
import gc
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
import weakref
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger("performance")


@dataclass
class TimingStats:
    """Statistics for timing measurements."""
    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_measurement(self, elapsed_time: float):
        """Add a timing measurement."""
        self.total_time += elapsed_time
        self.call_count += 1
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)
        self.recent_times.append(elapsed_time)
    
    @property
    def average_time(self) -> float:
        """Calculate average time per call."""
        return self.total_time / max(1, self.call_count)
    
    @property
    def recent_average(self) -> float:
        """Calculate average of recent measurements."""
        if not self.recent_times:
            return 0.0
        return sum(self.recent_times) / len(self.recent_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'name': self.name,
            'total_calls': self.call_count,
            'total_time_s': round(self.total_time, 6),
            'avg_time_ms': round(self.average_time * 1000, 3),
            'recent_avg_ms': round(self.recent_average * 1000, 3),
            'min_time_ms': round(self.min_time * 1000, 3) if self.min_time != float('inf') else 0,
            'max_time_ms': round(self.max_time * 1000, 3),
            'calls_per_sec': round(self.call_count / max(0.001, self.total_time), 1)
        }


class PerformanceMonitor:
    """
    Lightweight performance monitoring system for AFS Tracking.
    
    Features:
    - Function timing with decorators and context managers
    - System resource monitoring (CPU, memory, disk)
    - Memory leak detection
    - Performance bottleneck identification
    - Real-time statistics
    """
    
    def __init__(self):
        self.timing_stats: Dict[str, TimingStats] = {}
        self.system_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # 1 minute history
        self.lock = threading.RLock()
        
        # System monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 1.0  # seconds
        
        # Memory tracking
        self.object_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.last_gc_count = gc.get_count()
        
        # Performance thresholds (configurable)
        self.thresholds = {
            'cpu_usage': 80.0,      # % CPU usage warning
            'memory_usage': 85.0,   # % Memory usage warning
            'disk_usage': 90.0,     # % Disk usage warning
            'slow_function': 0.1,   # Function calls > 100ms warning
            'frame_drop_rate': 5.0, # % Frame drop rate warning
        }
        
        # Warning throttling to prevent spam (track last warning time)
        self.last_warning_time = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0
        }
        self.warning_interval = 30.0  # Only warn every 30 seconds for resource issues
    
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.record_timing(operation_name, elapsed)
    
    def record_timing(self, name: str, elapsed_time: float):
        """Record a timing measurement."""
        with self.lock:
            if name not in self.timing_stats:
                self.timing_stats[name] = TimingStats(name)
            
            self.timing_stats[name].add_measurement(elapsed_time)
            
            # Check for slow operations
            if elapsed_time > self.thresholds['slow_function']:
                logger.warning(f"Slow operation detected: {name} took {elapsed_time*1000:.1f}ms")
    
    def timer(self, name: Optional[str] = None):
        """Decorator for timing function calls."""
        def decorator(func: Callable):
            timing_name = name or f"{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                with self.measure(timing_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def start_system_monitoring(self):
        """Start background system monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_system,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.debug("Performance monitoring started")
    
    def stop_system_monitoring(self):
        """Stop background system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.debug("Performance monitoring stopped")
    
    def _monitor_system(self):
        """Background system monitoring loop."""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.system_stats['cpu_usage'].append((current_time, cpu_percent))
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_mb = memory.used / (1024 * 1024)
                self.system_stats['memory_usage'].append((current_time, memory_percent))
                self.system_stats['memory_mb'].append((current_time, memory_mb))
                
                # Disk usage (for recording drive)
                try:
                    disk = psutil.disk_usage('.')
                    disk_percent = (disk.used / disk.total) * 100
                    disk_free_gb = disk.free / (1024**3)
                    self.system_stats['disk_usage'].append((current_time, disk_percent))
                    self.system_stats['disk_free_gb'].append((current_time, disk_free_gb))
                except:
                    pass  # Ignore disk monitoring errors
                
                # Process-specific memory
                try:
                    process = psutil.Process()
                    process_memory_mb = process.memory_info().rss / (1024 * 1024)
                    self.system_stats['process_memory_mb'].append((current_time, process_memory_mb))
                except:
                    pass
                
                # Garbage collection stats
                gc_stats = gc.get_count()
                if gc_stats != self.last_gc_count:
                    self.system_stats['gc_collections'].append((current_time, sum(gc_stats)))
                    self.last_gc_count = gc_stats
                
                # Check thresholds
                self._check_thresholds(cpu_percent, memory_percent, 
                                     self.system_stats.get('disk_usage', deque())[-1][1] if self.system_stats.get('disk_usage') else 0)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.debug(f"System monitoring error: {e}")
                time.sleep(1.0)
    
    def _check_thresholds(self, cpu: float, memory: float, disk: float):
        """Check system thresholds and warn if exceeded (throttled to prevent spam)."""
        current_time = time.time()
        
        # Only warn if enough time has passed since last warning
        if cpu > self.thresholds['cpu_usage']:
            if current_time - self.last_warning_time['cpu_usage'] > self.warning_interval:
                logger.warning(f"High CPU usage: {cpu:.1f}% (throttled warnings every 30s)")
                self.last_warning_time['cpu_usage'] = current_time
        
        if memory > self.thresholds['memory_usage']:
            if current_time - self.last_warning_time['memory_usage'] > self.warning_interval:
                logger.warning(f"High memory usage: {memory:.1f}% (throttled warnings every 30s)")
                self.last_warning_time['memory_usage'] = current_time
        
        if disk > self.thresholds['disk_usage']:
            if current_time - self.last_warning_time['disk_usage'] > self.warning_interval:
                logger.warning(f"High disk usage: {disk:.1f}% (throttled warnings every 30s)")
                self.last_warning_time['disk_usage'] = current_time
    
    def get_timing_report(self, top_n: int = 10) -> str:
        """Generate a timing performance report."""
        if not self.timing_stats:
            return "No timing data collected"
        
        # Sort by total time (most time-consuming operations first)
        sorted_stats = sorted(
            self.timing_stats.values(),
            key=lambda s: s.total_time,
            reverse=True
        )
        
        report = ["=== Performance Timing Report ==="]
        report.append(f"{'Operation':<40} {'Calls':<8} {'Total(s)':<10} {'Avg(ms)':<10} {'Recent(ms)':<12} {'Rate(Hz)':<10}")
        report.append("-" * 100)
        
        for stat in sorted_stats[:top_n]:
            stats_dict = stat.get_stats()
            report.append(
                f"{stats_dict['name']:<40} "
                f"{stats_dict['total_calls']:<8} "
                f"{stats_dict['total_time_s']:<10.3f} "
                f"{stats_dict['avg_time_ms']:<10.3f} "
                f"{stats_dict['recent_avg_ms']:<12.3f} "
                f"{stats_dict['calls_per_sec']:<10.1f}"
            )
        
        return "\n".join(report)
    
    def get_system_report(self) -> str:
        """Generate a system performance report."""
        if not self.system_stats:
            return "No system monitoring data available"
        
        report = ["=== System Performance Report ==="]
        
        # Current values
        for metric, data in self.system_stats.items():
            if data:
                latest_time, latest_value = data[-1]
                if 'usage' in metric or 'percent' in metric:
                    report.append(f"{metric.replace('_', ' ').title()}: {latest_value:.1f}%")
                elif 'mb' in metric:
                    report.append(f"{metric.replace('_', ' ').title()}: {latest_value:.1f} MB")
                elif 'gb' in metric:
                    report.append(f"{metric.replace('_', ' ').title()}: {latest_value:.1f} GB")
                else:
                    report.append(f"{metric.replace('_', ' ').title()}: {latest_value}")
        
        return "\n".join(report)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
        }
        
        # Timing statistics
        if self.timing_stats:
            timing_summary = {}
            for name, stats in self.timing_stats.items():
                timing_summary[name] = stats.get_stats()
            summary['timing_stats'] = timing_summary
        
        # System statistics (latest values)
        system_summary = {}
        for metric, data in self.system_stats.items():
            if data:
                _, latest_value = data[-1]
                system_summary[metric] = latest_value
        summary['system_stats'] = system_summary
        
        # Performance recommendations
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check timing stats for bottlenecks
        if self.timing_stats:
            # Find slowest operations
            slow_ops = [(name, stats) for name, stats in self.timing_stats.items() 
                       if stats.recent_average > 0.05]  # > 50ms
            
            if slow_ops:
                slow_ops.sort(key=lambda x: x[1].recent_average, reverse=True)
                top_slow = slow_ops[0]
                recommendations.append(f"Optimize '{top_slow[0]}': averaging {top_slow[1].recent_average*1000:.1f}ms per call")
            
            # Check for frequently called functions
            frequent_ops = [(name, stats) for name, stats in self.timing_stats.items()
                          if stats.call_count > 1000 and stats.recent_average > 0.01]
            
            if frequent_ops:
                recommendations.append("Consider caching or optimization for frequently called operations")
        
        # Check system resources
        if self.system_stats:
            if 'cpu_usage' in self.system_stats and self.system_stats['cpu_usage']:
                recent_cpu = [x[1] for x in list(self.system_stats['cpu_usage'])[-10:]]
                avg_cpu = sum(recent_cpu) / len(recent_cpu)
                if avg_cpu > 70:
                    recommendations.append(f"High CPU usage ({avg_cpu:.1f}%): consider reducing processing load")
            
            if 'memory_usage' in self.system_stats and self.system_stats['memory_usage']:
                recent_memory = [x[1] for x in list(self.system_stats['memory_usage'])[-10:]]
                avg_memory = sum(recent_memory) / len(recent_memory)
                if avg_memory > 80:
                    recommendations.append(f"High memory usage ({avg_memory:.1f}%): check for memory leaks")
        
        if not recommendations:
            recommendations.append("Performance looks good! No specific recommendations at this time.")
        
        return recommendations
    
    def reset_stats(self):
        """Reset all performance statistics."""
        with self.lock:
            self.timing_stats.clear()
            self.system_stats.clear()
            self.object_counts.clear()
        logger.info("Performance statistics reset")
    
    def memory_profile(self, func_name: str = ""):
        """Decorator for memory profiling."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Measure memory before
                process = psutil.Process()
                mem_before = process.memory_info().rss
                
                # Call function
                result = func(*args, **kwargs)
                
                # Measure memory after
                mem_after = process.memory_info().rss
                mem_diff = (mem_after - mem_before) / (1024 * 1024)  # MB
                
                if abs(mem_diff) > 10:  # More than 10MB change
                    logger.info(f"Memory change in {func_name or func.__name__}: {mem_diff:+.1f} MB")
                
                return result
            return wrapper
        return decorator


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


# Convenience decorators
def timer(name: Optional[str] = None):
    """Decorator for timing functions."""
    return get_performance_monitor().timer(name)


def memory_profile(name: str = ""):
    """Decorator for memory profiling."""
    return get_performance_monitor().memory_profile(name)


@contextmanager
def measure_time(operation_name: str):
    """Context manager for measuring operation time."""
    with get_performance_monitor().measure(operation_name):
        yield


def start_monitoring():
    """Start system performance monitoring."""
    get_performance_monitor().start_system_monitoring()


def stop_monitoring():
    """Stop system performance monitoring."""
    get_performance_monitor().stop_system_monitoring()


def get_performance_report() -> str:
    """Get a comprehensive performance report."""
    monitor = get_performance_monitor()
    timing_report = monitor.get_timing_report()
    system_report = monitor.get_system_report()
    
    return f"{timing_report}\n\n{system_report}"


# Example usage
if __name__ == "__main__":
    # Demo performance monitoring
    monitor = get_performance_monitor()
    monitor.start_system_monitoring()
    
    # Simulate some operations
    @timer("test_operation")
    def test_function():
        time.sleep(0.01)  # 10ms operation
        return "done"
    
    # Run test operations
    for i in range(100):
        test_function()
        time.sleep(0.001)
    
    # Generate reports
    print(monitor.get_timing_report())
    print("\n" + monitor.get_system_report())
    
    summary = monitor.get_performance_summary()
    print(f"\nRecommendations: {summary['recommendations']}")
    
    monitor.stop_system_monitoring()