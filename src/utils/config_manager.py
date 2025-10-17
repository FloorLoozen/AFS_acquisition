"""Configuration management for AFS Tracking System.

Centralized configuration with performance tuning options.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading

from src.utils.logger import get_logger

logger = get_logger("config")


@dataclass
class PerformanceConfig:
    """Performance-related configuration options."""
    # Camera settings
    camera_queue_size: int = 10
    camera_frame_pool_size: int = 5
    camera_max_fps: int = 60
    
    # HDF5 settings
    hdf5_chunk_size_frames: int = 16
    hdf5_initial_size: int = 2000
    hdf5_write_batch_size: int = 10
    hdf5_flush_interval: float = 5.0
    hdf5_compression: str = "lzf"  # "lzf", "gzip", or "none"
    
    # Threading settings
    max_worker_threads: int = 4
    ui_update_interval_ms: int = 33  # ~30 FPS UI updates
    
    # Memory settings
    enable_frame_pooling: bool = True
    force_gc_interval: int = 1000  # Force garbage collection every N frames


@dataclass
class UIConfig:
    """User interface configuration options."""
    window_geometry: Dict[str, int] = None
    theme: str = "default"
    font_size: int = 9
    auto_save_settings: bool = True
    
    def __post_init__(self):
        if self.window_geometry is None:
            self.window_geometry = {"x": 100, "y": 100, "width": 1280, "height": 800}


@dataclass
class HardwareConfig:
    """Hardware-specific configuration."""
    # Camera settings
    camera_id: int = 0
    camera_auto_retry: bool = True
    camera_retry_attempts: int = 3
    
    # Stage settings
    stage_default_step_mm: float = 0.01
    stage_default_speed_mm_s: float = 0.5
    stage_auto_connect: bool = True
    
    # Function generator settings
    fg_default_frequency_mhz: float = 14.0
    fg_default_amplitude_vpp: float = 1.0
    fg_auto_connect: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    max_log_files: int = 10
    enable_performance_logging: bool = True


class ConfigManager:
    """
    Centralized configuration manager for AFS Tracking System.
    
    Features:
    - Automatic config file loading/saving
    - Runtime configuration updates
    - Performance optimization presets
    - Thread-safe configuration access
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path.home() / ".afs_tracking"
        
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "config.json"
        self._lock = threading.RLock()
        
        # Configuration sections
        self.performance = PerformanceConfig()
        self.ui = UIConfig()
        self.hardware = HardwareConfig()
        self.logging = LoggingConfig()
        
        # Load existing configuration
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Update configuration sections
                if 'performance' in data:
                    self._update_dataclass(self.performance, data['performance'])
                
                if 'ui' in data:
                    self._update_dataclass(self.ui, data['ui'])
                
                if 'hardware' in data:
                    self._update_dataclass(self.hardware, data['hardware'])
                
                if 'logging' in data:
                    self._update_dataclass(self.logging, data['logging'])
                
                logger.debug(f"Configuration loaded from {self.config_file}")
            else:
                logger.debug("No existing config file found, using defaults")
                
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with self._lock:
                # Ensure config directory exists
                self.config_dir.mkdir(parents=True, exist_ok=True)
                
                # Prepare data for serialization
                data = {
                    'performance': asdict(self.performance),
                    'ui': asdict(self.ui),
                    'hardware': asdict(self.hardware),
                    'logging': asdict(self.logging),
                    '_metadata': {
                        'version': '3.0',
                        'saved_at': str(Path(__file__).parent.parent.name),
                    }
                }
                
                # Write to file with pretty formatting
                with open(self.config_file, 'w') as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                
                logger.debug(f"Configuration saved to {self.config_file}")
                
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def apply_performance_preset(self, preset_name: str):
        """Apply a performance optimization preset."""
        # Streamlined presets - focus on essential performance settings
        presets = {
            'max_performance': {
                'camera_queue_size': 15,
                'hdf5_write_batch_size': 15, 
                'max_worker_threads': 6,
                'ui_update_interval_ms': 25
            },
            'balanced': {
                'camera_queue_size': 10,
                'hdf5_write_batch_size': 10,
                'max_worker_threads': 4, 
                'ui_update_interval_ms': 33
            },
            'memory_efficient': {
                'camera_queue_size': 5,
                'hdf5_write_batch_size': 5,
                'max_worker_threads': 2,
                'ui_update_interval_ms': 50
            },
        }
        
        if preset_name not in presets:
            logger.warning(f"Unknown preset: {preset_name}")
            return
        
        with self._lock:
            preset_data = presets[preset_name]
            self._update_dataclass(self.performance, preset_data)
            logger.info(f"Applied performance preset: {preset_name}")
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera-related configuration."""
        return {
            'queue_size': self.performance.camera_queue_size,
            'frame_pool_size': self.performance.camera_frame_pool_size,
            'max_fps': self.performance.camera_max_fps,
            'camera_id': self.hardware.camera_id,
            'auto_retry': self.hardware.camera_auto_retry,
            'retry_attempts': self.hardware.camera_retry_attempts,
        }
    
    def get_hdf5_config(self) -> Dict[str, Any]:
        """Get HDF5-related configuration."""
        return {
            'chunk_size_frames': self.performance.hdf5_chunk_size_frames,
            'initial_size': self.performance.hdf5_initial_size,
            'write_batch_size': self.performance.hdf5_write_batch_size,
            'flush_interval': self.performance.hdf5_flush_interval,
            'compression': self.performance.hdf5_compression,
        }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI-related configuration."""
        return {
            'update_interval_ms': self.performance.ui_update_interval_ms,
            'window_geometry': self.ui.window_geometry,
            'theme': self.ui.theme,
            'font_size': self.ui.font_size,
        }
    
    def update_window_geometry(self, geometry: Dict[str, int]):
        """Update window geometry configuration."""
        with self._lock:
            self.ui.window_geometry = geometry
            if self.ui.auto_save_settings:
                self.save_config()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for optimization."""
        try:
            import psutil
            
            # CPU information
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # Disk information
            disk = psutil.disk_usage('.')
            disk_free_gb = disk.free / (1024**3)
            
            return {
                'cpu_cores_physical': cpu_count,
                'cpu_cores_logical': cpu_count_logical,
                'memory_gb': round(memory_gb, 1),
                'disk_free_gb': round(disk_free_gb, 1),
                'platform': os.name,
            }
        except Exception as e:
            logger.debug(f"Error getting system info: {e}")
            return {}
    
    def auto_optimize(self):
        """Automatically optimize configuration based on system capabilities."""
        system_info = self.get_system_info()
        
        if not system_info:
            logger.warning("Could not get system info for auto-optimization")
            return
        
        with self._lock:
            # Adjust thread count based on CPU cores
            if 'cpu_cores_logical' in system_info:
                optimal_threads = min(6, max(2, system_info['cpu_cores_logical'] - 2))
                self.performance.max_worker_threads = optimal_threads
            
            # Adjust memory settings based on available RAM
            if 'memory_gb' in system_info:
                memory_gb = system_info['memory_gb']
                if memory_gb >= 16:
                    # High memory system
                    self.performance.camera_queue_size = 20
                    self.performance.camera_frame_pool_size = 10
                    self.performance.hdf5_initial_size = 5000
                elif memory_gb >= 8:
                    # Medium memory system
                    self.performance.camera_queue_size = 10
                    self.performance.camera_frame_pool_size = 5
                    self.performance.hdf5_initial_size = 2000
                else:
                    # Low memory system
                    self.performance.camera_queue_size = 5
                    self.performance.camera_frame_pool_size = 3
                    self.performance.hdf5_initial_size = 1000
            
            # Adjust based on available disk space
            if 'disk_free_gb' in system_info:
                disk_free = system_info['disk_free_gb']
                if disk_free < 5:
                    logger.warning(f"Low disk space: {disk_free:.1f}GB free")
                    # Use more compression and smaller buffers
                    self.performance.hdf5_compression = "gzip"
                    self.performance.hdf5_flush_interval = 2.0
        
        logger.info("Configuration auto-optimized based on system capabilities")


# Global configuration manager instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def save_config():
    """Save the current configuration."""
    get_config().save_config()


def apply_performance_preset(preset_name: str):
    """Apply a performance preset."""
    get_config().apply_performance_preset(preset_name)


def auto_optimize_config():
    """Auto-optimize configuration based on system."""
    get_config().auto_optimize()


# Example usage
if __name__ == "__main__":
    # Demo configuration management
    config = ConfigManager()
    
    print("System Info:")
    import json
    print(json.dumps(config.get_system_info(), indent=2))
    
    print("\nAuto-optimizing configuration...")
    config.auto_optimize()
    
    print("\nCamera Config:")
    print(json.dumps(config.get_camera_config(), indent=2))
    
    print("\nHDF5 Config:")
    print(json.dumps(config.get_hdf5_config(), indent=2))
    
    config.save_config()
    print(f"\nConfiguration saved to: {config.config_file}")