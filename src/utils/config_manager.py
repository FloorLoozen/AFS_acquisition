"""Simple configuration management for AFS Acquisition."""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from src.utils.logger import get_logger

logger = get_logger("config")


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    camera_queue_size: int = 20
    camera_frame_pool_size: int = 10
    camera_max_fps: int = 120
    hdf5_chunk_size_frames: int = 32
    hdf5_initial_size: int = 5000
    hdf5_write_batch_size: int = 20
    hdf5_flush_interval: float = 1.0
    hdf5_compression: str = "lzf"
    max_worker_threads: int = 16  # Optimized for i7-14700 (20 cores)
    ui_update_interval_ms: int = 16
    enable_frame_pooling: bool = True
    force_gc_interval: int = 2000


@dataclass
class UIConfig:
    """UI configuration."""
    window_geometry: Dict[str, int] = None
    theme: str = "default"
    font_size: int = 9
    auto_save_settings: bool = True
    
    def __post_init__(self):
        if self.window_geometry is None:
            self.window_geometry = {"x": 100, "y": 100, "width": 1280, "height": 800}


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    camera_id: int = 0
    camera_auto_retry: bool = True
    camera_retry_attempts: int = 3
    camera_connection_timeout: float = 5.0
    stage_default_step_mm: float = 0.01
    stage_default_speed_mm_s: float = 0.5
    stage_auto_connect: bool = True
    stage_connection_timeout: float = 10.0
    stage_max_position_mm: float = 50.0
    stage_min_position_mm: float = -50.0
    fg_default_frequency_mhz: float = 14.0
    fg_default_amplitude_vpp: float = 1.0
    fg_auto_connect: bool = True
    fg_connection_timeout: float = 5.0
    fg_max_frequency_mhz: float = 25.0
    fg_max_amplitude_vpp: float = 10.0
    osc_auto_connect: bool = True
    osc_connection_timeout: float = 5.0
    osc_default_timebase_s: float = 1e-3
    osc_default_voltage_scale_v: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration."""
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    max_log_files: int = 10
    enable_performance_logging: bool = True
    log_directory: str = "logs"
    max_log_size_mb: int = 10


@dataclass
class FilesConfig:
    """File and path configuration."""
    default_save_path: str = r"C:\Users\AFS\Documents\Floor\Software\tmp"
    auto_backup: bool = True
    backup_count: int = 5
    temp_directory: str = "temp"
    max_file_age_days: int = 30
    background_compression: bool = True
    wait_for_compression: bool = False


class ConfigManager:
    """Simple configuration manager."""
    
    def __init__(self, config_dir=None):
        # Hardcoded for Windows - faster than Path.home()
        if config_dir is None:
            config_dir = os.path.join(os.environ['USERPROFILE'], '.afs_tracking')
        self.config_dir = config_dir
        self.config_file = os.path.join(self.config_dir, 'config.json')
        
        self.performance = PerformanceConfig()
        self.ui = UIConfig()
        self.hardware = HardwareConfig()
        self.logging = LoggingConfig()
        self.files = FilesConfig()
        
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        if not os.path.exists(self.config_file):
            logger.info("No config found, using defaults")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            sections = {
                'performance': self.performance,
                'ui': self.ui,
                'hardware': self.hardware,
                'logging': self.logging,
                'files': self.files
            }
            
            for name, config in sections.items():
                if name in data:
                    for key, value in data[name].items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                            
            logger.debug("Configuration loaded")
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
    
    def save_config(self):
        """Save configuration to file."""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            
            data = {
                'performance': asdict(self.performance),
                'ui': asdict(self.ui),
                'hardware': asdict(self.hardware),
                'logging': asdict(self.logging),
                'files': asdict(self.files)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Configuration saved")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
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
        """Update window geometry."""
        self.ui.window_geometry = geometry
        if self.ui.auto_save_settings:
            self.save_config()


# Global singleton
_config_manager: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def save_config():
    """Save current configuration."""
    get_config().save_config()
