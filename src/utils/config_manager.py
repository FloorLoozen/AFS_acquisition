"""
Enhanced configuration management for AFS Acquisition.

Centralized configuration with validation, performance tuning, and robust error handling.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, ClassVar, Union, List, Callable
from dataclasses import dataclass, asdict, field
import threading

from src.utils.logger import get_logger
from src.utils.exceptions import ConfigurationError

logger = get_logger("config")


@dataclass
class PerformanceConfig:
    """Performance-related configuration options - optimized for maximum performance."""
    # Camera settings - maximum performance defaults
    camera_queue_size: int = 20
    camera_frame_pool_size: int = 10
    camera_max_fps: int = 120
    
    # HDF5 settings - maximum performance defaults
    hdf5_chunk_size_frames: int = 32
    hdf5_initial_size: int = 5000
    hdf5_write_batch_size: int = 20
    hdf5_flush_interval: float = 1.0  # Faster flushing for maximum performance
    hdf5_compression: str = "lzf"  # Fast compression for max performance
    
    # Threading settings - maximum performance defaults
    max_worker_threads: int = 8
    ui_update_interval_ms: int = 16  # ~60 FPS UI updates
    
    # Memory settings - maximum performance defaults
    enable_frame_pooling: bool = True
    force_gc_interval: int = 2000  # Less frequent GC for better performance
    
    def validate(self) -> None:
        """Validate performance configuration."""
        if self.camera_queue_size <= 0:
            raise ConfigurationError("Camera queue size must be positive")
        if self.camera_frame_pool_size <= 0:
            raise ConfigurationError("Camera frame pool size must be positive")
        if self.camera_max_fps <= 0:
            raise ConfigurationError("Camera max FPS must be positive")
        if self.hdf5_chunk_size_frames <= 0:
            raise ConfigurationError("HDF5 chunk size must be positive")
        if self.hdf5_initial_size <= 0:
            raise ConfigurationError("HDF5 initial size must be positive")
        if self.hdf5_write_batch_size <= 0:
            raise ConfigurationError("HDF5 write batch size must be positive")
        if self.hdf5_flush_interval <= 0:
            raise ConfigurationError("HDF5 flush interval must be positive")
        if self.max_worker_threads <= 0:
            raise ConfigurationError("Max worker threads must be positive")
        if self.ui_update_interval_ms <= 0:
            raise ConfigurationError("UI update interval must be positive")
        if self.force_gc_interval <= 0:
            raise ConfigurationError("GC interval must be positive")
        if self.hdf5_compression not in ["lzf", "gzip", "szip", None]:
            raise ConfigurationError(f"Invalid HDF5 compression: {self.hdf5_compression}")


@dataclass
class UIConfig:
    """User interface configuration options."""
    window_geometry: Optional[Dict[str, int]] = None
    theme: str = "default"
    font_size: int = 9
    auto_save_settings: bool = True
    
    def __post_init__(self):
        if self.window_geometry is None:
            self.window_geometry = {"x": 100, "y": 100, "width": 1280, "height": 800}
            
    def validate(self) -> None:
        """Validate UI configuration."""
        if self.font_size <= 0:
            raise ConfigurationError("Font size must be positive")
        if self.theme not in ["default", "dark", "light"]:
            raise ConfigurationError(f"Invalid theme: {self.theme}")
        if self.window_geometry:
            required_keys = ["x", "y", "width", "height"]
            if not all(key in self.window_geometry for key in required_keys):
                raise ConfigurationError("Window geometry must contain x, y, width, height")
            if any(self.window_geometry[key] <= 0 for key in ["width", "height"]):
                raise ConfigurationError("Window width and height must be positive")


@dataclass
class HardwareConfig:
    """Hardware-specific configuration."""
    # Camera settings
    camera_id: int = 0
    camera_auto_retry: bool = True
    camera_retry_attempts: int = 3
    camera_connection_timeout: float = 5.0
    
    # Stage settings
    stage_default_step_mm: float = 0.01
    stage_default_speed_mm_s: float = 0.5
    stage_auto_connect: bool = True
    stage_connection_timeout: float = 10.0
    stage_max_position_mm: float = 50.0
    stage_min_position_mm: float = -50.0
    
    # Function generator settings
    fg_default_frequency_mhz: float = 14.0
    fg_default_amplitude_vpp: float = 1.0
    fg_auto_connect: bool = True
    fg_connection_timeout: float = 5.0
    fg_max_frequency_mhz: float = 25.0
    fg_max_amplitude_vpp: float = 10.0
    
    # Oscilloscope settings
    osc_auto_connect: bool = True
    osc_connection_timeout: float = 5.0
    osc_default_timebase_s: float = 1e-3
    osc_default_voltage_scale_v: float = 1.0
    
    def validate(self) -> None:
        """Validate hardware configuration."""
        if self.camera_id < 0:
            raise ConfigurationError("Camera ID must be non-negative")
        if self.camera_retry_attempts < 0:
            raise ConfigurationError("Camera retry attempts must be non-negative")
        if self.camera_connection_timeout <= 0:
            raise ConfigurationError("Camera connection timeout must be positive")
        if self.stage_default_step_mm <= 0:
            raise ConfigurationError("Stage default step must be positive")
        if self.stage_default_speed_mm_s <= 0:
            raise ConfigurationError("Stage default speed must be positive")
        if self.stage_connection_timeout <= 0:
            raise ConfigurationError("Stage connection timeout must be positive")
        if self.stage_min_position_mm >= self.stage_max_position_mm:
            raise ConfigurationError("Stage min position must be less than max position")
        if self.fg_default_frequency_mhz <= 0:
            raise ConfigurationError("Function generator frequency must be positive")
        if self.fg_default_amplitude_vpp <= 0:
            raise ConfigurationError("Function generator amplitude must be positive")
        if self.fg_connection_timeout <= 0:
            raise ConfigurationError("Function generator connection timeout must be positive")
        if self.fg_max_frequency_mhz <= 0:
            raise ConfigurationError("Function generator max frequency must be positive")
        if self.fg_max_amplitude_vpp <= 0:
            raise ConfigurationError("Function generator max amplitude must be positive")
        if self.osc_connection_timeout <= 0:
            raise ConfigurationError("Oscilloscope connection timeout must be positive")
        if self.osc_default_timebase_s <= 0:
            raise ConfigurationError("Oscilloscope timebase must be positive")
        if self.osc_default_voltage_scale_v <= 0:
            raise ConfigurationError("Oscilloscope voltage scale must be positive")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    max_log_files: int = 10
    enable_performance_logging: bool = True
    log_directory: str = "logs"
    max_log_size_mb: int = 10
    
    def validate(self) -> None:
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.console_level not in valid_levels:
            raise ConfigurationError(f"Invalid console log level: {self.console_level}")
        if self.file_level not in valid_levels:
            raise ConfigurationError(f"Invalid file log level: {self.file_level}")
        if self.max_log_files <= 0:
            raise ConfigurationError("Max log files must be positive")
        if self.max_log_size_mb <= 0:
            raise ConfigurationError("Max log size must be positive")


@dataclass
class FilesConfig:
    """File and path configuration."""
    default_save_path: str = "C:/Users/fAFS/Documents/Floor/tmp"
    auto_backup: bool = True
    backup_count: int = 5
    temp_directory: str = "temp"
    max_file_age_days: int = 30
    # Post-recording compression behavior
    background_compression: bool = True  # If True, run post-process compression in background
    wait_for_compression: bool = False    # If True, wait for compression to finish before returning
    
    def validate(self) -> None:
        """Validate files configuration."""
        # Check if parent directory of save path exists
        save_path = Path(self.default_save_path)
        if not save_path.parent.exists():
            raise ConfigurationError(f"Parent directory of save path does not exist: {save_path.parent}")
        if self.backup_count < 0:
            raise ConfigurationError("Backup count must be non-negative")
        if self.max_file_age_days <= 0:
            raise ConfigurationError("Max file age must be positive")
        # background_compression and wait_for_compression are booleans - no further validation


class ConfigManager:
    """
    Enhanced configuration manager for AFS Acquisition.
    
    Features:
    - Automatic config file loading/saving with validation
    - Runtime configuration updates with error handling
    - Performance optimization presets
    - Thread-safe configuration access
    - Configuration change notifications
    - Backup and recovery
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir or Path.home() / ".afs_tracking")
        self.config_file = self.config_dir / "config.json"
        self.backup_file = self.config_dir / "config_backup.json"
        self._lock = threading.RLock()
        self._watchers: List[Callable] = []
        
        # Configuration sections
        self.performance = PerformanceConfig()
        self.ui = UIConfig()
        self.hardware = HardwareConfig()
        self.logging = LoggingConfig()
        self.files = FilesConfig()
        
        # Load existing configuration
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file with fallback to backup."""
        try:
            config_loaded = False
            
            # Try main config file first
            if self.config_file.exists():
                config_loaded = self._load_from_file(self.config_file)
                
            # Fallback to backup if main config failed
            if not config_loaded and self.backup_file.exists():
                logger.warning("Main config failed, trying backup...")
                config_loaded = self._load_from_file(self.backup_file)
                
            if not config_loaded:
                logger.info("No existing config found, using defaults")
                
            # Validate configuration
            self._validate_all_sections()
            
            logger.debug(f"Configuration loaded successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            # Reset to defaults on critical error
            self._reset_to_defaults()
            return False
    
    def _load_from_file(self, file_path: Path) -> bool:
        """Load configuration from a specific file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update configuration sections
            config_mapping = {
                'performance': self.performance,
                'ui': self.ui,
                'hardware': self.hardware,
                'logging': self.logging,
                'files': self.files
            }
            
            for section_name, config_obj in config_mapping.items():
                if section_name in data:
                    self._update_dataclass(config_obj, data[section_name])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file with backup and atomic write.
        
        Returns:
            True if save successful, False otherwise
        """
        try:
            with self._lock:
                # Validate before saving
                try:
                    self._validate_all_sections()
                except ConfigurationError as e:
                    logger.error(f"Cannot save invalid configuration: {e}")
                    return False
                
                # Ensure config directory exists
                try:
                    self.config_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logger.error(f"Cannot create config directory: {e}")
                    return False
                
                # Create backup of existing config
                if self.config_file.exists():
                    try:
                        self.config_file.replace(self.backup_file)
                    except Exception as e:
                        logger.warning(f"Failed to create config backup: {e}")
                        # Continue anyway - backup failure shouldn't prevent saving
                
                # Prepare data for serialization
                data = {
                    'performance': asdict(self.performance),
                    'ui': asdict(self.ui),
                    'hardware': asdict(self.hardware),
                    'logging': asdict(self.logging),
                    'files': asdict(self.files),
                    '_metadata': {
                        'version': '4.0',
                        'saved_at': str(Path(__file__).parent.parent.name),
                    }
                }
                
                # Write to temporary file first for atomic operation
                temp_file = self.config_file.with_suffix('.tmp')
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, sort_keys=True)
                except (OSError, IOError) as e:
                    logger.error(f"Failed to write config to temporary file: {e}")
                    # Clean up temp file if it exists
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                        except:
                            pass
                    return False
                
                # Atomic replace
                try:
                    temp_file.replace(self.config_file)
                except (OSError, IOError) as e:
                    logger.error(f"Failed to replace config file: {e}")
                    # Clean up temp file
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                        except:
                            pass
                    return False
                
                logger.debug(f"Configuration saved to {self.config_file}")
                return True
                
        except Exception as e:
            logger.error(f"Unexpected error saving config: {e}")
            return False
    
    def _validate_all_sections(self) -> None:
        """Validate all configuration sections."""
        try:
            self.performance.validate()
            self.ui.validate()
            self.hardware.validate()
            self.logging.validate()
            self.files.validate()
        except ConfigurationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _reset_to_defaults(self) -> None:
        """Reset all configuration to defaults."""
        self.performance = PerformanceConfig()
        self.ui = UIConfig()
        self.hardware = HardwareConfig()
        self.logging = LoggingConfig()
        self.files = FilesConfig()
    
    def _update_dataclass(self, obj, data: Dict[str, Any]) -> None:
        """Update dataclass fields from dictionary with type checking."""
        for key, value in data.items():
            if hasattr(obj, key):
                # Get the expected type from dataclass field
                field_type = obj.__annotations__.get(key)
                
                # Only check basic types to avoid subscripted generic issues
                basic_types = (int, float, str, bool)
                if field_type in basic_types and not isinstance(value, field_type):
                    try:
                        # Attempt type conversion for basic types
                        value = field_type(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Type mismatch for {key}: expected {field_type}, got {type(value)}")
                        continue
                        
                setattr(obj, key, value)
    
    def update_section(self, section_name: str, **kwargs) -> bool:
        """Update a configuration section with new values."""
        with self._lock:
            try:
                section = getattr(self, section_name, None)
                if not section:
                    raise ConfigurationError(f"Invalid section: {section_name}")
                
                # Update values
                for key, value in kwargs.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        raise ConfigurationError(f"Invalid key {key} for section {section_name}")
                
                # Validate the updated section
                section.validate()
                
                # Notify watchers
                self._notify_watchers(section_name, kwargs)
                
                return True
                
            except (AttributeError, ConfigurationError) as e:
                logger.error(f"Failed to update section {section_name}: {e}")
                return False
    
    def add_watcher(self, callback: Callable) -> None:
        """Add a callback to be notified of config changes."""
        self._watchers.append(callback)
        
    def remove_watcher(self, callback: Callable) -> bool:
        """Remove a config change watcher."""
        try:
            self._watchers.remove(callback)
            return True
        except ValueError:
            return False
    
    def _notify_watchers(self, section: str, changes: Dict[str, Any]) -> None:
        """Notify all watchers of configuration changes."""
        for watcher in self._watchers:
            try:
                watcher(section, changes)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
    
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
    
    def update_window_geometry(self, geometry: Dict[str, int]) -> None:
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
# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get the global configuration manager (singleton pattern)."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def save_config() -> None:
    """Save the current configuration."""
    get_config().save_config()





# Example usage
if __name__ == "__main__":
    # Demo configuration management
    config = ConfigManager()
    
    print("System Info:")
    import json
    print(json.dumps(config.get_system_info(), indent=2))
    
    print("\nUsing maximum performance configuration...")
    
    print("\nCamera Config (Max Performance):")
    print(json.dumps(config.get_camera_config(), indent=2))
    
    print("\nHDF5 Config (Max Performance):")
    print(json.dumps(config.get_hdf5_config(), indent=2))
    
    config.save_config()
    print(f"\nConfiguration saved to: {config.config_file}")