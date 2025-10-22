"""Custom exception classes for the AFS Tracking System.

Provides specific exception types for different error conditions,
enabling more precise error handling and better user feedback.
"""

from typing import Optional, Any


class AFSTrackingError(Exception):
    """Base exception class for all AFS Tracking System errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 component: Optional[str] = None):
        """Initialize the base error.
        
        Args:
            message: Human-readable error description
            error_code: Optional error code for programmatic handling
            component: Name of the component that generated the error
        """
        self.error_code = error_code
        self.component = component
        
        if component and error_code:
            full_message = f"[{component}:{error_code}] {message}"
        elif component:
            full_message = f"[{component}] {message}"
        elif error_code:
            full_message = f"[{error_code}] {message}"
        else:
            full_message = message
            
        super().__init__(full_message)


class HardwareError(AFSTrackingError):
    """Exception for hardware-related errors."""
    
    def __init__(self, message: str, hardware_type: str, 
                 error_code: Optional[str] = None):
        """Initialize hardware error.
        
        Args:
            message: Error description
            hardware_type: Type of hardware (e.g., 'camera', 'stage', 'function_generator')
            error_code: Optional hardware-specific error code
        """
        self.hardware_type = hardware_type
        super().__init__(message, error_code, f"hardware.{hardware_type}")


class CameraError(HardwareError):
    """Exception for camera-specific errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, "camera", error_code)


class StageError(HardwareError):
    """Exception for XY stage-specific errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, "stage", error_code)


class FunctionGeneratorError(HardwareError):
    """Exception for function generator-specific errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, "function_generator", error_code)


class OscilloscopeError(HardwareError):
    """Exception for oscilloscope-specific errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, "oscilloscope", error_code)


class RecordingError(AFSTrackingError):
    """Exception for video recording errors."""
    
    def __init__(self, message: str, frame_number: Optional[int] = None,
                 error_code: Optional[str] = None):
        """Initialize recording error.
        
        Args:
            message: Error description
            frame_number: Frame number where error occurred (if applicable)
            error_code: Optional recording-specific error code
        """
        self.frame_number = frame_number
        
        if frame_number is not None:
            message = f"Frame {frame_number}: {message}"
            
        super().__init__(message, error_code, "recording")


class FileSystemError(AFSTrackingError):
    """Exception for file system and I/O errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 operation: Optional[str] = None, error_code: Optional[str] = None):
        """Initialize file system error.
        
        Args:
            message: Error description
            file_path: Path of the file that caused the error
            operation: File operation that failed (e.g., 'read', 'write', 'create')
            error_code: Optional file system error code
        """
        self.file_path = file_path
        self.operation = operation
        
        if file_path and operation:
            message = f"Failed to {operation} '{file_path}': {message}"
        elif file_path:
            message = f"File '{file_path}': {message}"
        elif operation:
            message = f"File {operation} error: {message}"
            
        super().__init__(message, error_code, "filesystem")


class ConfigurationError(AFSTrackingError):
    """Exception for configuration and settings errors."""
    
    def __init__(self, message: str, setting_name: Optional[str] = None,
                 setting_value: Optional[Any] = None, error_code: Optional[str] = None):
        """Initialize configuration error.
        
        Args:
            message: Error description
            setting_name: Name of the setting that caused the error
            setting_value: Invalid value that was provided
            error_code: Optional configuration error code
        """
        self.setting_name = setting_name
        self.setting_value = setting_value
        
        if setting_name and setting_value is not None:
            message = f"Invalid setting '{setting_name}' = {setting_value}: {message}"
        elif setting_name:
            message = f"Setting '{setting_name}': {message}"
            
        super().__init__(message, error_code, "configuration")


class MeasurementError(AFSTrackingError):
    """Exception for measurement and data acquisition errors."""
    
    def __init__(self, message: str, measurement_type: Optional[str] = None,
                 error_code: Optional[str] = None):
        """Initialize measurement error.
        
        Args:
            message: Error description
            measurement_type: Type of measurement that failed
            error_code: Optional measurement-specific error code
        """
        self.measurement_type = measurement_type
        
        if measurement_type:
            message = f"Measurement '{measurement_type}': {message}"
            
        super().__init__(message, error_code, "measurement")


class ValidationError(AFSTrackingError):
    """Exception for input validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None,
                 field_value: Optional[Any] = None, error_code: Optional[str] = None):
        """Initialize validation error.
        
        Args:
            message: Error description
            field_name: Name of the field that failed validation
            field_value: Invalid value that was provided
            error_code: Optional validation error code
        """
        self.field_name = field_name
        self.field_value = field_value
        
        if field_name and field_value is not None:
            message = f"Invalid value for '{field_name}' = {field_value}: {message}"
        elif field_name:
            message = f"Field '{field_name}': {message}"
            
        super().__init__(message, error_code, "validation")