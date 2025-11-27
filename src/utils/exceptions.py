"""Custom exception classes for AFS Acquisition."""


class AFSError(Exception):
    """Base exception for all AFS Acquisition errors."""
    pass


class HardwareError(AFSError):
    """Hardware communication or control errors."""
    pass


class RecordingError(AFSError):
    """Video recording and data capture errors."""
    pass


class ValidationError(AFSError):
    """Input validation errors."""
    pass


class ConfigurationError(AFSError):
    """Configuration and settings errors."""
    pass


# Backwards compatibility aliases
AFSTrackingError = AFSError
CameraError = HardwareError
StageError = HardwareError
FunctionGeneratorError = HardwareError
OscilloscopeError = HardwareError
FileSystemError = AFSError
MeasurementError = AFSError