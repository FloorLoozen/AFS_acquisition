"""Input validation utilities for the AFS Tracking System.

Provides consistent validation functions for common input types used
throughout the application. Helps ensure data integrity and provides
clear error messages for invalid inputs.
"""

import os
from pathlib import Path
from typing import Union, Tuple, Any, Optional
from src.utils.exceptions import ValidationError


def validate_positive_number(value: Union[int, float], field_name: str = "value") -> Union[int, float]:
    """Validate that a number is positive.
    
    Args:
        value: The number to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise ValidationError("Must be a number", field_name, value)
    
    if value <= 0:
        raise ValidationError("Must be positive", field_name, value)
    
    return value


def validate_frame_shape(shape: Tuple[int, ...], field_name: str = "frame_shape") -> Tuple[int, int, int]:
    """Validate frame shape for video recording.
    
    Args:
        shape: Tuple representing (height, width, channels)
        field_name: Name of the field for error messages
        
    Returns:
        The validated shape tuple
        
    Raises:
        ValidationError: If shape is invalid for video frames
    """
    if not isinstance(shape, (tuple, list)):
        raise ValidationError("Must be a tuple or list", field_name, shape)
    
    if len(shape) != 3:
        raise ValidationError("Must have exactly 3 dimensions (height, width, channels)", field_name, shape)
    
    if not all(isinstance(dim, int) for dim in shape):
        raise ValidationError("All dimensions must be integers", field_name, shape)
    
    if any(dim <= 0 for dim in shape):
        raise ValidationError("All dimensions must be positive", field_name, shape)
    
    height, width, channels = shape
    
    # Reasonable bounds checking
    if height > 10000 or width > 10000:
        raise ValidationError("Frame dimensions too large (max 10000x10000)", field_name, shape)
    
    if channels not in [1, 3, 4]:
        raise ValidationError("Channels must be 1 (grayscale), 3 (RGB), or 4 (RGBA)", field_name, shape)
    
    return tuple(shape)


def validate_file_path(path: Union[str, Path], must_exist: bool = False, 
                      create_parent: bool = True, field_name: str = "file_path") -> Path:
    """Validate and normalize a file path.
    
    Args:
        path: The file path to validate
        must_exist: Whether the file must already exist
        create_parent: Whether to create parent directories if they don't exist
        field_name: Name of the field for error messages
        
    Returns:
        Normalized Path object
        
    Raises:
        ValidationError: If path is invalid or file operations fail
    """
    if not isinstance(path, (str, Path)):
        raise ValidationError("Must be a string or Path object", field_name, path)
    
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise ValidationError(f"File does not exist: {path_obj}", field_name, path)
    
    if create_parent and not must_exist:
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValidationError(f"Cannot create parent directory: {e}", field_name, path)
    
    # Check if parent directory is writable (for new files)
    if not must_exist and path_obj.parent.exists():
        if not os.access(path_obj.parent, os.W_OK):
            raise ValidationError(f"Parent directory is not writable: {path_obj.parent}", field_name, path)
    
    return path_obj


def validate_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float], 
                  field_name: str = "value") -> Union[int, float]:
    """Validate that a number is within a specified range.
    
    Args:
        value: The number to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        field_name: Name of the field for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is outside the allowed range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"Must be a number", field_name, value)
    
    if value < min_val or value > max_val:
        raise ValidationError(f"Must be between {min_val} and {max_val}", field_name, value)
    
    return value


def validate_string_not_empty(value: str, field_name: str = "value") -> str:
    """Validate that a string is not empty or just whitespace.
    
    Args:
        value: The string to validate
        field_name: Name of the field for error messages
        
    Returns:
        The stripped string
        
    Raises:
        ValidationError: If string is empty or just whitespace
    """
    if not isinstance(value, str):
        raise ValidationError("Must be a string", field_name, value)
    
    stripped = value.strip()
    if not stripped:
        raise ValidationError("Cannot be empty or just whitespace", field_name, value)
    
    return stripped