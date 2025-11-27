"""Input validation utilities for AFS Acquisition."""

import os
from typing import Union, Tuple

from src.utils.exceptions import ValidationError


def validate_positive_number(value: Union[int, float], field_name: str = "value") -> Union[int, float]:
    """Validate that a number is positive."""
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValidationError("Must be a positive number", field_name, value)
    return value


def validate_range(value: Union[int, float], min_val: Union[int, float], 
                  max_val: Union[int, float], field_name: str = "value") -> Union[int, float]:
    """Validate that a number is within a specified range."""
    if not (min_val <= value <= max_val):
        raise ValidationError(f"Must be between {min_val} and {max_val}", field_name, value)
    return value


def validate_frame_shape(shape: Tuple[int, ...], field_name: str = "frame_shape") -> Tuple[int, int, int]:
    """Validate frame shape for video recording."""
    if len(shape) != 3 or not all(isinstance(d, int) and d > 0 for d in shape):
        raise ValidationError("Must be (height, width, channels) with positive integers", field_name, shape)
    if shape[2] not in {1, 3, 4}:
        raise ValidationError("Channels must be 1, 3, or 4", field_name, shape)
    return tuple(shape)


def validate_file_path(path: Union[str, os.PathLike], must_exist: bool = False, 
                      create_parent: bool = True, field_name: str = "file_path") -> str:
    """Validate and normalize a file path - returns string path for Windows."""
    path_str = os.fspath(path) if hasattr(os, 'fspath') else str(path)
    
    if must_exist and not os.path.exists(path_str):
        raise ValidationError("File does not exist", field_name, path)
    
    if create_parent and not must_exist:
        parent = os.path.dirname(path_str)
        if parent:
            os.makedirs(parent, exist_ok=True)
    
    return path_str