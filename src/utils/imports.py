"""
DEPRECATED: This module will be removed in the next version.

Centralized imports are an anti-pattern that can cause circular dependencies.
Import dependencies directly in each module instead.

Kept temporarily for backwards compatibility.
"""

# Common type hints
from typing import Dict, Any

ConfigDict = Dict[str, Any]
SettingsDict = Dict[str, Any]
MetadataDict = Dict[str, Any]