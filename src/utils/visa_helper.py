"""VISA Helper Utilities for instrument controllers."""

import pyvisa

from src.utils.logger import get_logger

logger = get_logger("visa_helper")


class VISAHelper:
    """Helper class for common VISA operations."""
    
    _resource_manager = None
    
    @classmethod
    def get_resource_manager(cls):
        if cls._resource_manager is None:
            cls._resource_manager = pyvisa.ResourceManager()
            logger.debug("Created VISA ResourceManager")
        return cls._resource_manager
    
    @classmethod
    def list_resources(cls, query='?*'):
        try:
            rm = cls.get_resource_manager()
            resources = rm.list_resources(query)
            return list(resources) if resources else []
        except Exception as e:
            logger.error(f"Failed to list VISA resources: {e}")
            return []
    
    @classmethod
    def open_resource(cls, resource_string, **kwargs):
        try:
            rm = cls.get_resource_manager()
            resource = rm.open_resource(resource_string, **kwargs)
            logger.debug(f"Opened VISA resource: {resource_string}")
            return resource
        except Exception as e:
            logger.error(f"Failed to open VISA resource {resource_string}: {e}")
            return None
    
    @classmethod
    def close_all(cls):
        if cls._resource_manager is not None:
            try:
                cls._resource_manager.close()
                cls._resource_manager = None
                logger.debug("Closed VISA ResourceManager")
            except Exception as e:
                logger.warning(f"Error closing VISA ResourceManager: {e}")
