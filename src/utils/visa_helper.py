"""
VISA Helper Utilities

Centralized VISA resource management to avoid duplicate code patterns
across instrument controllers.
"""
from typing import Optional, List
import pyvisa

from src.utils.logger import get_logger

logger = get_logger("visa_helper")


class VISAHelper:
    """Helper class for common VISA operations."""
    
    _resource_manager: Optional[pyvisa.ResourceManager] = None
    
    @classmethod
    def get_resource_manager(cls) -> pyvisa.ResourceManager:
        """Get or create a shared VISA ResourceManager instance.
        
        Returns:
            pyvisa.ResourceManager: Shared resource manager instance
        """
        if cls._resource_manager is None:
            try:
                cls._resource_manager = pyvisa.ResourceManager()
                logger.debug("Created VISA ResourceManager")
            except Exception as e:
                logger.error(f"Failed to create VISA ResourceManager: {e}")
                raise
        return cls._resource_manager
    
    @classmethod
    def list_resources(cls, query: str = '?*') -> List[str]:
        """List available VISA resources.
        
        Args:
            query: VISA query string (default: all resources)
            
        Returns:
            List of resource strings
        """
        try:
            rm = cls.get_resource_manager()
            resources = rm.list_resources(query)
            return list(resources) if resources else []
        except Exception as e:
            logger.error(f"Failed to list VISA resources: {e}")
            return []
    
    @classmethod
    def open_resource(cls, resource_string: str, **kwargs) -> Optional[pyvisa.Resource]:
        """Open a VISA resource with common error handling.
        
        Args:
            resource_string: VISA resource identifier
            **kwargs: Additional arguments for open_resource
            
        Returns:
            VISA resource or None if failed
        """
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
        """Close the resource manager and all resources."""
        if cls._resource_manager is not None:
            try:
                cls._resource_manager.close()
                cls._resource_manager = None
                logger.debug("Closed VISA ResourceManager")
            except Exception as e:
                logger.warning(f"Error closing VISA ResourceManager: {e}")
