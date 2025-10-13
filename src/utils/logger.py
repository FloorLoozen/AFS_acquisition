"""
Logger module for the AFS Tracking System.
Provides consistent logging capabilities across the application.
"""

import logging
import os
import datetime
import sys
import time
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama for Windows terminal color support
init(autoreset=True)


# Color definitions using colorama for cross-platform support
class Colors:
    # Only using colors for warnings and errors as per user preference
    WHITE = ""  # Default terminal color
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    BOLD_RED = Style.BRIGHT + Fore.RED
    
    # Reset color
    RESET = Style.RESET_ALL


class SmartVerbosityFilter(logging.Filter):
    """
    Smart filter that reduces verbosity while keeping important messages.
    Suppresses repetitive messages and reduces noise from frequent operations.
    """
    
    def __init__(self, name=''):
        super().__init__(name)
        self.last_log = {}
        self.repeat_count = {}
        self.max_repeats = 2  # Only show first 2 instances of repeated messages
        
        # Patterns for messages to suppress or reduce frequency
        self.suppress_patterns = [
            'Loaded camera settings',
            'Camera settings updated from dialog', 
            'Applied default brighter camera settings',
            'Moving:',  # Stage movement details
            'button pressed',  # Button press details
        ]
        
        # Messages to show only occasionally (every Nth occurrence)  
        self.throttle_patterns = {
            'Exposure set to': 5,  # Show every 5th exposure change
            'Gain set to': 5,     # Show every 5th gain change
        }
        
    def filter(self, record):
        message = record.getMessage()
        
        # Check if message should be suppressed based on patterns
        for pattern in self.suppress_patterns:
            if pattern in message:
                return False  # Suppress these messages completely
                
        # Check for throttled messages (show only every Nth occurrence)
        for pattern, frequency in self.throttle_patterns.items():
            if pattern in message:
                msg_key = f"{record.name}:{pattern}"
                self.repeat_count[msg_key] = self.repeat_count.get(msg_key, 0) + 1
                return self.repeat_count[msg_key] % frequency == 1  # Show 1st, 6th, 11th, etc.
        
        # Handle general repetitive messages
        msg_key = f"{record.name}:{record.levelno}:{message}"
        
        if msg_key in self.last_log:
            self.repeat_count[msg_key] += 1
            
            # Only let through the message if it's below our repeat threshold
            if self.repeat_count[msg_key] <= self.max_repeats:
                return True
                
            # Suppress further repeats
            return False
        else:
            # New message, record it
            self.last_log[msg_key] = message
            self.repeat_count[msg_key] = 1
            return True


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log levels in console output.
    Color scheme:
    - DEBUG: Blue
    - INFO: Default terminal color (white)
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bold Red
    """
    
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Colors.WHITE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD_RED
    }
    
    def format(self, record):
        # First, format the message using the parent class
        formatted_message = super().format(record)
        
        # Apply color based on log level
        if record.levelno == logging.DEBUG:
            color_code = self.COLORS[record.levelno]
            return f"{color_code}{formatted_message}{Colors.RESET}"
        elif record.levelno >= logging.WARNING and record.levelno in self.COLORS:
            color_code = self.COLORS[record.levelno]
            return f"{color_code}{formatted_message}{Colors.RESET}"
        
        # For INFO, use plain white (default terminal color)
        return formatted_message


class AFSLogger:
    """
    A custom logger for the AFS Tracking System that provides consistent formatting
    and log file management.
    """
    
    # Log levels with their corresponding logging method
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, name="main", log_dir=None, console_level='INFO', file_level='DEBUG'):
        """
        Initialize the logger with the specified name and levels.
        
        Args:
            name (str): Logger name
            log_dir (str): Directory to store log files. If None, logs will be stored in 'logs' directory
            console_level (str): Minimum level for console output ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            file_level (str): Minimum level for file output ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers = []
        
        # Create logs directory if it doesn't exist
        if log_dir is None:
            base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
            log_dir = os.path.join(base_dir, 'logs')
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"afs_tracking_{timestamp}.log")
        
        # Create console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.LEVELS.get(console_level.upper(), logging.INFO))
        
        # Add smart verbosity filter to reduce log noise
        verbosity_filter = SmartVerbosityFilter('smart_verbosity')
        console_handler.addFilter(verbosity_filter)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.LEVELS.get(file_level.upper(), logging.DEBUG))
        
        # Create formatters
        # Format: [HH:MM:SS] [LEVEL] [MODULE] message
        base_format = '[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s'
        date_format = '%H:%M:%S'
        
        # Plain formatter for file
        file_formatter = logging.Formatter(base_format, datefmt=date_format)
        
        # Colored formatter for console
        console_formatter = ColoredFormatter(base_format, datefmt=date_format)
        
        # Apply formatters
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Store log file name
        self.log_file = os.path.basename(log_file)
        
        # Log initialization (can be silenced by overriding this method)
        self._log_initialization()
    
    def _log_initialization(self):
        """Log the initialization message - can be overridden for silent initialization"""
        self.logger.info(f"Started logging to {self.log_file}")
        
    def get_logger(self):
        """Get the configured logger instance."""
        return self.logger


# Create a default logger instance if it doesn't already exist
default_logger = None

def _get_default_logger():
    global default_logger
    if default_logger is None:
        # Create a subclass of AFSLogger that overrides the initialization log method
        class SilentAFSLogger(AFSLogger):
            def _log_initialization(self):
                # Don't log initialization message for default logger
                pass
                
        # Create silent logger instance and get the logger
        default_logger = SilentAFSLogger().get_logger()
    return default_logger


def get_logger(name=None):
    """
    Get a logger with the specified name.
    If name is None, return the default logger.
    
    Args:
        name (str): Optional name for the logger
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if name is None:
        return _get_default_logger()
        
    # Extract just the module name from the full path if __name__ was passed
    if '.' in name:
        name = name.split('.')[-1]
    
    # Create a new logger with proper handlers if it doesn't exist    
    logger = logging.getLogger(name)
    
    # Only configure the logger if it doesn't have any handlers yet
    if not logger.handlers:
        # Create a silent AFSLogger for this named logger
        class SilentAFSLogger(AFSLogger):
            def _log_initialization(self):
                # Don't log initialization message
                pass
                
        # Configure the logger silently
        SilentAFSLogger(name=name)
    
    return logger


# Convenience functions for logging
def debug(message, logger=None):
    """Log a debug message."""
    (logger or _get_default_logger()).debug(message)


def info(message, logger=None):
    """Log an info message."""
    (logger or _get_default_logger()).info(message)


def warning(message, logger=None):
    """Log a warning message."""
    (logger or _get_default_logger()).warning(message)


def error(message, logger=None):
    """Log an error message."""
    (logger or _get_default_logger()).error(message)


def critical(message, logger=None):
    """Log a critical message."""
    (logger or _get_default_logger()).critical(message)


# Example usage
if __name__ == "__main__":
    # Example of how to use this logger
    debug("This is a debug message")
    info("This is an info message")
    warning("This is a warning message")
    error("This is an error message")
    critical("This is a critical message")
    
    # Example of using a custom logger
    custom_logger = AFSLogger(name="CustomModule").get_logger()
    custom_logger.info("This is logged from a custom logger")
