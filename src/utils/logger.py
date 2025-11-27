"""Logger module for AFS Acquisition with smart verbosity filtering."""

import logging
import os
import sys
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)


class SmartFilter(logging.Filter):
    """Filter to reduce log noise from repetitive messages."""
    
    def __init__(self):
        super().__init__()
        self.last_messages = {}
        self.max_repeats = 2
        self.suppress_patterns = [
            'Loaded camera settings', 'Camera settings updated', 'Moving:', 
            'button pressed', 'Connection failed on attempt', 'DeviceManager: attempting',
            'reconnect failed', 'Fast connect', 'Available VISA', 'No VISA resources'
        ]
        self.throttle_patterns = {'Exposure set to': 5, 'Gain set to': 5}
        
    def filter(self, record):
        msg = record.getMessage()
        
        # Suppress common noise
        if any(p in msg for p in self.suppress_patterns):
            return False
        
        # Throttle frequent messages
        for pattern, freq in self.throttle_patterns.items():
            if pattern in msg:
                key = f"{record.name}:{pattern}"
                count = self.last_messages.get(key, 0) + 1
                self.last_messages[key] = count
                return count % freq == 1
        
        # Limit repeats
        key = f"{record.name}:{msg}"
        count = self.last_messages.get(key, 0) + 1
        self.last_messages[key] = count
        return count <= self.max_repeats


class ColoredFormatter(logging.Formatter):
    """Add colors to console log levels."""
    
    COLORS = {
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Style.BRIGHT + Fore.RED
    }
    
    def format(self, record):
        msg = super().format(record)
        if record.levelno in self.COLORS:
            return f"{self.COLORS[record.levelno]}{msg}{Style.RESET_ALL}"
        return msg


def get_logger(name=None):
    """Get a configured logger instance."""
    logger = logging.getLogger(name or "main")
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(ColoredFormatter('[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s', '%H:%M:%S'))
    console.addFilter(SmartFilter())
    logger.addHandler(console)
    
    # File handler
    try:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"afs_{datetime.now():%Y%m%d_%H%M%S}.log")
        
        file = logging.FileHandler(log_file)
        file.setLevel(logging.DEBUG)
        file.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s', '%H:%M:%S'))
        logger.addHandler(file)
    except Exception:
        pass  # File logging optional
    
    return logger
