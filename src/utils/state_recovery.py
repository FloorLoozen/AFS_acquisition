"""State recovery system for crash recovery and session persistence.

Automatically saves application state and allows recovery after crashes.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import threading

from src.utils.logger import get_logger

logger = get_logger("state_recovery")


class StateRecovery:
    """Manages application state saving and recovery."""
    
    def __init__(self, state_file: Path):
        """Initialize state recovery manager.
        
        Args:
            state_file: Path to state recovery file
        """
        self.state_file = state_file
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}
        self._last_save_time = 0.0
        self._save_interval = 5.0  # Save every 5 seconds max
        
        # Try to load existing state
        self._load_state()
    
    def _load_state(self):
        """Load state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    self._state = json.load(f)
                
                # Check if state is recent (within last hour)
                if 'timestamp' in self._state:
                    state_age = time.time() - self._state['timestamp']
                    if state_age < 3600:
                        logger.info(f"Loaded recovery state from {state_age:.0f}s ago")
                        return
                
                logger.info("Recovery state is stale, starting fresh")
                self._state = {}
        
        except Exception as e:
            logger.warning(f"Could not load recovery state: {e}")
            self._state = {}
    
    def save_state(self, key: str, value: Any, force: bool = False):
        """Save a state value.
        
        Args:
            key: State key
            value: Value to save (must be JSON-serializable)
            force: Force immediate save, ignore interval
        """
        with self._lock:
            self._state[key] = value
            self._state['timestamp'] = time.time()
            
            # Rate-limit saves unless forced
            current_time = time.time()
            if force or (current_time - self._last_save_time) > self._save_interval:
                self._write_to_file()
                self._last_save_time = current_time
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value.
        
        Args:
            key: State key
            default: Default value if key not found
        
        Returns:
            State value or default
        """
        with self._lock:
            return self._state.get(key, default)
    
    def has_recovery_state(self) -> bool:
        """Check if there is valid recovery state available."""
        with self._lock:
            return bool(self._state and 'timestamp' in self._state)
    
    def get_recovery_info(self) -> Dict[str, Any]:
        """Get information about recoverable state.
        
        Returns:
            Dictionary with recovery information
        """
        with self._lock:
            if not self._state or 'timestamp' not in self._state:
                return {'available': False}
            
            state_time = self._state['timestamp']
            state_age = time.time() - state_time
            
            return {
                'available': True,
                'timestamp': datetime.fromtimestamp(state_time).isoformat(),
                'age_seconds': state_age,
                'recording_active': self._state.get('recording_active', False),
                'file_path': self._state.get('recording_file_path'),
                'frame_count': self._state.get('frame_count', 0),
                'session_duration': self._state.get('session_duration', 0.0)
            }
    
    def clear_state(self):
        """Clear all saved state."""
        with self._lock:
            self._state = {}
            self._write_to_file()
            logger.info("Recovery state cleared")
    
    def _write_to_file(self):
        """Write current state to file atomically."""
        try:
            # Write to temp file first, then rename (atomic operation)
            base, _ = os.path.splitext(self.state_file)
            temp_file = base + '.tmp'
            
            with open(temp_file, 'w') as f:
                json.dump(self._state, f, indent=2)
            
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to write recovery state: {e}")
    
    def create_session_snapshot(self, recording_active: bool, file_path: Optional[Path],
                               frame_count: int, duration: float, 
                               additional_data: Optional[Dict] = None):
        """Create a complete snapshot of current recording session.
        
        Args:
            recording_active: Whether recording is currently active
            file_path: Path to recording file
            frame_count: Number of frames recorded
            duration: Session duration in seconds
            additional_data: Optional additional session data
        """
        snapshot = {
            'recording_active': recording_active,
            'recording_file_path': str(file_path) if file_path else None,
            'frame_count': frame_count,
            'session_duration': duration,
            'snapshot_time': datetime.now().isoformat()
        }
        
        if additional_data:
            snapshot.update(additional_data)
        
        with self._lock:
            self._state.update(snapshot)
            self._state['timestamp'] = time.time()
            self._write_to_file()
