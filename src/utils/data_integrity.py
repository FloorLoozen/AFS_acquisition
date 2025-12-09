"""Data integrity verification utilities.

Provides checksums, validation, and audit trail functionality to ensure
data integrity for HDF5 files and critical operations.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading

import h5py
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("data_integrity")


class DataIntegrityError(Exception):
    """Exception raised when data integrity checks fail."""
    pass


def compute_file_checksum(file_path: Path, algorithm: str = 'sha256') -> str:
    """Compute checksum for a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha256', 'sha512')
    
    Returns:
        Hex digest of the file checksum
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def compute_dataset_checksum(dataset: h5py.Dataset, sample_rate: int = 100) -> str:
    """Compute checksum for HDF5 dataset.
    
    For large datasets, samples every Nth frame to balance speed and coverage.
    
    Args:
        dataset: HDF5 dataset to checksum
        sample_rate: Sample every Nth frame (1 = all frames, 100 = every 100th)
    
    Returns:
        SHA256 checksum of sampled data
    """
    hash_func = hashlib.sha256()
    
    n_frames = dataset.shape[0]
    indices = range(0, n_frames, sample_rate)
    
    for i in indices:
        frame_data = dataset[i]
        hash_func.update(frame_data.tobytes())
    
    # Include metadata in checksum
    metadata = f"{dataset.shape}_{dataset.dtype}_{sample_rate}"
    hash_func.update(metadata.encode())
    
    return hash_func.hexdigest()


def verify_hdf5_integrity(file_path: Path) -> Dict[str, Any]:
    """Verify integrity of HDF5 file.
    
    Checks:
    - File can be opened
    - Required datasets exist
    - Dataset shapes and dtypes are valid
    - Checksums match (if stored)
    
    Args:
        file_path: Path to HDF5 file
    
    Returns:
        Dictionary with verification results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'file_size_mb': 0.0,
        'datasets': {},
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Check file exists and get size
        if not file_path.exists():
            results['valid'] = False
            results['errors'].append("File does not exist")
            return results
        
        results['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
        
        # Open and verify HDF5 structure
        with h5py.File(file_path, 'r') as f:
            # Check for main video dataset
            if 'raw_data' in f and 'main_video' in f['raw_data']:
                video_ds = f['raw_data/main_video']
                
                dataset_info = {
                    'shape': video_ds.shape,
                    'dtype': str(video_ds.dtype),
                    'compression': video_ds.compression,
                    'compression_opts': video_ds.compression_opts
                }
                
                # Verify shape is valid
                if len(video_ds.shape) != 3:
                    results['errors'].append(f"Invalid video shape: {video_ds.shape}")
                    results['valid'] = False
                
                # Check for stored checksum
                if 'data_checksum' in video_ds.attrs:
                    stored_checksum = video_ds.attrs['data_checksum']
                    logger.info("Computing dataset checksum for verification...")
                    computed_checksum = compute_dataset_checksum(video_ds, sample_rate=100)
                    
                    if stored_checksum == computed_checksum:
                        dataset_info['checksum_verified'] = True
                    else:
                        results['errors'].append("Dataset checksum mismatch")
                        results['valid'] = False
                        dataset_info['checksum_verified'] = False
                else:
                    results['warnings'].append("No checksum stored for verification")
                
                results['datasets']['main_video'] = dataset_info
            else:
                results['errors'].append("Main video dataset not found")
                results['valid'] = False
            
            # Check for LUT dataset if present
            if 'lookup_table' in f:
                lut_info = {
                    'exists': True,
                    'shape': f['lookup_table'].shape if hasattr(f['lookup_table'], 'shape') else None
                }
                results['datasets']['lookup_table'] = lut_info
            
            # Verify metadata
            required_attrs = ['recording_start_time', 'fps', 'frame_shape']
            for attr in required_attrs:
                if attr not in f.attrs:
                    results['warnings'].append(f"Missing metadata: {attr}")
    
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Error reading file: {str(e)}")
        logger.error(f"Error verifying {file_path}: {e}", exc_info=True)
    
    return results


def add_integrity_metadata(hdf5_file: h5py.File, dataset_name: str = 'raw_data/main_video'):
    """Add integrity metadata to HDF5 file.
    
    Computes and stores checksums for verification.
    
    Args:
        hdf5_file: Open HDF5 file object
        dataset_name: Name of dataset to add checksum for
    """
    try:
        if dataset_name in hdf5_file:
            dataset = hdf5_file[dataset_name]
            logger.info(f"Computing checksum for {dataset_name}...")
            
            # Compute checksum (sample every 100th frame for speed)
            checksum = compute_dataset_checksum(dataset, sample_rate=100)
            dataset.attrs['data_checksum'] = checksum
            dataset.attrs['checksum_algorithm'] = 'sha256_sampled'
            dataset.attrs['checksum_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Added checksum: {checksum[:16]}...")
    
    except Exception as e:
        logger.error(f"Error adding integrity metadata: {e}", exc_info=True)


class AuditTrail:
    """Audit trail for tracking all operations on data files."""
    
    def __init__(self, file_path: Optional[Path] = None):
        """Initialize audit trail.
        
        Args:
            file_path: Optional path to store audit trail. If None, keeps in memory only.
        """
        self.file_path = file_path
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def log_event(self, event_type: str, description: str, metadata: Optional[Dict] = None):
        """Log an audit event.
        
        Args:
            event_type: Type of event (e.g., 'file_created', 'compression', 'data_modified')
            description: Human-readable description
            metadata: Optional additional metadata
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'description': description,
            'metadata': metadata or {}
        }
        
        with self._lock:
            self.events.append(event)
            logger.debug(f"Audit: {event_type} - {description}")
            
            # Optionally write to file
            if self.file_path:
                self._write_to_file(event)
    
    def _write_to_file(self, event: Dict):
        """Append event to audit file."""
        try:
            with open(self.file_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")
    
    def get_events(self, event_type: Optional[str] = None) -> List[Dict]:
        """Get audit events, optionally filtered by type."""
        with self._lock:
            if event_type:
                return [e for e in self.events if e['event_type'] == event_type]
            return self.events.copy()
    
    def save_to_hdf5(self, hdf5_file: h5py.File):
        """Save audit trail to HDF5 file as structured table under meta_data/audit_trail."""
        try:
            with self._lock:
                if not self.events:
                    logger.debug("No audit events to save")
                    return
                
                # Create or get meta_data group
                if 'meta_data' not in hdf5_file:
                    meta_group = hdf5_file.create_group('meta_data')
                else:
                    meta_group = hdf5_file['meta_data']
                
                # Remove old audit_trail if it exists
                if 'audit_trail' in meta_group:
                    del meta_group['audit_trail']
                
                # Create audit trail group
                audit_group = meta_group.create_group('audit_trail')
                audit_group.attrs['description'] = b'Audit trail of all operations'
                audit_group.attrs['event_count'] = len(self.events)
                
                # Define compound datatype for audit events
                import numpy as np
                event_dtype = np.dtype([
                    ('timestamp', 'S32'),      # ISO format timestamp
                    ('event_type', 'S50'),     # Event type
                    ('description', 'S200'),   # Event description
                ])
                
                # Convert events to structured array
                event_data = np.zeros(len(self.events), dtype=event_dtype)
                for i, event in enumerate(self.events):
                    event_data[i]['timestamp'] = event['timestamp'].encode('utf-8')
                    event_data[i]['event_type'] = event['event_type'].encode('utf-8')
                    event_data[i]['description'] = event['description'][:200].encode('utf-8')
                
                # Create dataset
                audit_group.create_dataset('events', data=event_data, compression='gzip', compression_opts=4)
                
                # Save metadata for each event as separate datasets (if they exist)
                for i, event in enumerate(self.events):
                    if event.get('metadata'):
                        meta_str = json.dumps(event['metadata'])
                        if len(meta_str) < 1000:  # Only save small metadata
                            audit_group.attrs[f'event_{i}_metadata'] = meta_str.encode('utf-8')
                
                logger.info(f"Saved {len(self.events)} audit events to HDF5 meta_data/audit_trail")
        
        except Exception as e:
            logger.error(f"Failed to save audit trail to HDF5: {e}", exc_info=True)
    
    def load_from_hdf5(self, hdf5_file: h5py.File):
        """Load audit trail from HDF5 file."""
        try:
            import numpy as np
            
            # Check new structured format (meta_data/audit_trail group)
            if 'meta_data' in hdf5_file and 'audit_trail' in hdf5_file['meta_data']:
                audit_obj = hdf5_file['meta_data/audit_trail']
                
                # New structured format (group with events dataset)
                if isinstance(audit_obj, h5py.Group) and 'events' in audit_obj:
                    event_data = audit_obj['events'][()]
                    events = []
                    for i, row in enumerate(event_data):
                        event = {
                            'timestamp': row['timestamp'].decode('utf-8'),
                            'event_type': row['event_type'].decode('utf-8'),
                            'description': row['description'].decode('utf-8'),
                            'metadata': {}
                        }
                        # Load metadata if exists
                        meta_key = f'event_{i}_metadata'
                        if meta_key in audit_obj.attrs:
                            try:
                                event['metadata'] = json.loads(audit_obj.attrs[meta_key].decode('utf-8'))
                            except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                                logger.debug(f"Could not decode metadata for event {i}: {e}")
                        events.append(event)
                    
                    with self._lock:
                        self.events = events
                        logger.info(f"Loaded {len(self.events)} audit events from HDF5 (structured format)")
                
                # Old JSON format (dataset)
                else:
                    audit_json = audit_obj[()]
                    if isinstance(audit_json, bytes):
                        audit_json = audit_json.decode('utf-8')
                    
                    with self._lock:
                        self.events = json.loads(audit_json)
                        logger.info(f"Loaded {len(self.events)} audit events from HDF5 (JSON format)")
            
            # Fallback to old location (root) for backwards compatibility
            elif 'audit_trail' in hdf5_file:
                audit_json = hdf5_file['audit_trail'][()]
                if isinstance(audit_json, bytes):
                    audit_json = audit_json.decode('utf-8')
                
                with self._lock:
                    self.events = json.loads(audit_json)
                    logger.info(f"Loaded {len(self.events)} audit events from HDF5 (old location)")
        
        except Exception as e:
            logger.error(f"Failed to load audit trail from HDF5: {e}", exc_info=True)
