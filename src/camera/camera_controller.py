"""Camera controller for handling camera operations and settings."""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from src.utils.logger import get_logger

logger = get_logger("camera")

class CameraController:
    """Controls camera operations and settings."""
    
    def __init__(self):
        self.camera = None
        self.is_connected = False
        self._current_settings = {}
        self._available_settings = {}
        
    def connect(self, camera_index: int = 0) -> bool:
        """Connect to camera and detect available settings."""
        try:
            # Try different backends for better control
            backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.camera = cv2.VideoCapture(camera_index, backend)
                    if self.camera.isOpened():
                        logger.info(f"Connected using backend: {backend}")
                        break
                except:
                    continue
            
            if not self.camera or not self.camera.isOpened():
                return False
            
            # Disable auto-exposure and auto-white-balance first
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_connected = True
            self._detect_available_settings()
            self._apply_tracking_defaults()
            logger.info(f"Connected to camera {camera_index}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            return False
    
    def _detect_available_settings(self) -> None:
        """Detect which camera settings actually work."""
        if not self.camera:
            return
            
        # Focus on most useful settings for tracking
        priority_settings = {
            'exposure': (cv2.CAP_PROP_EXPOSURE, (-13, -1), 'float', 0.5),
            'gain': (cv2.CAP_PROP_GAIN, (0, 100), 'int', 10),
            'brightness': (cv2.CAP_PROP_BRIGHTNESS, (0, 255), 'int', 20),
            'contrast': (cv2.CAP_PROP_CONTRAST, (0, 255), 'int', 20),
            'saturation': (cv2.CAP_PROP_SATURATION, (0, 255), 'int', 20),
            'sharpness': (cv2.CAP_PROP_SHARPNESS, (0, 255), 'int', 20),
            'focus': (cv2.CAP_PROP_FOCUS, (0, 255), 'int', 20),
            'fps': (cv2.CAP_PROP_FPS, (5, 60), 'int', 5),
        }
        
        self._available_settings = {}
        
        for setting_name, (prop, (min_val, max_val), data_type, test_step) in priority_settings.items():
            if self._test_setting_works(setting_name, prop, min_val, max_val, data_type, test_step):
                self._available_settings[setting_name] = {
                    'property': prop,
                    'range': (min_val, max_val),
                    'type': data_type,
                    'step': 0.1 if data_type == 'float' else 1
                }
                current = self.camera.get(prop)
                self._current_settings[setting_name] = current
                logger.info(f"✓ {setting_name}: {current:.1f} (range: {min_val}-{max_val})")
        
        logger.info(f"Found {len(self._available_settings)} working settings")
    
    def _test_setting_works(self, name: str, prop: int, min_val: float, max_val: float, 
                           data_type: str, test_step: float) -> bool:
        """Test if a setting actually changes the camera."""
        try:
            # Get original value
            original = self.camera.get(prop)
            if original == -1:
                return False
            
            # Calculate test values
            mid_val = (min_val + max_val) / 2
            test_vals = [min_val, mid_val, max_val]
            
            results = []
            for test_val in test_vals:
                # Set value
                success = self.camera.set(prop, test_val)
                if not success:
                    continue
                    
                # Small delay for camera to process
                time.sleep(0.1)
                
                # Read back value
                actual = self.camera.get(prop)
                results.append(actual)
                
            # Restore original
            self.camera.set(prop, original)
            
            # Check if we got different values
            if len(set(results)) > 1:
                return True
                
            # For exposure, even small changes matter
            if name == 'exposure' and len(results) >= 2:
                return abs(results[0] - results[-1]) > 0.01
                
            return False
            
        except Exception as e:
            logger.debug(f"Test failed for {name}: {e}")
            return False
    
    def _apply_tracking_defaults(self) -> None:
        """Apply optimal settings for tracking."""
        # High exposure for better lighting
        if 'exposure' in self._available_settings:
            self.set_setting('exposure', -3)
            
        # Test gain at high value
        if 'gain' in self._available_settings:
            self.set_setting('gain', 80)
            
        # Good contrast for edge detection
        if 'contrast' in self._available_settings:
            self.set_setting('contrast', 160)
            
        # Moderate fps for smooth tracking
        if 'fps' in self._available_settings:
            self.set_setting('fps', 30)
    
    def set_setting(self, setting_name: str, value: Any) -> bool:
        """Set a camera setting and verify it worked."""
        if not self.camera or setting_name not in self._available_settings:
            logger.warning(f"Setting '{setting_name}' not available")
            return False
            
        try:
            setting_info = self._available_settings[setting_name]
            prop = setting_info['property']
            min_val, max_val = setting_info['range']
            
            # Clamp to valid range
            value = max(min_val, min(max_val, float(value)))
            
            # Get original for comparison
            original = self.camera.get(prop)
            
            # Set new value
            success = self.camera.set(prop, value)
            if not success:
                logger.warning(f"Failed to set {setting_name} property")
                return False
            
            # Allow time for camera to process
            time.sleep(0.05)
            
            # Verify the change
            actual = self.camera.get(prop)
            self._current_settings[setting_name] = actual
            
            # Check if change was significant
            tolerance = 0.01 if setting_name == 'exposure' else 1.0
            if abs(actual - original) > tolerance:
                logger.info(f"✓ Set {setting_name}: {original:.1f} → {actual:.1f}")
                return True
            else:
                logger.warning(f"✗ {setting_name} didn't change: {original:.1f} → {actual:.1f}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting {setting_name}: {e}")
            return False
    
    def apply_settings(self, settings: Dict[str, Any]) -> Dict[str, bool]:
        """Apply multiple settings and return results."""
        results = {}
        for setting_name, value in settings.items():
            results[setting_name] = self.set_setting(setting_name, value)
        return results
    
    def get_available_settings(self) -> Dict[str, Dict[str, Any]]:
        """Get available settings info."""
        return self._available_settings.copy()
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current values of all settings."""
        if not self.camera:
            return {}
            
        # Read fresh values from camera
        for setting_name, setting_info in self._available_settings.items():
            try:
                current_val = self.camera.get(setting_info['property'])
                if current_val != -1:
                    self._current_settings[setting_name] = current_val
            except:
                pass
                
        return self._current_settings.copy()
    
    def test_gain_specifically(self) -> None:
        """Test gain control thoroughly."""
        if 'gain' not in self._available_settings:
            print("❌ Gain not available")
            return
            
        print("🔧 Testing Gain Control...")
        original = self.camera.get(cv2.CAP_PROP_GAIN)
        
        test_values = [0, 25, 50, 75, 100]
        for test_val in test_values:
            self.camera.set(cv2.CAP_PROP_GAIN, test_val)
            time.sleep(0.1)
            actual = self.camera.get(cv2.CAP_PROP_GAIN)
            print(f"  Set {test_val} → Got {actual:.1f}")
            
        # Restore
        self.camera.set(cv2.CAP_PROP_GAIN, original)
        print(f"✓ Gain test complete, restored to {original:.1f}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get camera frame."""
        if not self.camera or not self.is_connected:
            return None
            
        ret, frame = self.camera.read()
        return frame if ret else None
    
    def disconnect(self) -> None:
        """Clean disconnect."""
        if self.camera:
            self.camera.release()
            self.camera = None
        self.is_connected = False
        self._current_settings.clear()
        self._available_settings.clear()
        logger.info("Camera disconnected")
    def disconnect(self) -> None:
        """Disconnect from camera."""
        if self.camera:
            self.camera.release()
            self.camera = None
        self.is_connected = False
        self._current_settings.clear()
        self._available_settings.clear()
        logger.info("Camera disconnected")
