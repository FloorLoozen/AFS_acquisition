"""
System Verification Script for AFS Acquisition
Tests all major components and reports status
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all critical imports work."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    tests = {
        "PyQt5": lambda: __import__('PyQt5.QtWidgets'),
        "NumPy": lambda: __import__('numpy'),
        "OpenCV": lambda: __import__('cv2'),
        "H5PY": lambda: __import__('h5py'),
        "PyVISA": lambda: __import__('pyvisa'),
        "Matplotlib": lambda: __import__('matplotlib'),
    }
    
    results = {}
    for name, test_func in tests.items():
        try:
            test_func()
            results[name] = "✓ OK"
        except ImportError as e:
            results[name] = f"✗ MISSING: {e}"
    
    for name, result in results.items():
        print(f"  {name:20s} {result}")
    
    return all("✓" in r for r in results.values())


def test_gpu_support():
    """Test GPU acceleration availability."""
    print("\n" + "=" * 60)
    print("TESTING GPU ACCELERATION")
    print("=" * 60)
    
    try:
        import cv2
        
        # Check OpenCL
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                print("  ✓ OpenCL GPU acceleration: AVAILABLE")
                
                # Get device info
                try:
                    device = cv2.ocl.Device.getDefault()
                    print(f"    Device: {device.name()}")
                    print(f"    Type: {device.type()}")
                    return True
                except Exception as e:
                    print(f"    (Could not get device info: {e})")
                    return True
            else:
                print("  ✗ OpenCL detected but failed to enable")
                return False
        else:
            print("  ✗ No OpenCL support detected")
            return False
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
        return False


def test_hdf5_compression():
    """Test HDF5 compression functionality."""
    print("\n" + "=" * 60)
    print("TESTING HDF5 COMPRESSION")
    print("=" * 60)
    
    try:
        import h5py
        import numpy as np
        import tempfile
        import os
        
        # Create test file
        test_file = tempfile.mktemp(suffix='.hdf5')
        
        # Write test data
        with h5py.File(test_file, 'w') as f:
            # Create dataset with GZIP compression
            data = np.random.randint(0, 256, (100, 100, 100), dtype=np.uint8)
            ds = f.create_dataset(
                'test',
                data=data,
                compression='gzip',
                compression_opts=9,
                shuffle=True
            )
            
            print(f"  ✓ Created test dataset: {ds.shape}")
            print(f"  ✓ Compression: {ds.compression}")
            print(f"  ✓ Compression level: {ds.compression_opts}")
        
        # Read back and verify
        with h5py.File(test_file, 'r') as f:
            data_read = f['test'][:]
            if np.array_equal(data, data_read):
                print("  ✓ Data integrity: VERIFIED (lossless)")
            else:
                print("  ✗ Data integrity: FAILED")
                return False
        
        # Check file size
        file_size = os.path.getsize(test_file) / 1024  # KB
        print(f"  ✓ Compressed file size: {file_size:.1f} KB")
        
        # Cleanup
        os.remove(test_file)
        print("  ✓ HDF5 compression: WORKING")
        return True
        
    except Exception as e:
        print(f"  ✗ HDF5 test failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from src.utils.config_manager import get_config
        
        config = get_config()
        print(f"  ✓ Config loaded successfully")
        print(f"  ✓ Camera queue size: {config.performance.camera_queue_size}")
        print(f"  ✓ Background compression: {config.files.background_compression}")
        print(f"  ✓ Default save path: {config.files.default_save_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def test_error_handling():
    """Test improved error handling."""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)
    
    try:
        # Check that we can import without bare except clauses
        from src.utils import hdf5_video_recorder
        from src.ui import camera_widget
        from src.controllers import xy_stage_controller
        
        # Read the source to verify no bare except
        import inspect
        
        modules_to_check = [
            ('hdf5_video_recorder', hdf5_video_recorder),
            ('camera_widget', camera_widget),
            ('xy_stage_controller', xy_stage_controller),
        ]
        
        bare_except_found = False
        for name, module in modules_to_check:
            source = inspect.getsource(module)
            if '\n    except:\n' in source or '\n        except:\n' in source:
                print(f"  ✗ {name}: Still has bare except clauses")
                bare_except_found = True
            else:
                print(f"  ✓ {name}: No bare except clauses")
        
        if not bare_except_found:
            print("  ✓ Error handling: IMPROVED")
            return True
        else:
            print("  ⚠ Some bare except clauses remain")
            return False
            
    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("*" * 60)
    print(" AFS ACQUISITION SYSTEM VERIFICATION")
    print("*" * 60)
    print()
    
    results = {
        "Imports": test_imports(),
        "GPU Support": test_gpu_support(),
        "HDF5 Compression": test_hdf5_compression(),
        "Configuration": test_configuration(),
        "Error Handling": test_error_handling(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test:25s} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("  System is ready for use!")
    else:
        print("  ⚠ SOME TESTS FAILED")
        print("  Check errors above for details")
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
