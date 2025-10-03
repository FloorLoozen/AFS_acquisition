"""
HDF5 Compression Analyzer for AFS Tracking System.
Analyzes and compares compression performance for video recordings.
"""

import h5py
import numpy as np
import os
import time
from typing import Dict, List, Tuple
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger("compression_analyzer")


class CompressionAnalyzer:
    """Analyze and compare HDF5 compression methods for video data."""
    
    def __init__(self):
        self.compression_methods = ['lzf', 'gzip', None]
        
    def analyze_existing_file(self, file_path: str) -> Dict:
        """
        Analyze compression performance of an existing HDF5 file.
        
        Args:
            file_path: Path to the HDF5 file
            
        Returns:
            Dictionary with compression analysis results
        """
        try:
            with h5py.File(file_path, 'r') as f:
                if 'video' not in f:
                    return {'error': 'No video dataset found'}
                
                video_ds = f['video']
                
                # Basic file information
                file_size = os.path.getsize(file_path)
                uncompressed_size = np.prod(video_ds.shape) * video_ds.dtype.itemsize
                
                # Get compression info from attributes
                compression = getattr(video_ds, 'compression', None)
                compression_opts = getattr(video_ds, 'compression_opts', None)
                
                # Calculate compression ratio
                compression_ratio = uncompressed_size / file_size if file_size > 0 else 1.0
                space_saved_percent = (1 - 1/compression_ratio) * 100 if compression_ratio > 1 else 0
                
                analysis = {
                    'file_path': file_path,
                    'file_size_mb': file_size / (1024 * 1024),
                    'uncompressed_size_mb': uncompressed_size / (1024 * 1024),
                    'compression_method': compression,
                    'compression_options': compression_opts,
                    'compression_ratio': compression_ratio,
                    'space_saved_percent': space_saved_percent,
                    'video_shape': video_ds.shape,
                    'dtype': str(video_ds.dtype),
                    'chunks': video_ds.chunks,
                }
                
                # Add metadata if available
                if hasattr(video_ds, 'attrs'):
                    for key in ['fps', 'recording_duration_s', 'total_frames']:
                        if key in video_ds.attrs:
                            analysis[key] = video_ds.attrs[key]
                
                return analysis
                
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return {'error': str(e)}
    
    def compare_compression_methods(self, test_data: np.ndarray, 
                                   output_dir: str = "compression_test") -> Dict:
        """
        Compare different compression methods on test data.
        
        Args:
            test_data: 4D numpy array (frames, height, width, channels)
            output_dir: Directory to store test files
            
        Returns:
            Dictionary comparing compression methods
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for compression in self.compression_methods:
            try:
                # Generate test filename
                comp_name = compression or 'uncompressed'
                test_file = os.path.join(output_dir, f"test_{comp_name}.hdf5")
                
                # Create dataset with compression
                start_time = time.time()
                
                with h5py.File(test_file, 'w') as f:
                    dataset_kwargs = {
                        'data': test_data,
                        'chunks': (1, *test_data.shape[1:]),  # Frame-level chunks
                        'shuffle': True,  # Enable shuffle filter
                    }
                    
                    if compression:
                        dataset_kwargs['compression'] = compression
                        if compression == 'gzip':
                            dataset_kwargs['compression_opts'] = 9
                        elif compression == 'lzf':
                            dataset_kwargs['fletcher32'] = True
                    
                    f.create_dataset('video', **dataset_kwargs)
                
                write_time = time.time() - start_time
                
                # Analyze results
                file_size = os.path.getsize(test_file)
                uncompressed_size = test_data.nbytes
                
                results[comp_name] = {
                    'compression_method': compression,
                    'write_time_seconds': write_time,
                    'file_size_mb': file_size / (1024 * 1024),
                    'uncompressed_size_mb': uncompressed_size / (1024 * 1024),
                    'compression_ratio': uncompressed_size / file_size,
                    'space_saved_percent': (1 - file_size/uncompressed_size) * 100,
                    'write_speed_mbps': (uncompressed_size / (1024 * 1024)) / write_time,
                    'file_path': test_file
                }
                
                logger.info(f"Compression test '{comp_name}': "
                           f"{results[comp_name]['compression_ratio']:.2f}x compression, "
                           f"{results[comp_name]['write_speed_mbps']:.1f} MB/s")
                
            except Exception as e:
                logger.error(f"Error testing compression {compression}: {e}")
                results[comp_name] = {'error': str(e)}
        
        return results
    
    def generate_test_data(self, shape: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Generate realistic test video data.
        
        Args:
            shape: (frames, height, width, channels)
            
        Returns:
            Test video data as numpy array
        """
        frames, height, width, channels = shape
        
        # Create more realistic video-like data with spatial and temporal correlation
        test_data = np.zeros(shape, dtype=np.uint8)
        
        # Generate base pattern
        y, x = np.ogrid[:height, :width]
        base_pattern = np.sin(x * 0.02) * np.cos(y * 0.02) * 127 + 128
        
        for i in range(frames):
            # Add temporal variation
            time_factor = np.sin(i * 0.1) * 0.3 + 0.7
            noise = np.random.normal(0, 20, (height, width))
            
            # Create frame
            frame = np.clip(base_pattern * time_factor + noise, 0, 255).astype(np.uint8)
            
            # Repeat for all channels
            for c in range(channels):
                test_data[i, :, :, c] = frame * (0.8 + c * 0.1)  # Slight channel variation
        
        return test_data
    
    def benchmark_compression(self, frame_shape: Tuple[int, int, int], 
                            num_frames: int = 100) -> Dict:
        """
        Run a comprehensive compression benchmark.
        
        Args:
            frame_shape: Shape of individual frames (height, width, channels)
            num_frames: Number of frames to test
            
        Returns:
            Benchmark results
        """
        logger.info(f"Starting compression benchmark: {num_frames} frames of {frame_shape}")
        
        # Generate test data
        test_shape = (num_frames, *frame_shape)
        test_data = self.generate_test_data(test_shape)
        
        logger.info(f"Generated {test_data.nbytes / (1024*1024):.1f} MB of test data")
        
        # Compare compression methods
        results = self.compare_compression_methods(test_data)
        
        # Add summary statistics
        if len(results) > 1:
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if valid_results:
                best_compression = max(valid_results.items(), 
                                     key=lambda x: x[1]['compression_ratio'])
                fastest_write = max(valid_results.items(), 
                                  key=lambda x: x[1]['write_speed_mbps'])
                
                results['summary'] = {
                    'best_compression': {
                        'method': best_compression[0],
                        'ratio': best_compression[1]['compression_ratio'],
                        'space_saved_percent': best_compression[1]['space_saved_percent']
                    },
                    'fastest_write': {
                        'method': fastest_write[0],
                        'speed_mbps': fastest_write[1]['write_speed_mbps']
                    },
                    'recommendation': self._get_recommendation(valid_results)
                }
        
        return results
    
    def _get_recommendation(self, results: Dict) -> str:
        """Get compression method recommendation based on results."""
        if 'lzf' in results and 'gzip' in results:
            lzf = results['lzf']
            gzip = results['gzip']
            
            # If gzip provides significantly better compression (>20% smaller)
            # but write speed is still acceptable (>50 MB/s), recommend gzip
            if (gzip['compression_ratio'] / lzf['compression_ratio'] > 1.2 and 
                gzip['write_speed_mbps'] > 50):
                return "gzip - Better compression with acceptable speed"
            else:
                return "lzf - Best balance of compression and speed"
        elif 'lzf' in results:
            return "lzf - Good compression with fast writing"
        elif 'gzip' in results:
            return "gzip - Maximum compression"
        else:
            return "No compression data available"
    
    def print_analysis_report(self, analysis: Dict, title: str = "HDF5 Compression Analysis"):
        """Print a formatted analysis report."""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        if 'summary' in analysis:
            # Benchmark report
            print(f"\nSUMMARY:")
            print(f"  Best compression: {analysis['summary']['best_compression']['method']} "
                  f"({analysis['summary']['best_compression']['ratio']:.2f}x, "
                  f"{analysis['summary']['best_compression']['space_saved_percent']:.1f}% saved)")
            print(f"  Fastest write: {analysis['summary']['fastest_write']['method']} "
                  f"({analysis['summary']['fastest_write']['speed_mbps']:.1f} MB/s)")
            print(f"  Recommendation: {analysis['summary']['recommendation']}")
            
            print(f"\nDETAILED RESULTS:")
            for method, data in analysis.items():
                if method != 'summary' and 'error' not in data:
                    print(f"\n  {method.upper()}:")
                    print(f"    File size: {data['file_size_mb']:.1f} MB")
                    print(f"    Compression ratio: {data['compression_ratio']:.2f}x")
                    print(f"    Space saved: {data['space_saved_percent']:.1f}%")
                    print(f"    Write speed: {data['write_speed_mbps']:.1f} MB/s")
                    print(f"    Write time: {data['write_time_seconds']:.2f} seconds")
        else:
            # Single file analysis
            print(f"\nFile: {analysis.get('file_path', 'Unknown')}")
            print(f"Video shape: {analysis.get('video_shape', 'Unknown')}")
            print(f"Compression method: {analysis.get('compression_method', 'None')}")
            print(f"File size: {analysis.get('file_size_mb', 0):.1f} MB")
            print(f"Uncompressed size: {analysis.get('uncompressed_size_mb', 0):.1f} MB")
            print(f"Compression ratio: {analysis.get('compression_ratio', 1):.2f}x")
            print(f"Space saved: {analysis.get('space_saved_percent', 0):.1f}%")
            
            if 'fps' in analysis:
                print(f"Recording FPS: {analysis['fps']:.1f}")
            if 'recording_duration_s' in analysis:
                print(f"Duration: {analysis['recording_duration_s']:.1f} seconds")


def analyze_recording_files(directory: str = "C:/Users/fAFS/Documents/Floor/tmp"):
    """Analyze all HDF5 files in the recordings directory."""
    analyzer = CompressionAnalyzer()
    
    # Find all HDF5 files
    hdf5_files = list(Path(directory).glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {directory}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to analyze...")
    
    total_uncompressed = 0
    total_compressed = 0
    
    for file_path in hdf5_files:
        analysis = analyzer.analyze_existing_file(str(file_path))
        
        if 'error' not in analysis:
            total_uncompressed += analysis.get('uncompressed_size_mb', 0)
            total_compressed += analysis.get('file_size_mb', 0)
            
            print(f"\n{file_path.name}:")
            print(f"  Size: {analysis.get('file_size_mb', 0):.1f} MB")
            print(f"  Compression: {analysis.get('compression_method', 'None')}")
            print(f"  Ratio: {analysis.get('compression_ratio', 1):.2f}x")
            print(f"  Space saved: {analysis.get('space_saved_percent', 0):.1f}%")
        else:
            print(f"\n{file_path.name}: Error - {analysis['error']}")
    
    if total_uncompressed > 0:
        overall_ratio = total_uncompressed / total_compressed if total_compressed > 0 else 1
        space_saved = (1 - total_compressed/total_uncompressed) * 100
        
        print(f"\n{'='*60}")
        print(f"OVERALL STATISTICS:")
        print(f"  Total files: {len(hdf5_files)}")
        print(f"  Total compressed size: {total_compressed:.1f} MB")
        print(f"  Total uncompressed size: {total_uncompressed:.1f} MB")
        print(f"  Overall compression ratio: {overall_ratio:.2f}x")
        print(f"  Total space saved: {space_saved:.1f}% ({total_uncompressed - total_compressed:.1f} MB)")


if __name__ == "__main__":
    # Run analysis on existing files
    print("Analyzing existing HDF5 recording files...")
    analyze_recording_files()
    
    # Run benchmark test
    print("\n\nRunning compression benchmark...")
    analyzer = CompressionAnalyzer()
    
    # Test with typical camera frame size
    frame_shape = (480, 640, 3)  # Height, width, channels
    benchmark = analyzer.benchmark_compression(frame_shape, num_frames=50)
    
    analyzer.print_analysis_report(benchmark, "Compression Benchmark Results")
    
    # Clean up test files
    import shutil
    if os.path.exists("compression_test"):
        shutil.rmtree("compression_test")
        print("\nTest files cleaned up.")