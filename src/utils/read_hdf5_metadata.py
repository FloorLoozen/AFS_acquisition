#!/usr/bin/env python3
"""
HDF5 Metadata Reader - Utility to inspect camera and stage settings stored in HDF5 files.
Usage: python read_hdf5_metadata.py <hdf5_file_path>
"""

import sys
import h5py
import json
from pathlib import Path

def read_hdf5_metadata(file_path):
    """Read and display all metadata from an HDF5 video file."""
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"📁 HDF5 File: {file_path}")
            print("=" * 60)
            
            # Video dataset info
            if 'video' in f:
                video_ds = f['video']
                print(f"\n📹 Video Dataset:")
                print(f"   Shape: {video_ds.shape}")
                print(f"   Dtype: {video_ds.dtype}")
                print(f"   Compression: {video_ds.compression}")
                print(f"   Chunks: {video_ds.chunks}")
                
                print(f"\n📋 Video Attributes:")
                for key, value in video_ds.attrs.items():
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    print(f"   {key:25}: {value}")
            
            # Camera settings
            if 'settings/camera' in f:
                camera_group = f['settings/camera']
                print(f"\n📸 Camera Settings ({len(camera_group.attrs)} parameters):")
                for key, value in camera_group.attrs.items():
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    print(f"   {key:25}: {value}")
            else:
                print(f"\n📸 Camera Settings: Not found")
            
            # Stage settings
            if 'settings/xy_stage' in f:
                stage_group = f['settings/xy_stage']
                print(f"\n🔧 Stage Settings ({len(stage_group.attrs)} parameters):")
                for key, value in stage_group.attrs.items():
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    print(f"   {key:25}: {value}")
            else:
                print(f"\n🔧 Stage Settings: Not found")
            
            # User metadata
            if 'metadata' in f:
                meta_group = f['metadata']
                print(f"\n📝 User Metadata ({len(meta_group.attrs)} parameters):")
                for key, value in meta_group.attrs.items():
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    print(f"   {key:25}: {value}")
            else:
                print(f"\n📝 User Metadata: Not found")
            
            # Timeseries data
            if 'timeseries' in f:
                timeseries_group = f['timeseries']
                print(f"\n📊 Timeseries Data:")
                for dataset_name in timeseries_group.keys():
                    dataset = timeseries_group[dataset_name]
                    print(f"   {dataset_name:20}: {dataset.shape} points")
                    if hasattr(dataset, 'attrs'):
                        for attr_key, attr_value in dataset.attrs.items():
                            if isinstance(attr_value, bytes):
                                attr_value = attr_value.decode('utf-8')
                            print(f"      {attr_key}: {attr_value}")
            else:
                print(f"\n📊 Timeseries Data: None")
            
            # File structure overview
            print(f"\n🗂️  HDF5 Structure:")
            def print_structure(name, obj):
                indent = "   " + "  " * name.count('/')
                if isinstance(obj, h5py.Group):
                    print(f"{indent}{name}/ (group)")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}{name} (dataset: {obj.shape}, {obj.dtype})")
            
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python read_hdf5_metadata.py <hdf5_file_path>")
        print("Example: python read_hdf5_metadata.py recording_20250924_143000.hdf5")
        return
    
    file_path = sys.argv[1]
    read_hdf5_metadata(file_path)

if __name__ == "__main__":
    main()