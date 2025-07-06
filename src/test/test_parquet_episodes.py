#!/usr/bin/env python3
"""
Test script to verify that episode indices in parquet files are consecutive.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_parquet_episode_indices():
    """Test that episode indices in parquet files are consecutive."""
    from utils.batch_processor import process_all_hdf5_files
    
    # Use a small test dataset
    input_dir = "datasets/libero_object"
    output_dir = "test_output_parquet"
    
    # Clean up any existing output
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    
    print("Testing parquet episode indices...")
    print("=" * 60)
    
    # Process the files
    process_all_hdf5_files(input_dir, output_dir)
    
    # Check all parquet files
    parquet_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    parquet_files.sort()
    
    print(f"Found {len(parquet_files)} parquet files:")
    
    all_episode_indices = []
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            episode_indices = df['episode_index'].unique()
            all_episode_indices.extend(episode_indices)
            print(f"  {parquet_file}: episode indices {episode_indices}")
        except Exception as e:
            print(f"  Error reading {parquet_file}: {e}")
    
    # Check if episode indices are consecutive
    all_episode_indices = sorted(list(set(all_episode_indices)))
    expected_indices = list(range(len(all_episode_indices)))
    
    if all_episode_indices == expected_indices:
        print(f"\n✅ Episode indices in parquet files are consecutive!")
        print(f"   Episode indices: {all_episode_indices}")
        return True
    else:
        print(f"\n❌ Episode indices in parquet files are not consecutive!")
        print(f"   Expected: {expected_indices}")
        print(f"   Actual: {all_episode_indices}")
        return False


if __name__ == "__main__":
    success = test_parquet_episode_indices()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Parquet episode indices test passed!")
        print("Episode indices in parquet files are now consecutive across all chunks.")
    else:
        print("❌ Parquet episode indices test failed.")
    
    sys.exit(0 if success else 1) 