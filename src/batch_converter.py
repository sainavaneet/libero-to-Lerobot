#!/usr/bin/env python3


import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from utils.batch_processor import process_all_hdf5_files
from config import DATASET_DIR, OUTPUT_DIR

def main():
    """Main function to run the batch conversion process."""
    # Define paths
    input_dir = DATASET_DIR
    output_dir = OUTPUT_DIR
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return
    
    # Check if there are HDF5 files in the input directory
    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    if not hdf5_files:
        print(f"Error: No HDF5 files found in {input_dir}")
        return
    
    # print(f"Found {len(hdf5_files)} HDF5 files to process:")
    for i, file_name in enumerate(hdf5_files):
        print(f"  {i+1}. {file_name}")
    
    # Process all HDF5 files with progress bars
    try:
        chunks_metadata = process_all_hdf5_files(input_dir, output_dir)
        
        if chunks_metadata:
            print("\n" + "="*60)
            print("BATCH CONVERSION COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Print summary statistics
            total_episodes = sum(chunk["episodes_count"] for chunk in chunks_metadata)
            total_frames = sum(chunk["total_frames"] for chunk in chunks_metadata)
            
            print(f"\nSummary:")
            print(f"  Total chunks created: {len(chunks_metadata)}")
            print(f"  Total episodes: {total_episodes}")
            print(f"  Total frames: {total_frames}")
            print(f"  Output directory: {output_dir}")
            
            print(f"\nChunk details:")
            for chunk in chunks_metadata:
                print(f"  {chunk['chunk_name']}: {chunk['task_name']} ({chunk['episodes_count']} episodes)")
            
            # Show directory structure
            print(f"\nDirectory structure created:")
            for chunk in chunks_metadata:
                chunk_dir = os.path.join(output_dir, "data", chunk["chunk_name"])
                if os.path.exists(chunk_dir):
                    parquet_files = [f for f in os.listdir(chunk_dir) if f.endswith('.parquet')]
                    print(f"  {chunk['chunk_name']}: {len(parquet_files)} parquet files")
            
        else:
            print("❌ No chunks were successfully processed.")
            
    except Exception as e:
        print(f"Error during batch conversion: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 