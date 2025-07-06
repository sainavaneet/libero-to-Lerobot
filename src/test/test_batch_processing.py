#!/usr/bin/env python3
"""
Test script to verify batch processing functionality.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_batch_processor_imports():
    """Test that batch processing modules can be imported successfully."""
    try:
        from utils.batch_processor import (
            get_hdf5_files,
            extract_task_name_from_filename,
            process_single_hdf5_file,
            process_all_hdf5_files,
            create_global_metadata
        )
        
        print("✅ Batch processor imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_hdf5_file_discovery():
    """Test that HDF5 files can be discovered in the input directory."""
    try:
        from utils.batch_processor import get_hdf5_files
        
        input_dir = "/home/navaneet/libero-to-Lerobot/datasets/libero_object"
        hdf5_files = get_hdf5_files(input_dir)
        
        print(f"✅ Found {len(hdf5_files)} HDF5 files:")
        for i, file_path in enumerate(hdf5_files):
            print(f"  {i+1}. {os.path.basename(file_path)}")
        
        return True
        
    except Exception as e:
        print(f"❌ HDF5 file discovery error: {e}")
        return False


def test_task_name_extraction():
    """Test that task names can be extracted from filenames."""
    try:
        from utils.batch_processor import extract_task_name_from_filename
        
        test_files = [
            "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
            "pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5",
            "pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5"
        ]
        
        print("✅ Task name extraction test:")
        for filename in test_files:
            task_name = extract_task_name_from_filename(filename)
            print(f"  {filename} -> {task_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Task name extraction error: {e}")
        return False


def test_directory_structure_creation():
    """Test that chunk directory structures can be created."""
    try:
        from utils.batch_processor import create_chunk_directory_structure
        
        # Test with a temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_name = create_chunk_directory_structure(temp_dir, 0)
            print(f"✅ Created chunk directory: {chunk_name}")
            
            # Check if directories were created
            expected_dirs = [
                os.path.join(temp_dir, "data", "chunk-000"),
                os.path.join(temp_dir, "videos", "chunk-000", "observation.images.agentview_rgb"),
                os.path.join(temp_dir, "videos", "chunk-000", "observation.images.eye_in_hand_rgb"),
                os.path.join(temp_dir, "meta"),
                os.path.join(temp_dir, "images", "agentview"),
                os.path.join(temp_dir, "images", "eye_in_hand")
            ]
            
            for dir_path in expected_dirs:
                if os.path.exists(dir_path):
                    print(f"  ✅ {dir_path}")
                else:
                    print(f"  ❌ {dir_path} (missing)")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure creation error: {e}")
        return False


def main():
    """Run all batch processing tests."""
    print("Testing batch processing functionality...")
    print("=" * 60)
    
    success = True
    
    # Test imports
    if not test_batch_processor_imports():
        success = False
    
    print()
    
    # Test HDF5 file discovery
    if not test_hdf5_file_discovery():
        success = False
    
    print()
    
    # Test task name extraction
    if not test_task_name_extraction():
        success = False
    
    print()
    
    # Test directory structure creation
    if not test_directory_structure_creation():
        success = False
    
    print()
    print("=" * 60)
    
    if success:
        print("🎉 All batch processing tests passed!")
        print("\nYou can now run the batch converter with:")
        print("  python src/batch_converter.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success


if __name__ == "__main__":
    main() 