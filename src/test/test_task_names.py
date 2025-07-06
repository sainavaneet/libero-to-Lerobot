#!/usr/bin/env python3
"""
Test script to verify task name extraction from HDF5 filenames.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_task_name_extraction():
    """Test that task names are extracted correctly from filenames."""
    from utils.batch_processor import extract_task_name_from_filename
    
    # Test cases with expected results
    test_cases = [
        {
            "filename": "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
        },
        {
            "filename": "pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_bbq_sauce_and_place_it_in_the_basket"
        },
        {
            "filename": "pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_butter_and_place_it_in_the_basket"
        },
        {
            "filename": "pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
        },
        {
            "filename": "pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_cream_cheese_and_place_it_in_the_basket"
        },
        {
            "filename": "pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_ketchup_and_place_it_in_the_basket"
        },
        {
            "filename": "pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_milk_and_place_it_in_the_basket"
        },
        {
            "filename": "pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_orange_juice_and_place_it_in_the_basket"
        },
        {
            "filename": "pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_salad_dressing_and_place_it_in_the_basket"
        },
        {
            "filename": "pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5",
            "expected": "pick_up_the_tomato_sauce_and_place_it_in_the_basket"
        }
    ]
    
    print("Testing task name extraction...")
    print("=" * 60)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases):
        filename = test_case["filename"]
        expected = test_case["expected"]
        
        # Extract task name
        actual = extract_task_name_from_filename(filename)
        
        # Check if correct
        if actual == expected:
            print(f"✅ Test {i+1}: {filename}")
            print(f"   Extracted: {actual}")
        else:
            print(f"❌ Test {i+1}: {filename}")
            print(f"   Expected:  {expected}")
            print(f"   Actual:    {actual}")
            all_passed = False
        
        print()
    
    print("=" * 60)
    if all_passed:
        print("🎉 All task name extraction tests passed!")
    else:
        print("❌ Some tests failed.")
    
    return all_passed


def test_actual_files():
    """Test with actual files from the dataset directory."""
    from utils.batch_processor import get_hdf5_files, extract_task_name_from_filename
    
    input_dir = "/home/navaneet/libero-to-Lerobot/datasets/libero_object"
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    hdf5_files = get_hdf5_files(input_dir)
    
    print(f"\nTesting with actual files from {input_dir}:")
    print("=" * 60)
    
    for i, file_path in enumerate(hdf5_files):
        filename = os.path.basename(file_path)
        task_name = extract_task_name_from_filename(filename)
        
        print(f"{i+1:2d}. {filename}")
        print(f"    → {task_name}")
        print()
    
    return True


def main():
    """Run all tests."""
    success = True
    
    # Test task name extraction
    if not test_task_name_extraction():
        success = False
    
    # Test with actual files
    if not test_actual_files():
        success = False
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("\nThe task names will now be used correctly in the batch processing.")
    else:
        print("\n❌ Some tests failed.")
    
    return success


if __name__ == "__main__":
    main() 