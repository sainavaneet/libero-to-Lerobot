#!/usr/bin/env python3
"""
Main entry point for the Libero to LeRobot format converter.

This module orchestrates the conversion process using the utility modules.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from batch_converter import main as batch_converter_main


def main():
    """
    Main entry point for the conversion process.
    
    This function can be called directly or used as a module.
    """
    print()

    print("="*60)
    print("AVAILABLE DATASETS:")
    print("="*60 + "\n")

    # Use the batch converter's main function
    batch_converter_main()


if __name__ == "__main__":
    main()
