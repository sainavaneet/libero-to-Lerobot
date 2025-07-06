"""
Utils package for the Libero to LeRobot format converter.

This package contains utility modules for file operations, image processing,
metadata generation, and HDF5 data processing.
"""

from .file_operations import create_directory_structure, ensure_output_directory
from .image_processing import (
    save_image_as_png,
    create_video_from_images,
    process_episode_images,
    create_episode_videos
)
from .metadata_generator import (
    create_info_json,
    create_modality_json,
    create_stats_json
)
from .hdf5_processor import (
    extract_demo_data,
    create_timestep_data,
    process_single_demo_for_chunk,
    get_demo_keys
)
from .batch_processor import (
    get_hdf5_files,
    extract_task_name_from_filename,
    process_single_hdf5_file,
    process_all_hdf5_files,
    create_global_metadata
)

__all__ = [
    # File operations
    'create_directory_structure',
    'ensure_output_directory',
    
    # Image processing
    'save_image_as_png',
    'create_video_from_images',
    'process_episode_images',
    'create_episode_videos',
    
    # Metadata generation
    'create_info_json',
    'create_modality_json',
    'create_stats_json',
    
    # HDF5 processing
    'extract_demo_data',
    'create_timestep_data',
    'process_single_demo_for_chunk',
    'get_demo_keys',
    
    # Batch processing
    'get_hdf5_files',
    'extract_task_name_from_filename',
    'process_single_hdf5_file',
    'process_all_hdf5_files',
    'create_global_metadata',
] 