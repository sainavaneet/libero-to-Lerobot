# Libero to LeRobot Format Converter

This module has been refactored into multiple smaller, more manageable files for better maintainability and organization.

## File Structure

```
src/
├── main.py                    # Main entry point (simplified)
├── converter.py               # Main conversion orchestrator
├── config.py                  # Configuration and constants
├── utils/
│   ├── file_operations.py     # Directory and file management
│   ├── image_processing.py    # Image and video processing
│   ├── metadata_generator.py  # Metadata file generation
│   ├── hdf5_processor.py      # HDF5 data extraction and processing
│   ├── explore_hdf5.py        # HDF5 exploration utilities
│   └── explore_obs.py         # Observation exploration utilities
└── README.md                  # This file
```

## Module Responsibilities

### `main.py`
- **Purpose**: Entry point for the conversion process
- **Responsibilities**: 
  - Sets up Python path
  - Delegates to converter module
  - Provides clean interface

### `converter.py`
- **Purpose**: Main conversion orchestrator
- **Responsibilities**:
  - Coordinates the entire conversion process
  - Manages HDF5 file reading
  - Handles statistics and reporting
  - Error handling and logging

### `config.py`
- **Purpose**: Centralized configuration
- **Responsibilities**:
  - Default file paths
  - Task descriptions
  - Video and image settings
  - Robot configuration parameters

### `utils/file_operations.py`
- **Purpose**: File and directory management
- **Responsibilities**:
  - Creating directory structure
  - Ensuring output directories exist
  - Path validation

### `utils/image_processing.py`
- **Purpose**: Image and video processing
- **Responsibilities**:
  - Saving images as PNG files
  - Creating videos from image sequences
  - Image normalization and format conversion
  - Episode-specific image processing

### `utils/metadata_generator.py`
- **Purpose**: Metadata file generation
- **Responsibilities**:
  - Creating episodes.jsonl
  - Creating tasks.jsonl
  - Creating info.json
  - Creating modality.json
  - Creating stats.json

### `utils/hdf5_processor.py`
- **Purpose**: HDF5 data extraction and processing
- **Responsibilities**:
  - Extracting data from HDF5 groups
  - Processing individual demos
  - Creating timestep data
  - Managing episode metadata

## Usage

### Running the Converter

```bash
# From the project root
python src/main.py

# Or directly
python src/converter.py
```

### Using Individual Modules

```python
# Import specific functionality
from utils.file_operations import ensure_output_directory
from utils.image_processing import save_image_as_png
from utils.metadata_generator import create_metadata_files
from utils.hdf5_processor import process_single_demo

# Use individual functions
ensure_output_directory("/path/to/output")
save_image_as_png(image_array, "/path/to/image.png")
```

## Benefits of the Modular Structure

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Individual functions can be tested in isolation
3. **Reusability**: Functions can be imported and used independently
4. **Readability**: Smaller files are easier to understand
5. **Debugging**: Issues can be isolated to specific modules
6. **Extensibility**: New features can be added to appropriate modules

## Configuration

All configuration parameters are centralized in `config.py`:

- File paths
- Task descriptions
- Video settings (FPS, codec)
- Image dimensions
- Robot configuration
- Dataset structure parameters

## Error Handling

The modular structure allows for better error handling:
- File operations errors are isolated to `file_operations.py`
- Image processing errors are isolated to `image_processing.py`
- HDF5 processing errors are isolated to `hdf5_processor.py`

## Future Improvements

1. Add type hints to all functions
2. Add comprehensive error handling
3. Add unit tests for each module
4. Add logging throughout the modules
5. Add configuration validation
6. Add progress bars for long operations 