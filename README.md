# LIBERO to LeRobot Dataset Converter

This repo helps to convert LIBERO (LIkelihood-BasEd RObotic learning) datasets to the LeRobot dataset format. This converter processes HDF5 files containing robotic demonstration data and transforms them into the standardized LeRobot format with proper directory structure, metadata, and data organization.

## 📁 Project Structure

```
libero-to-Lerobot/
├── src/
│   ├── batch_converter.py          # Main conversion script
│   ├── config.py                   # Configuration parameters
│   └── utils/
│       ├── __init__.py
│       ├── batch_processor.py      # Batch processing logic
│       ├── hdf5_processor.py       # HDF5 file processing
│       ├── image_processing.py     # Image and video processing
│       ├── file_operations.py      # File and directory operations
│       └── metadata_generator.py   # Metadata file generation
├── datasets/
│   ├── libero_object/             # Input LIBERO dataset directory
│   └── libero_object_lerobot_format/  # Output LeRobot format directory
├── requirements.txt                # Python dependencies
├── README.md                      # This file
└── .gitignore
```

## 🚀 Features

- **Progress Tracking**: Multi-level progress bars showing file, demo, and frame processing
- **Batch Processing**: Convert multiple HDF5 files simultaneously
- **Image Processing**: Extract and save RGB images from demonstrations
- **Video Generation**: Create MP4 videos from image sequences
- **Metadata Generation**: Generate comprehensive dataset metadata
- **Error Handling**: Robust error handling with detailed logging
- **Configurable**: Easy configuration through `config.py`

## 📋 Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `h5py` - HDF5 file handling
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `PIL` - Image processing
- `opencv-python` - Video creation
- `tqdm` - Progress bars

## ⚙️ Configuration

Edit `src/config.py` to customize the conversion parameters:

```python
# Dataset Configuration
FPS = 20.0                    # Frames per second
TIMESTEP_DURATION = 0.05      # Duration per timestep (seconds)
IMAGE_HEIGHT = 128            # Image height in pixels
IMAGE_WIDTH = 128             # Image width in pixels
IMAGE_CHANNELS = 3            # Number of color channels
JOINT_COUNT = 7               # Number of robot joints
SLEEP_TIME = 0.001           # Sleep time between operations

# Paths
INPUT_DIR = "/path/to/libero/dataset"
OUTPUT_DIR = "/path/to/output/lerobot/dataset"
```

## 🎯 Usage

### Basic Usage

1. **Prepare your LIBERO dataset**:
   - Place HDF5 files in the input directory
   - Ensure files follow the naming convention: `{task_name}_demo.hdf5`

2. **Run the converter**:
   ```bash
   python src/batch_converter.py
   ```

### Advanced Usage

You can also use the converter programmatically:

```python
from src.utils.batch_processor import process_all_hdf5_files

# Process all HDF5 files
chunks_metadata = process_all_hdf5_files(
    input_dir="/path/to/input",
    output_dir="/path/to/output"
)
```

## 📊 Output Format

The converter creates a LeRobot-compatible dataset with the following structure:

```
output_directory/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   ├── chunk-001/
│   │   └── ...
│   └── ...
├── videos/
│   ├── chunk-000/
│   │   ├── observation.images.agentview_rgb/
│   │   │   ├── episode_000000.mp4
│   │   │   └── ...
│   │   └── observation.images.eye_in_hand_rgb/
│   │       ├── episode_000000.mp4
│   │       └── ...
│   └── ...
├── images/
│   ├── agentview/
│   │   ├── episode_000000_timestamp_0.000.png
│   │   └── ...
│   └── eye_in_hand/
│       ├── episode_000000_timestamp_0.000.png
│       └── ...
└── meta/
    ├── info.json
    ├── modality.json
    └── stats.json
```

## 📈 Progress Tracking

The converter provides detailed progress tracking with multiple levels:

1. **File Level**: Overall progress through HDF5 files
2. **Demo Level**: Progress through demonstrations within each file
3. **Frame Level**: Progress through image processing
4. **Timestep Level**: Progress through data creation

Example output:
```
============================================================
CONVERTING INTO LEROBOT DATASET FORMAT
============================================================

Processing HDF5 files: 100%|██████████| 5/5 [02:30<00:00, 30.0s/file]
Processing demos in task1_demo.hdf5: 100%|██████████| 10/10 [00:45<00:00, 4.5s/demo]
Processing images for demo_0: 100%|██████████| 500/500 [00:10<00:00, 50.0frame/s]
Creating timestep data for demo_0: 100%|██████████| 500/500 [00:05<00:00, 100.0timestep/s]
```

## 🔧 Data Format Conversion

### Input (LIBERO Format)
- **HDF5 files** containing demonstration data
- **Structure**: `data/demo_X/` with observations, actions, rewards
- **Images**: RGB arrays for agentview and eye_in_hand cameras
- **Actions**: 7-dimensional robot joint actions
- **States**: 7-dimensional robot joint states

### Output (LeRobot Format)
- **Parquet files** for each episode with structured data
- **Videos**: MP4 files for each camera view
- **Images**: PNG files for each timestep
- **Metadata**: JSON files with dataset statistics and modality information

### Data Schema
```python
{
    'observation.state': [joint_states],      # Robot joint states
    'action': [actions],                      # Robot actions
    'timestamp': [timestamp],                 # Time in seconds
    'episode_index': [episode_id],           # Episode identifier
    'index': [timestep_index],               # Timestep within episode
    'next.reward': [rewards],                # Reward values
    'next.done': [dones],                    # Episode termination flags
    'annotation.human.action.task_description': [task_index],
    'annotation.human.validity': [validity]
}
```

## 🛠️ Customization

### Adding New Data Modalities

1. **Extend the data extraction** in `hdf5_processor.py`:
   ```python
   def extract_demo_data(demo_group):
       # Add your new data extraction here
       new_data = demo_group['new_modality'][:]
       return actions, dones, rewards, ..., new_data
   ```

2. **Update the timestep creation** in `hdf5_processor.py`:
   ```python
   def create_timestep_data(t, actions, rewards, dones, joint_states, episode_index, new_data):
       return {
           # ... existing fields
           'new_modality': new_data[t].tolist(),
       }
   ```

3. **Update metadata generation** in `metadata_generator.py`:
   ```python
   def create_modality_json():
       return {
           # ... existing modalities
           "new_modality": {
               "dtype": "float64",
               "shape": [your_shape],
               "names": ["dim1", "dim2", ...]
           }
       }
   ```

### Modifying Image Processing

Edit `src/utils/image_processing.py` to customize:
- Image resolution
- Video codec settings
- Image format and compression
- Video frame rate


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- [LIBERO Dataset](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
- [HDF5 Documentation](https://www.hdfgroup.org/solutions/hdf5/)
- [Pandas Documentation](https://pandas.pydata.org/)
