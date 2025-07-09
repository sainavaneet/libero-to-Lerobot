# LIBERO to LeRobot Dataset Converter Configuration

# Dataset Configuration
FPS = 20.0                    # Frames per second
TIMESTEP_DURATION = 0.05      # Duration per timestep (seconds)
IMAGE_HEIGHT = 128            # Image height in pixels
IMAGE_WIDTH = 128             # Image width in pixels
IMAGE_CHANNELS = 3            # Number of color channels
JOINT_COUNT = 7               # Number of robot joints
SLEEP_TIME = 0.001           # Sleep time between operations

# Video Configuration
VIDEO_FPS = 20.0             # Video frame rate
VIDEO_CODEC = 'mp4v'         # Video codec
VIDEO_PIX_FMT = 'yuv420p'    # Video pixel format




# Processing Configuration
BATCH_SIZE = 1               # Number of files to process simultaneously
ENABLE_PROGRESS_BARS = True  # Enable/disable progress bars
ENABLE_VIDEO_CREATION = True # Enable/disable video creation
ENABLE_IMAGE_SAVING = True   # Enable/disable image saving

# Data Schema Configuration
TASK_DESCRIPTION_DEFAULT = 0  # Default task description index
VALIDITY_DEFAULT = 1          # Default validity flag (1 = valid, 0 = invalid)

# Error Handling
MAX_RETRIES = 3              # Maximum retries for failed operations
RETRY_DELAY = 1.0           # Delay between retries (seconds)

# This is COnfig file 

DATASET_DIR = "/home/navaneet/libero-to-Lerobot/datasets/libero_object"
OUTPUT_DIR = "/home/navaneet/libero-to-Lerobot/datasets/libero_object_lerobot_format5"