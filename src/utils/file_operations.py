import os
from typing import List


def create_directory_structure(base_dir: str) -> None:
    """Create the directory structure matching the target dataset."""
    dirs = [
        os.path.join(base_dir, "data", "chunk-000"),
        os.path.join(base_dir, "videos", "chunk-000", "observation.images.agentview_rgb"),
        os.path.join(base_dir, "videos", "chunk-000", "observation.images.eye_in_hand_rgb"),
        os.path.join(base_dir, "meta"),
        os.path.join(base_dir, "images", "agentview"),
        os.path.join(base_dir, "images", "eye_in_hand")
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def ensure_output_directory(output_dir: str) -> None:
    """Ensure the output directory exists and create the required structure."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    create_directory_structure(output_dir) 