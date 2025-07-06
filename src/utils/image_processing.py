import os
import cv2
import numpy as np
from PIL import Image
from typing import List
from config import FPS, TIMESTEP_DURATION, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, JOINT_COUNT, SLEEP_TIME

def save_image_as_png(image_array: np.ndarray, output_path: str) -> None:
    """Save a numpy array as a PNG image with proper normalization."""
    # Convert numpy array to PIL Image and save
    if image_array.dtype != np.uint8:
        # Normalize to 0-255 range if needed
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    pil_image = Image.fromarray(image_array)
    pil_image.save(output_path)


def create_video_from_images(image_files: List[str], images_dir: str, video_path: str, fps: float = FPS) -> None:
    """Create a video from a list of image files."""
    if not image_files:
        print(f"  No images found for video creation")
        return
    
    # Read first image to get shape
    first_img = cv2.imread(os.path.join(images_dir, image_files[0]))
    height, width, layers = first_img.shape
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for img_file in image_files:
        img = cv2.imread(os.path.join(images_dir, img_file))
        out.write(img)
    
    out.release()
    # print(f"  Saved video at {video_path}")


def process_episode_images(episode_index: int, output_dir: str, agentview_rgb: np.ndarray, 
                          eye_in_hand_rgb: np.ndarray, num_timesteps: int, pbar=None) -> List[str]:
    """Process and save images for an episode, return list of image filenames."""
    image_filenames = []
    
    for t in range(num_timesteps):
        timestamp = float(t) * TIMESTEP_DURATION  # Assuming 20 FPS (0.05s per frame)
        
        # Save agentview image
        agentview_img = agentview_rgb[t]
        agentview_filename = f"episode_{episode_index:06d}_timestamp_{timestamp:.3f}.png"
        agentview_path = os.path.join(output_dir, "images", "agentview", agentview_filename)
        save_image_as_png(agentview_img, agentview_path)
        
        # Save eye_in_hand image
        eye_in_hand_img = eye_in_hand_rgb[t]
        eye_in_hand_filename = f"episode_{episode_index:06d}_timestamp_{timestamp:.3f}.png"
        eye_in_hand_path = os.path.join(output_dir, "images", "eye_in_hand", eye_in_hand_filename)
        save_image_as_png(eye_in_hand_img, eye_in_hand_path)
        
        image_filenames.append(agentview_filename)
        
        # Update progress bar if provided
        if pbar:
            pbar.update(1)
        # Progress indicator every 100 timesteps (only if no progress bar)
        elif t % 100 == 0 and t > 0:
            print(f"    Processed {t}/{num_timesteps} timesteps...")
    
    return image_filenames


def create_episode_videos(episode_index: int, output_dir: str, chunk_index: int = 0) -> None:
    """Create videos from saved images for an episode."""
    for cam_type in ["agentview", "eye_in_hand"]:
        images_dir = os.path.join(output_dir, "images", cam_type)
        
        # Find all images for this episode
        prefix = f"episode_{episode_index:06d}_timestamp_"
        image_files = [f for f in os.listdir(images_dir) if f.startswith(prefix) and f.endswith('.png')]
        
        # Sort by timestamp in filename
        image_files.sort(key=lambda x: float(x.split("timestamp_")[1].replace(".png", "")))
        
        if not image_files:
            print(f"  No images found for {cam_type} in episode {episode_index}")
            continue
        
        # Set video output path using chunk_index
        chunk_name = f"chunk-{chunk_index:03d}"
        if cam_type == "agentview":
            video_dir = os.path.join(output_dir, "videos", chunk_name, "observation.images.agentview_rgb")
        else:
            video_dir = os.path.join(output_dir, "videos", chunk_name, "observation.images.eye_in_hand_rgb")
        
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"episode_{episode_index:06d}.mp4")
        
        create_video_from_images(image_files, images_dir, video_path) 