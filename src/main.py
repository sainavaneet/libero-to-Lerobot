import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
import json
from typing import Dict, List, Any
from PIL import Image
import cv2

def create_directory_structure(base_dir: str):
    """Create the directory structure matching the target dataset."""
    dirs = [
        os.path.join(base_dir, "data", "chunk-000"),
        os.path.join(base_dir, "videos", "chunk-000", "observation.images.agentview"),
        os.path.join(base_dir, "videos", "chunk-000", "observation.images.eye_in_hand"),
        os.path.join(base_dir, "meta"),
        os.path.join(base_dir, "images", "agentview"),
        os.path.join(base_dir, "images", "eye_in_hand")
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def extract_hdf5_to_lerobot_format(hdf5_path: str, output_dir: str):
    """
    Extract all demo data from HDF5 file and convert to LeRobot format.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
        output_dir (str): Directory to save the converted dataset
    """
    print(f"Reading HDF5 file: {hdf5_path}")
    
    # Create directory structure
    create_directory_structure(output_dir)
    
    # Initialize metadata
    episodes_data = []
    tasks_data = []
    task_descriptions = [
        "pick up the alphabet soup and place it in the basket",
        "valid"
    ]
    
    with h5py.File(hdf5_path, 'r') as f:
        # Access the data group
        data_group = f['data']
        
        # Get all demo keys (demo_0, demo_1, ..., demo_49)
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        demo_keys.sort()  # Sort to ensure consistent order
        
        print(f"Found {len(demo_keys)} demos: {demo_keys}")
        
        for i, demo_key in enumerate(demo_keys):
            demo_group = data_group[demo_key]
            print(f"Processing {demo_key}...")
            
            # Get the data for this demo
            actions = demo_group['actions'][:]
            dones = demo_group['dones'][:]
            rewards = demo_group['rewards'][:]
            obs = demo_group['obs']
            
            # Extract observation components
            agentview_rgb = obs['agentview_rgb'][:]
            ee_ori = obs['ee_ori'][:]
            ee_pos = obs['ee_pos'][:]
            ee_states = obs['ee_states'][:]
            eye_in_hand_rgb = obs['eye_in_hand_rgb'][:]
            gripper_states = obs['gripper_states'][:]
            joint_states = obs['joint_states'][:]
            
            # Get the number of timesteps in this demo
            num_timesteps = len(actions)
            print(f"  Saving {num_timesteps} images for agentview and eye_in_hand...")
            
            # Create data for each timestep
            demo_data = []
            for t in range(num_timesteps):
                timestamp = float(t) * 0.05  # Assuming 20 FPS (0.05s per frame)
                
                # Save agentview image
                agentview_img = agentview_rgb[t]
                agentview_filename = f"episode_{i:06d}_timestamp_{timestamp:.3f}.png"
                agentview_path = os.path.join(output_dir, "images", "agentview", agentview_filename)
                
                # Convert numpy array to PIL Image and save
                if agentview_img.dtype != np.uint8:
                    # Normalize to 0-255 range if needed
                    if agentview_img.max() <= 1.0:
                        agentview_img = (agentview_img * 255).astype(np.uint8)
                    else:
                        agentview_img = agentview_img.astype(np.uint8)
                
                agentview_pil = Image.fromarray(agentview_img)
                agentview_pil.save(agentview_path)
                
                # Save eye_in_hand image
                eye_in_hand_img = eye_in_hand_rgb[t]
                eye_in_hand_filename = f"episode_{i:06d}_timestamp_{timestamp:.3f}.png"
                eye_in_hand_path = os.path.join(output_dir, "images", "eye_in_hand", eye_in_hand_filename)
                
                # Convert numpy array to PIL Image and save
                if eye_in_hand_img.dtype != np.uint8:
                    # Normalize to 0-255 range if needed
                    if eye_in_hand_img.max() <= 1.0:
                        eye_in_hand_img = (eye_in_hand_img * 255).astype(np.uint8)
                    else:
                        eye_in_hand_img = eye_in_hand_img.astype(np.uint8)
                
                eye_in_hand_pil = Image.fromarray(eye_in_hand_img)
                eye_in_hand_pil.save(eye_in_hand_path)

                timestep_data = {
                    'observation.state': joint_states[t].tolist(),
                    'action': actions[t].tolist(),
                    'timestamp': timestamp,
                    'annotation.human.action.task_description': 0,  # Default task index
                    'task_index': 0,  # Default task index
                    'annotation.human.validity': 1,  # Valid by default
                    'episode_index': i,
                    'index': t,
                    'next.reward': float(rewards[t]),
                    'next.done': bool(dones[t]),
                }
                demo_data.append(timestep_data)
                
                # Progress indicator every 100 timesteps
                if t % 100 == 0 and t > 0:
                    print(f"    Processed {t}/{num_timesteps} timesteps...")
                
                time.sleep(0.001)
            
            # Convert to DataFrame
            df = pd.DataFrame(demo_data)
            
            # Create episode filename (episode000000, episode000001, etc.)
            episode_filename = f"episode_{i:06d}.parquet"
            output_path = os.path.join(output_dir, "data", "chunk-000", episode_filename)
            
            # Save to parquet
            print(f"Saving {demo_key} as {episode_filename} with {len(df)} timesteps...")
            df.to_parquet(output_path, index=False)
            
            # Add episode metadata
            episodes_data.append({
                "episode_index": i,
                "tasks": [task_descriptions[0], task_descriptions[1]],
                "length": len(df)
            })
            
            print(f"Successfully saved {len(df)} rows to {output_path}")
            print(f"  Saved {num_timesteps} agentview images and {num_timesteps} eye_in_hand images")

            # --- Create videos from saved images ---
            for cam_type in ["agentview", "eye_in_hand"]:
                images_dir = os.path.join(output_dir, "images", cam_type)
                # Find all images for this episode
                prefix = f"episode_{i:06d}_timestamp_"
                image_files = [f for f in os.listdir(images_dir) if f.startswith(prefix) and f.endswith('.png')]
                # Sort by timestamp in filename
                image_files.sort(key=lambda x: float(x.split("timestamp_")[1].replace(".png", "")))
                if not image_files:
                    print(f"  No images found for {cam_type} in episode {i}")
                    continue
                # Read first image to get shape
                first_img = cv2.imread(os.path.join(images_dir, image_files[0]))
                height, width, layers = first_img.shape
                # Set video output path
                if cam_type == "agentview":
                    video_dir = os.path.join(output_dir, "videos", "chunk-000", "observation.images.agentview")
                else:
                    video_dir = os.path.join(output_dir, "videos", "chunk-000", "observation.images.eye_in_hand")
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f"episode_{i:06d}.mp4")
                # Define video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
                for img_file in image_files:
                    img = cv2.imread(os.path.join(images_dir, img_file))
                    out.write(img)
                out.release()
                print(f"  Saved video for {cam_type} at {video_path}")
    
    # Create metadata files
    create_metadata_files(output_dir, episodes_data, tasks_data, task_descriptions, len(demo_keys))
    
    print(f"All demos processed. Files saved in: {output_dir}")

def create_metadata_files(output_dir: str, episodes_data: List[Dict], tasks_data: List[Dict], 
                        task_descriptions: List[str], total_episodes: int):
    """Create all metadata files matching the target dataset format."""
    
    # Create episodes.jsonl
    episodes_path = os.path.join(output_dir, "meta", "episodes.jsonl")
    with open(episodes_path, 'w') as f:
        for episode in episodes_data:
            f.write(json.dumps(episode) + '\n')
    
    # Create tasks.jsonl
    tasks_path = os.path.join(output_dir, "meta", "tasks.jsonl")
    with open(tasks_path, 'w') as f:
        for i, task in enumerate(task_descriptions):
            f.write(json.dumps({"task_index": i, "task": task}) + '\n')
    
    # Create info.json
    info_data = {
        "codebase_version": "v2.0",
        "robot_type": "panda-robot",
        "total_episodes": total_episodes,
        "total_frames": sum(ep["length"] for ep in episodes_data),
        "total_tasks": len(task_descriptions),
        "total_videos": len(episodes_data) * 2,  # We have agentview and eye_in_hand videos
        "total_chunks": 0,
        "chunks_size": 1000,
        "fps": 20.0,
        "splits": {
            "train": "0:50"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.agentview_rgb": {
                "dtype": "video",
                "shape": [128, 128, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": 20.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.eye_in_hand_rgb": {
                "dtype": "video",
                "shape": [128, 128, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": 20.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [7],
                "names": [f"motor_{i}" for i in range(7)]
            },
            "action": {
                "dtype": "float64",
                "shape": [7],
                "names": [f"motor_{i}" for i in range(7)]
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [1]
            },
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1]
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1]
            },
            "annotation.human.validity": {
                "dtype": "int64",
                "shape": [1]
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1]
            },
            "index": {
                "dtype": "int64",
                "shape": [1]
            },
            "next.reward": {
                "dtype": "float64",
                "shape": [1]
            },
            "next.done": {
                "dtype": "bool",
                "shape": [1]
            }
        }
    }
    
    info_path = os.path.join(output_dir, "meta", "info.json")
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=4)
    
    # Create modality.json
    modality_data = {
        "observation.images.agentview_rgb": {
            "dtype": "video",
            "shape": [128, 128, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": 20.0,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.images.eye_in_hand_rgb": {
            "dtype": "video",
            "shape": [128, 128, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": 20.0,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.state": {
            "dtype": "float64",
            "shape": [7],
            "names": [f"motor_{i}" for i in range(7)]
        },
        "action": {
            "dtype": "float64",
            "shape": [7],
            "names": [f"motor_{i}" for i in range(7)]
        }
    }
    
    modality_path = os.path.join(output_dir, "meta", "modality.json")
    with open(modality_path, 'w') as f:
        json.dump(modality_data, f, indent=4)
    
    # Create stats.json (simplified version)
    stats_data = {
        "observation.state": {
            "mean": [0.0] * 7,
            "std": [1.0] * 7,
            "min": [-1.0] * 7,
            "max": [1.0] * 7
        },
        "action": {
            "mean": [0.0] * 7,
            "std": [1.0] * 7,
            "min": [-1.0] * 7,
            "max": [1.0] * 7
        }
    }
    
    stats_path = os.path.join(output_dir, "meta", "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=4)
    
    print("Created all metadata files")

def main():
    # Define paths
    hdf5_file = "/home/navaneet/libero-to-Lerobot/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    output_dir = "/home/navaneet/libero-to-Lerobot/datasets/libero_object_lerobot_format"
    
    # Check if input file exists
    if not os.path.exists(hdf5_file):
        print(f"Error: HDF5 file not found at {hdf5_file}")
        return
    
    # Extract and convert data
    try:
        extract_hdf5_to_lerobot_format(hdf5_file, output_dir)
        print("Conversion completed successfully!")
        
        # Print some statistics
        print("\nData Statistics:")
        parquet_files = [f for f in os.listdir(os.path.join(output_dir, "data", "chunk-000")) if f.endswith('.parquet')]
        print(f"Total parquet files created: {len(parquet_files)}")
        
        # Show sample of the first file
        if parquet_files:
            first_file = os.path.join(output_dir, "data", "chunk-000", parquet_files[0])
            df_sample = pd.read_parquet(first_file)
            print(f"\nSample data from {parquet_files[0]} (first 3 rows):")
            print(df_sample.head(3))
            print(f"\nColumns: {df_sample.columns.tolist()}")
        
        # Count total images saved
        agentview_images = len([f for f in os.listdir(os.path.join(output_dir, "images", "agentview")) if f.endswith('.png')])
        eye_in_hand_images = len([f for f in os.listdir(os.path.join(output_dir, "images", "eye_in_hand")) if f.endswith('.png')])
        print(f"\nTotal images saved:")
        print(f"  Agentview images: {agentview_images}")
        print(f"  Eye-in-hand images: {eye_in_hand_images}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
