import h5py
import pandas as pd
import numpy as np
import os
import time
from typing import Dict, List, Any, Tuple, Union
from tqdm import tqdm
from .image_processing import process_episode_images, create_episode_videos
from config import FPS, TIMESTEP_DURATION, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, JOINT_COUNT , SLEEP_TIME, OUTPUT_DIR

def extract_demo_data(demo_group: h5py.Group) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray, np.ndarray]:
    """Extract all data from a demo group."""
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
    
    return actions, dones, rewards, agentview_rgb, ee_ori, ee_pos, ee_states, eye_in_hand_rgb, gripper_states, joint_states


def create_timestep_data(t: int, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, 
                        joint_states: np.ndarray, global_episode_index: int) -> Dict[str, Any]:
    """Create data dictionary for a single timestep."""
    timestamp = float(t) * TIMESTEP_DURATION  # Assuming 20 FPS (0.05s per frame)
    
    return {
        'observation.state': joint_states[t].tolist(),
        'action': actions[t].tolist(),
        'timestamp': timestamp,
        'annotation.human.action.task_description': 0,  # Default task index
        'task_index': 0,  # Default task index
        'annotation.human.validity': 1,  # Valid by default
        'episode_index': global_episode_index,  # Use global episode index
        'index': t,
        'next.reward': float(rewards[t]),
        'next.done': bool(dones[t]),
    }


def process_single_demo_for_chunk(demo_group: h5py.Group, episode_index: int, output_dir: str, 
                                 chunk_index: int) -> Dict[str, Any]:
    """Process a single demo for a specific chunk."""
    # print(f"Processing demo_{episode_index}...")
    
    # Extract all data
    actions, dones, rewards, agentview_rgb, ee_ori, ee_pos, ee_states, eye_in_hand_rgb, gripper_states, joint_states = extract_demo_data(demo_group)
    
    # Get the number of timesteps in this demo
    num_timesteps = len(actions)
    # print(f"  Saving {num_timesteps} images for agentview and eye_in_hand...")
    
    # Process images with progress bar
    with tqdm(total=num_timesteps, desc=f"Processing images for demo_{episode_index}", 
              unit="frame", leave=False, position=2) as img_pbar:
        process_episode_images(episode_index, output_dir, agentview_rgb, eye_in_hand_rgb, num_timesteps, img_pbar)
    
    # Create data for each timestep with progress bar
    demo_data = []
    with tqdm(total=num_timesteps, desc=f"Creating timestep data for demo_{episode_index}", 
              unit="timestep", leave=False, position=3) as data_pbar:
        for t in range(num_timesteps):
            timestep_data = create_timestep_data(t, actions, rewards, dones, joint_states, episode_index)
            demo_data.append(timestep_data)
            data_pbar.update(1)
            time.sleep(SLEEP_TIME)
    
    # Convert to DataFrame
    df = pd.DataFrame(demo_data)
    
    # Create episode filename (episode000000, episode000001, etc.)
    episode_filename = f"episode_{episode_index:06d}.parquet"
    
    # Use chunk-specific path
    chunk_name = f"chunk-{chunk_index:03d}"
    output_path = os.path.join(output_dir, "data", chunk_name, episode_filename)
    
    # Save to parquet
    # print(f"Saving demo_{episode_index} as {episode_filename} with {len(df)} timesteps...")
    df.to_parquet(output_path, index=False)
    
    # Create episode metadata
    episode_metadata = {
        "episode_index": episode_index,
        "tasks": ["pick up the object and place it in the basket", "valid"],  # This will be overridden by batch processor
        "length": len(df)
    }
    
    # print(f"Successfully saved {len(df)} rows to {output_path}")
    # print(f"  Saved {num_timesteps} agentview images and {num_timesteps} eye_in_hand images")
    
    # Create videos from saved images
    create_episode_videos(episode_index, output_dir, chunk_index)
    
    return episode_metadata


def get_demo_keys(data_group: h5py.Group) -> List[str]:
    """Get all demo keys from the data group."""
    demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
    demo_keys.sort()  # Sort to ensure consistent order
    return demo_keys 