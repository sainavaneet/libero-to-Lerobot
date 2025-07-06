import os
import glob
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from .file_operations import ensure_output_directory
from .hdf5_processor import get_demo_keys, process_single_demo_for_chunk
from .metadata_generator import create_info_json, create_modality_json, create_stats_json
from config import FPS, TIMESTEP_DURATION, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, JOINT_COUNT

def get_hdf5_files(input_dir: str) -> List[str]:
    """Get all HDF5 files from the input directory."""
    pattern = os.path.join(input_dir, "*.hdf5")
    hdf5_files = glob.glob(pattern)
    # Sort files to ensure consistent ordering
    hdf5_files.sort()
    return hdf5_files


def extract_task_name_from_filename(filename: str) -> str:
    """Extract task name from HDF5 filename."""
    # Remove path and extension
    basename = os.path.basename(filename)
    task_name = basename.replace("_demo.hdf5", "")
    return task_name


def create_chunk_directory_structure(base_dir: str, chunk_index: int) -> str:
    """Create directory structure for a specific chunk."""
    chunk_name = f"chunk-{chunk_index:03d}"
    
    dirs = [
        os.path.join(base_dir, "data", chunk_name),
        os.path.join(base_dir, "videos", chunk_name, "observation.images.agentview_rgb"),
        os.path.join(base_dir, "videos", chunk_name, "observation.images.eye_in_hand_rgb"),
        os.path.join(base_dir, "meta"),
        os.path.join(base_dir, "images", "agentview"),
        os.path.join(base_dir, "images", "eye_in_hand")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return chunk_name


def process_single_hdf5_file(hdf5_path: str, output_dir: str, chunk_index: int, 
                            global_episode_index: int, pbar=None) -> Dict[str, Any]:
    """
    Process a single HDF5 file and create a chunk for it.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
        output_dir (str): Base output directory
        chunk_index (int): Index for this chunk
        global_episode_index (int): Starting episode index for this chunk
        pbar: Optional progress bar for updating progress
    
    Returns:
        Dict[str, Any]: Metadata about the processed chunk
    """
    import h5py
    
    # print(f"\n{'='*60}")
    # print(f"Processing chunk {chunk_index:03d}: {os.path.basename(hdf5_path)}")
    # print(f"{'='*60}")
    
    # Create chunk-specific directory structure
    chunk_name = create_chunk_directory_structure(output_dir, chunk_index)
    
    # Extract task name from filename
    task_name = extract_task_name_from_filename(hdf5_path)
    
    # Initialize metadata for this chunk
    episodes_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Access the data group
        data_group = f['data']
        
        # Get all demo keys
        demo_keys = get_demo_keys(data_group)
        
        # print(f"Found {len(demo_keys)} demos in {os.path.basename(hdf5_path)}")
        # print()  # Add spacing before demo progress bar
        
        # Progress bar for demos within this file
        with tqdm(total=len(demo_keys), 
                  unit="demo", leave=False, position=1) as demo_pbar:

            for i, demo_key in enumerate(demo_keys):
                demo_group = data_group[demo_key]
                
                # Process the demo with global episode index
                episode_metadata = process_single_demo_for_chunk(demo_group, global_episode_index + i, output_dir, chunk_index)
                
                # Update episode metadata with the correct task name
                episode_metadata["tasks"] = [task_name, "valid"]
                
                episodes_data.append(episode_metadata)
                
                # Update demo progress bar
                demo_pbar.update(1)
                
                # Update main progress bar description
                if pbar:
                    pbar.set_postfix_str(f"Current: Demo {i+1}/{len(demo_keys)}")
    
    # Return chunk metadata
    chunk_metadata = {
        "chunk_index": chunk_index,
        "chunk_name": chunk_name,
        "hdf5_file": os.path.basename(hdf5_path),
        "task_name": task_name,
        "episodes_count": len(episodes_data),
        "total_frames": sum(ep["length"] for ep in episodes_data),
        "episodes_data": episodes_data,
        "global_episode_start": global_episode_index,
        "global_episode_end": global_episode_index + len(episodes_data) - 1
    }
    
    print(f"✅ Completed chunk {chunk_index:03d} with {len(episodes_data)} episodes")
    print(f"   Global episode range: {global_episode_index} to {global_episode_index + len(episodes_data) - 1}")
    return chunk_metadata


def process_all_hdf5_files(input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Process all HDF5 files in the input directory, creating individual chunks.
    
    Args:
        input_dir (str): Directory containing HDF5 files
        output_dir (str): Directory to save the converted dataset
    
    Returns:
        List[Dict[str, Any]]: Metadata for all processed chunks
    """
    # print(f"Starting batch processing of HDF5 files...")
    # print(f"Input directory: {input_dir}")
    # print(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    ensure_output_directory(output_dir)
    
    # Get all HDF5 files
    hdf5_files = get_hdf5_files(input_dir)
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return []
    
    # print(f"Found {len(hdf5_files)} HDF5 files to process:")
    # for i, file_path in enumerate(hdf5_files):
    #     print(f"  {i+1}. {os.path.basename(file_path)}")
    
    # Process each HDF5 file as a separate chunk with progress bar
    all_chunks_metadata = []
    global_episode_index = 0
    
    # Main progress bar for files
    print("\n" + "="*60)
    print("CONVERTING INTO LEROBOT DATASET FORMAT")
    print("="*60 + "\n")
    
    with tqdm(total=len(hdf5_files), desc="Processing HDF5 files", unit="file", position=0, leave=True) as pbar:

        for chunk_index, hdf5_path in enumerate(hdf5_files):

            try:
                chunk_metadata = process_single_hdf5_file(
                    hdf5_path, output_dir, chunk_index, global_episode_index, pbar
                )
                all_chunks_metadata.append(chunk_metadata)
                
                # Update global episode index for next chunk
                global_episode_index += chunk_metadata["episodes_count"]
                
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(hdf5_path)}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # Update main progress bar
            pbar.update(1)
    
    # Create global metadata combining all chunks
    create_global_metadata(output_dir, all_chunks_metadata)
    
    print(f"\n\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"Successfully processed {len(all_chunks_metadata)} chunks")
    print(f"{'='*60}\n")
    
    return all_chunks_metadata


def create_global_metadata(output_dir: str, chunks_metadata: List[Dict[str, Any]]) -> None:
    """Create global metadata files that combine information from all chunks."""
    
    # Collect all episodes from all chunks
    all_episodes = []
    all_tasks = set()  # Use set to avoid duplicates
    
    for chunk in chunks_metadata:
        # Add episodes from this chunk
        all_episodes.extend(chunk["episodes_data"])
        
        # Add task from this chunk
        task_name = chunk["task_name"]
        all_tasks.add(task_name)
    
    # Convert tasks set to list and add "valid" task
    all_tasks_list = list(all_tasks) + ["valid"]
    
    # Calculate global statistics
    total_episodes = len(all_episodes)
    total_frames = sum(ep["length"] for ep in all_episodes)
    total_chunks = len(chunks_metadata)
    
    # Calculate chunks_size based on the number of episodes in each chunk
    # Since each HDF5 file becomes a chunk, chunks_size should represent the number of demos per chunk
    chunks_size = max(chunk["episodes_count"] for chunk in chunks_metadata) if chunks_metadata else 0
    
    print(f"Creating global metadata...")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total tasks: {len(all_tasks_list)}")
    print(f"  Chunks size (max episodes per chunk): {chunks_size}")
    
    # Create episodes.jsonl
    episodes_path = os.path.join(output_dir, "meta", "episodes.jsonl")
    import json
    with open(episodes_path, 'w') as f:
        for episode in all_episodes:
            f.write(json.dumps(episode) + '\n')
    
    # Create tasks.jsonl
    tasks_path = os.path.join(output_dir, "meta", "tasks.jsonl")
    with open(tasks_path, 'w') as f:
        for i, task in enumerate(all_tasks_list):
            f.write(json.dumps({"task_index": i, "task": task}) + '\n')
    
    # Create global info.json
    global_info = {
        "codebase_version": "v2.0",
        "robot_type": "panda-robot",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(all_tasks_list),
        "total_videos": total_episodes * 2,  # agentview and eye_in_hand
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": FPS,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_info": [
            {
                "chunk_index": chunk["chunk_index"],
                "chunk_name": chunk["chunk_name"],
                "task_name": chunk["task_name"],
                "episodes_count": chunk["episodes_count"],
                "total_frames": chunk["total_frames"],
                "global_episode_start": chunk["global_episode_start"],
                "global_episode_end": chunk["global_episode_end"]
            }
            for chunk in chunks_metadata
        ],
        "features": {
            "observation.images.agentview_rgb": {
                "dtype": "video",
                "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": FPS,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.eye_in_hand_rgb": {
                "dtype": "video",
                "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": FPS,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [JOINT_COUNT],
                "names": [f"motor_{i}" for i in range(JOINT_COUNT)]
            },
            "action": {
                "dtype": "float64",
                "shape": [JOINT_COUNT],
                "names": [f"motor_{i}" for i in range(JOINT_COUNT)]
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
    
    # Save global info.json
    global_info_path = os.path.join(output_dir, "meta", "info.json")
    with open(global_info_path, 'w') as f:
        json.dump(global_info, f, indent=4)
    
    # Create other metadata files
    from .metadata_generator import create_info_json, create_modality_json, create_stats_json
    
    # # Create info.json
    # info_data = create_info_json(all_episodes, all_tasks_list, total_episodes)
    # info_path = os.path.join(output_dir, "meta", "info.json")
    # with open(info_path, 'w') as f:
    #     json.dump(info_data, f, indent=4)
    
    # Create modality.json
    modality_data = create_modality_json()
    modality_path = os.path.join(output_dir, "meta", "modality.json")
    with open(modality_path, 'w') as f:
        json.dump(modality_data, f, indent=4)
    
    # Create stats.json
    stats_data = create_stats_json()
    stats_path = os.path.join(output_dir, "meta", "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=4)
    
    print(f"✅ Created global metadata files:")
    print(f"   episodes.jsonl: {episodes_path}")
    print(f"   tasks.jsonl: {tasks_path}")
    print(f"   global_info.json: {global_info_path}")
    print(f"   modality.json: {modality_path}")
    print(f"   stats.json: {stats_path}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Total episodes: {total_episodes}")
    print(f"   Total frames: {total_frames}")
    print(f"   Total tasks: {len(all_tasks_list)}") 