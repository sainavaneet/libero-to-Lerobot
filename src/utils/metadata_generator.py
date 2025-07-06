import os
import json
from typing import Dict, List, Any


def create_info_json(episodes_data: List[Dict], task_descriptions: List[str], total_episodes: int) -> Dict[str, Any]:
    """Create the info.json metadata structure."""
    return {
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


def create_modality_json() -> Dict[str, Any]:
    """Create the modality.json metadata structure."""
    return {
    "state": {
      "joints": {
        "start": 0,
        "end": 7
      }
    },
    "action": {
      "joints": {
        "start": 0,
        "end": 7
      }
    },
    "video": {
      "agentview_rgb": {
        "original_key": "observation.images.agentview_rgb"
      },
      "eye_in_hand_rgb": {
        "original_key": "observation.images.eye_in_hand_rgb"
      }
    },
    "annotation": {
      "human.action.task_description": {},
      "human.validity": {}
    }
  }


def create_stats_json() -> Dict[str, Any]:
    """Create the stats.json metadata structure."""
    return {
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