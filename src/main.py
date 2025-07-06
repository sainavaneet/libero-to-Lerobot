import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import os

def extract_hdf5_to_parquet(hdf5_path, output_path):
    """
    Extract all demo data from HDF5 file and save as parquet file.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
        output_path (str): Path for the output parquet file
    """
    print(f"Reading HDF5 file: {hdf5_path}")
    
    # List to store all demo data
    all_demos = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Access the data group
        data_group = f['data']
        
        # Get all demo keys (demo_0, demo_1, ..., demo_49)
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        demo_keys.sort()  # Sort to ensure consistent order
        
        print(f"Found {len(demo_keys)} demos: {demo_keys}")
        
        for demo_key in demo_keys:
            demo_group = data_group[demo_key]
            print(f"Processing {demo_key}...")
            
            # Get the data for this demo
            actions = demo_group['actions'][:]
            dones = demo_group['dones'][:]
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
            
            # Create data for each timestep
            for t in range(num_timesteps):
                demo_data = {
                    'demo_id': demo_key,
                    'timestep': t,
                    'action': actions[t].tolist(),
                    'done': bool(dones[t]),
                    'agentview_rgb': agentview_rgb[t].tolist(),
                    'ee_ori': ee_ori[t].tolist(),
                    'ee_pos': ee_pos[t].tolist(),
                    'ee_states': ee_states[t].tolist(),
                    'eye_in_hand_rgb': eye_in_hand_rgb[t].tolist(),
                    'gripper_states': gripper_states[t].tolist(),
                    'joint_states': joint_states[t].tolist()
                }
                all_demos.append(demo_data)
    
    # Convert to DataFrame
    print(f"Creating DataFrame with {len(all_demos)} total timesteps...")
    df = pd.DataFrame(all_demos)
    
    # Save to parquet
    print(f"Saving to parquet file: {output_path}")
    df.to_parquet(output_path, index=False)
    
    print(f"Successfully saved {len(df)} rows to {output_path}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def main():
    # Define paths
    hdf5_file = "/home/navaneet/libero-to-Lerobot/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    output_file = "/home/navaneet/libero-to-Lerobot/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.parquet"
    
    # Check if input file exists
    if not os.path.exists(hdf5_file):
        print(f"Error: HDF5 file not found at {hdf5_file}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and convert data
    try:
        df = extract_hdf5_to_parquet(hdf5_file, output_file)
        print("Conversion completed successfully!")
        
        # Print some statistics
        print("\nData Statistics:")
        print(f"Total demos: {df['demo_id'].nunique()}")
        print(f"Total timesteps: {len(df)}")
        print(f"Average timesteps per demo: {len(df) / df['demo_id'].nunique():.2f}")
        
        # Show sample of the data
        print("\nSample data (first 3 rows):")
        print(df.head(3))
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
