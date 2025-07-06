import h5py
import numpy as np

def explore_obs_structure(file_path):
    """Explore the obs group structure in detail"""
    print(f"Exploring obs structure in: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        data_group = f['data']
        demo_0 = data_group['demo_0']
        
        print(f"Demo 0 keys: {list(demo_0.keys())}")
        
        # Explore obs group
        obs_group = demo_0['obs']
        print(f"\nObs group keys: {list(obs_group.keys())}")
        
        for key in obs_group.keys():
            item = obs_group[key]
            if isinstance(item, h5py.Dataset):
                print(f"  Dataset: {key}, Shape: {item.shape}, Dtype: {item.dtype}")
                if len(item.shape) == 3 and item.shape[0] < 5:  # Show sample for image data
                    print(f"    Sample data shape: {item[0].shape}")
                elif len(item.shape) == 2 and item.shape[0] < 5:  # Show sample for vector data
                    print(f"    Sample data: {item[0]}")
            elif isinstance(item, h5py.Group):
                print(f"  Group: {key}")
                print(f"    Subkeys: {list(item.keys())}")
                
                for subkey in item.keys():
                    subitem = item[subkey]
                    if isinstance(subitem, h5py.Dataset):
                        print(f"      Dataset: {subkey}, Shape: {subitem.shape}, Dtype: {subitem.dtype}")

if __name__ == "__main__":
    hdf5_file = "/home/navaneet/libero-to-Lerobot/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    explore_obs_structure(hdf5_file) 