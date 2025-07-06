import h5py
import numpy as np

def explore_hdf5_structure(file_path):
    """Explore the structure of an HDF5 file"""
    print(f"Exploring HDF5 file: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"File keys: {list(f.keys())}")
        
        for key in f.keys():
            print(f"\n--- {key} ---")
            item = f[key]
            
            if isinstance(item, h5py.Group):
                print(f"Group: {key}")
                print(f"  Subkeys: {list(item.keys())}")
                
                for subkey in item.keys():
                    subitem = item[subkey]
                    if isinstance(subitem, h5py.Dataset):
                        print(f"    Dataset: {subkey}, Shape: {subitem.shape}, Dtype: {subitem.dtype}")
                    elif isinstance(subitem, h5py.Group):
                        print(f"    Subgroup: {subkey}")
                        print(f"      Subgroup keys: {list(subitem.keys())}")
                        
                        for subsubkey in subitem.keys():
                            subsubitem = subitem[subsubkey]
                            if isinstance(subsubitem, h5py.Dataset):
                                print(f"        Dataset: {subsubkey}, Shape: {subsubitem.shape}, Dtype: {subsubitem.dtype}")
            elif isinstance(item, h5py.Dataset):
                print(f"Dataset: {key}, Shape: {item.shape}, Dtype: {item.dtype}")
                if item.shape[0] < 10:  # Only show small datasets
                    print(f"  Sample data: {item[:]}")

if __name__ == "__main__":
    hdf5_file = "/home/navaneet/libero-to-Lerobot/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    explore_hdf5_structure(hdf5_file) 