"""
Quick script to inspect what's in the pkl files
"""
import pickle
import numpy as np
from pathlib import Path
import config

def inspect_pkl(pkl_path: Path):
    """Inspect a pkl file and print its structure"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {pkl_path}")
    print(f"{'='*60}")
    
    if not pkl_path.exists():
        print(f"File does not exist!")
        return
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nData type: {type(data)}")
        
        if isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                print(f"\nFirst item type: {type(data[0])}")
                first = data[0]
                if hasattr(first, '__dict__'):
                    print(f"First item attributes: {list(first.__dict__.keys())}")
                    for key in list(first.__dict__.keys())[:20]:
                        val = getattr(first, key)
                        print(f"  {key}: {type(val)}")
                        if isinstance(val, (list, np.ndarray)):
                            print(f"    Length/Shape: {len(val) if isinstance(val, list) else val.shape}")
                        elif isinstance(val, dict):
                            print(f"    Dict keys: {list(val.keys())[:10]}")
                            if key == 'trajectories' and len(val) > 0:
                                first_agent = list(val.values())[0]
                                if hasattr(first_agent, '__dict__'):
                                    print(f"    First agent trajectory attributes: {list(first_agent.__dict__.keys())}")
                        elif hasattr(val, '__dict__'):
                            print(f"    Object attributes: {list(val.__dict__.keys())}")
                            if key == 'map_data':
                                for map_key in list(val.__dict__.keys())[:20]:
                                    map_val = getattr(val, map_key)
                                    print(f"      map_data.{map_key}: {type(map_val)}")
                                    if isinstance(map_val, (list, np.ndarray)):
                                        print(f"        Length/Shape: {len(map_val) if isinstance(map_val, list) else map_val.shape}")
                                        if isinstance(map_val, list) and len(map_val) > 0:
                                            first_item = map_val[0]
                                            print(f"        First item type: {type(first_item)}")
                                            if isinstance(first_item, (list, np.ndarray)):
                                                print(f"          First item shape: {len(first_item) if isinstance(first_item, list) else first_item.shape}")
                                                if isinstance(first_item, np.ndarray) and first_item.ndim == 2:
                                                    print(f"          First item sample: {first_item[:3] if len(first_item) > 3 else first_item}")
                                    elif isinstance(map_val, dict):
                                        print(f"        Dict keys: {list(map_val.keys())[:10]}")
                elif isinstance(first, dict):
                    print(f"First item keys: {list(first.keys())}")
                    for key in list(first.keys())[:20]:
                        val = first[key]
                        print(f"  {key}: {type(val)}")
                        if isinstance(val, (list, np.ndarray)):
                            print(f"    Length/Shape: {len(val) if isinstance(val, list) else val.shape}")
                        elif isinstance(val, dict):
                            print(f"    Dict keys: {list(val.keys())[:10]}")
        
        elif isinstance(data, dict):
            print(f"Dict keys: {list(data.keys())}")
            for key, value in list(data.items())[:10]:
                print(f"  {key}: {type(value)}")
                if isinstance(value, (list, np.ndarray)):
                    print(f"    Length/Shape: {len(value) if isinstance(value, list) else value.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    data_root = config.DATA_ROOT
    val_pkl = data_root / "validation_processed.pkl"
    train_pkl = data_root / "training_processed.pkl"
    
    if val_pkl.exists():
        inspect_pkl(val_pkl)
    elif train_pkl.exists():
        inspect_pkl(train_pkl)
    else:
        print(f"Neither {val_pkl} nor {train_pkl} exists!")

