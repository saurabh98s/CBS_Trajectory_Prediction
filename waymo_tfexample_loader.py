"""
Waymo Data Loader
Loads from preprocessed pkl files or parses TFExample format from tfrecord files
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import config


# Define classes for unpickling pkl files
# Generic class factory for any missing classes
class GenericUnpickler(pickle.Unpickler):
    """Custom unpickler that creates generic classes for missing types"""
    def find_class(self, module, name):
        # For waymo_tfexample_loader module, create generic classes
        if module == 'waymo_tfexample_loader' or module == '__main__':
            # Create a generic class on the fly
            class GenericClass:
                def __init__(self, *args, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                    for i, arg in enumerate(args):
                        setattr(self, f'arg_{i}', arg)
                
                def to_dict(self):
                    return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            
            # Cache the class
            if not hasattr(self, '_class_cache'):
                self._class_cache = {}
            if name not in self._class_cache:
                self._class_cache[name] = type(name, (GenericClass,), {})
            
            return self._class_cache[name]
        
        return super().find_class(module, name)


# Define known classes for unpickling pkl files
class Trajectory:
    """Placeholder class for unpickling"""
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        for i, arg in enumerate(args):
            setattr(self, f'arg_{i}', arg)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class ProcessedScenario:
    """Placeholder class for unpickling"""
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        for i, arg in enumerate(args):
            setattr(self, f'arg_{i}', arg)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class MapData:
    """Placeholder class for unpickling"""
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        for i, arg in enumerate(args):
            setattr(self, f'arg_{i}', arg)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class WaymoDataset(Dataset):
    """
    PyTorch Dataset for Waymo Motion Dataset
    Supports both preprocessed pkl files and raw tfrecord files
    """
    
    def __init__(
        self,
        data_files: List[str] = None,
        pkl_file: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        max_scenarios: Optional[int] = None
    ):
        """
        Initialize Waymo dataset
        
        Args:
            data_files: List of paths to tfrecord files (if pkl not available)
            pkl_file: Path to preprocessed pkl file (preferred)
            cache_dir: Directory to cache preprocessed data
            use_cache: Whether to use cached data if available
            max_scenarios: Maximum number of scenarios to load (None for all)
        """
        self.data_files = data_files or []
        self.pkl_file = pkl_file
        self.cache_dir = cache_dir or config.WAYMO_DATA_CACHE
        self.use_cache = use_cache
        self.max_scenarios = max_scenarios
        
        # Load or process data
        self.scenarios = self._load_data()
        
        print(f"Loaded {len(self.scenarios)} scenarios")
    
    def _load_data(self) -> List[Dict]:
        """Load data from pkl files or tfrecords"""
        # Priority 1: Use preprocessed pkl file if available
        if self.pkl_file and os.path.exists(self.pkl_file):
            print(f"Loading data from pkl file: {self.pkl_file}")
            return self._load_from_pkl(self.pkl_file)
        
        # Priority 2: Check for standard pkl files in data directory
        data_root = config.DATA_ROOT
        if self.pkl_file is None:
            # Try to find pkl files
            if 'training' in str(self.data_files[0] if self.data_files else ''):
                pkl_path = data_root / "training_processed.pkl"
            else:
                pkl_path = data_root / "validation_processed.pkl"
            
            if pkl_path.exists():
                print(f"Found pkl file: {pkl_path}")
                return self._load_from_pkl(str(pkl_path))
        
        # Priority 3: Use cache if available
        cache_file = self.cache_dir / "scenarios_cache.pkl"
        if self.use_cache and cache_file.exists():
            print(f"Loading data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                scenarios = pickle.load(f)
            if self.max_scenarios:
                scenarios = scenarios[:self.max_scenarios]
            return scenarios
        
        # Priority 4: Parse tfrecord files
        if self.data_files:
            print("Processing tfrecord files...")
            scenarios = []
            
            for tfrecord_file in self.data_files:
                if not os.path.exists(tfrecord_file):
                    print(f"Warning: File not found: {tfrecord_file}")
                    continue
                
                file_scenarios = self._parse_tfrecord_file(tfrecord_file)
                scenarios.extend(file_scenarios)
                
                if self.max_scenarios and len(scenarios) >= self.max_scenarios:
                    scenarios = scenarios[:self.max_scenarios]
                    break
            
            # Save to cache
            if self.use_cache and scenarios:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print(f"Saving data to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(scenarios, f)
            
            return scenarios
        
        raise ValueError("No data files or pkl files found!")
    
    def _load_from_pkl(self, pkl_path: str) -> List[Dict]:
        """Load data from preprocessed pkl file"""
        try:
            with open(pkl_path, 'rb') as f:
                # Try with custom unpickler first
                try:
                    unpickler = GenericUnpickler(f)
                    data = unpickler.load()
                except:
                    # Fall back to regular pickle
                    f.seek(0)
                    data = pickle.load(f)
            
            # Handle different pkl file structures
            if isinstance(data, list):
                # List of scenarios
                scenarios = data
            elif isinstance(data, dict):
                # Dictionary with scenarios
                if 'scenarios' in data:
                    scenarios = data['scenarios']
                elif 'data' in data:
                    scenarios = data['data']
                else:
                    # Try to extract scenarios from dict values
                    scenarios = list(data.values()) if data else []
            else:
                raise ValueError(f"Unexpected pkl file structure: {type(data)}")
            
            # Convert to standard format if needed
            formatted_scenarios = []
            for i, scenario in enumerate(scenarios):
                formatted = self._format_scenario(scenario, i)
                if formatted:
                    formatted_scenarios.append(formatted)
            
            if self.max_scenarios:
                formatted_scenarios = formatted_scenarios[:self.max_scenarios]
            
            return formatted_scenarios
        
        except Exception as e:
            print(f"Error loading pkl file {pkl_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _format_scenario(self, scenario, idx: int) -> Optional[Dict]:
        """Format scenario to standard structure"""
        try:
            # Convert ProcessedScenario object to dict if needed
            if hasattr(scenario, 'to_dict'):
                scenario = scenario.to_dict()
            elif hasattr(scenario, '__dict__'):
                scenario = {k: v for k, v in scenario.__dict__.items() if not k.startswith('_')}
            
            # LOG: Print all keys in scenario for first scenario only
            if idx == 0:
                print(f"\n=== SCENARIO {idx} KEYS ===")
                if isinstance(scenario, dict):
                    print(f"Scenario keys: {list(scenario.keys())}")
                    for key in scenario.keys():
                        val = scenario[key]
                        if hasattr(val, '__dict__'):
                            print(f"  {key}: object with attributes {list(val.__dict__.keys())[:10]}")
                        elif isinstance(val, (list, np.ndarray)):
                            print(f"  {key}: {type(val).__name__} of length {len(val)}")
                        elif isinstance(val, dict):
                            print(f"  {key}: dict with keys {list(val.keys())[:10]}")
                        else:
                            print(f"  {key}: {type(val).__name__}")
            
            # Handle ProcessedScenario format: trajectories is dict of Trajectory objects
            if isinstance(scenario, dict) and 'trajectories' in scenario:
                trajectories_dict = scenario['trajectories']
                
                # Convert Trajectory objects to numpy arrays
                agent_trajectories = []
                for agent_id in sorted(trajectories_dict.keys()):
                    traj_obj = trajectories_dict[agent_id]
                    
                    # Convert Trajectory object to dict if needed
                    if hasattr(traj_obj, '__dict__'):
                        traj_dict = {k: v for k, v in traj_obj.__dict__.items() if not k.startswith('_')}
                    elif hasattr(traj_obj, 'to_dict'):
                        traj_dict = traj_obj.to_dict()
                    else:
                        continue
                    
                    # Extract trajectory data
                    positions = traj_dict.get('positions', np.zeros((91, 2), dtype=np.float32))
                    headings = traj_dict.get('headings', np.zeros(91, dtype=np.float32))
                    velocities = traj_dict.get('velocities', np.zeros((91, 2), dtype=np.float32))
                    valid_mask = traj_dict.get('valid_mask', np.ones(91, dtype=bool))
                    
                    # Ensure correct shapes
                    if positions.ndim == 1:
                        positions = positions.reshape(-1, 2)
                    if headings.ndim == 0:
                        headings = np.array([headings])
                    if velocities.ndim == 1:
                        velocities = velocities.reshape(-1, 2)
                    
                    num_timesteps = len(positions)
                    
                    # Compute accelerations from velocities (finite difference)
                    accelerations = np.zeros_like(velocities)
                    if num_timesteps > 1:
                        dt = 0.1  # 10Hz = 0.1s per timestep
                        accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt
                        accelerations[0] = accelerations[1]  # Copy first acceleration
                    
                    # Combine into [x, y, heading, vx, vy, ax, ay] format
                    traj_features = np.zeros((num_timesteps, 7), dtype=np.float32)
                    traj_features[:, 0:2] = positions  # x, y
                    traj_features[:, 2] = headings  # heading
                    traj_features[:, 3:5] = velocities  # vx, vy
                    traj_features[:, 5:7] = accelerations  # ax, ay
                    
                    # Apply valid mask (set invalid timesteps to zero or last valid)
                    if not np.all(valid_mask):
                        for t in range(num_timesteps):
                            if not valid_mask[t]:
                                # Use last valid value
                                prev_valid = t - 1
                                while prev_valid >= 0 and not valid_mask[prev_valid]:
                                    prev_valid -= 1
                                if prev_valid >= 0:
                                    traj_features[t] = traj_features[prev_valid]
                                else:
                                    traj_features[t] = 0
                    
                    agent_trajectories.append(traj_features)
                
                if not agent_trajectories:
                    return None
                
                # Stack into [num_agents, timesteps, 7]
                trajectories = np.stack(agent_trajectories, axis=0)
                
                # Ensure correct number of timesteps (91 = 10 past + 1 current + 80 future)
                expected_timesteps = config.DATA_CONFIG['total_timesteps']
                num_timesteps = trajectories.shape[1]
                
                if num_timesteps < expected_timesteps:
                    # Pad with last value
                    padded = np.zeros((trajectories.shape[0], expected_timesteps, 7), dtype=np.float32)
                    padded[:, :num_timesteps] = trajectories
                    padded[:, num_timesteps:] = trajectories[:, -1:]
                    trajectories = padded
                elif num_timesteps > expected_timesteps:
                    # Trim
                    trajectories = trajectories[:, :expected_timesteps]
                
                # Pad to expected number of agents if needed
                expected_agents = config.DATA_CONFIG['num_agents']
                if trajectories.shape[0] < expected_agents:
                    padded = np.zeros((expected_agents, expected_timesteps, 7), dtype=np.float32)
                    padded[:trajectories.shape[0]] = trajectories
                    trajectories = padded
                elif trajectories.shape[0] > expected_agents:
                    trajectories = trajectories[:expected_agents]
                
                # Convert map_data to dict format for visualization
                map_data = scenario.get('map_data')
                if idx == 0:
                    print(f"\n=== MAP DATA (scenario {idx}) ===")
                    print(f"map_data type: {type(map_data)}")
                    if map_data is not None:
                        if hasattr(map_data, '__dict__'):
                            print(f"map_data attributes: {list(map_data.__dict__.keys())}")
                        elif isinstance(map_data, dict):
                            print(f"map_data dict keys: {list(map_data.keys())}")
                
                if map_data is not None:
                    if hasattr(map_data, '__dict__'):
                        map_data = {k: v for k, v in map_data.__dict__.items() if not k.startswith('_')}
                    elif hasattr(map_data, 'to_dict'):
                        map_data = map_data.to_dict()
                    # Keep map_data for visualization (convert numpy arrays to lists if needed)
                    if isinstance(map_data, dict):
                        if idx == 0:
                            print(f"Processed map_data keys: {list(map_data.keys())}")
                        for key, value in map_data.items():
                            if isinstance(value, np.ndarray):
                                map_data[key] = value.tolist() if value.size < 10000 else value  # Keep large arrays as numpy
                    if idx == 0:
                        print(f"Final map_data: {type(map_data)}, keys: {list(map_data.keys()) if isinstance(map_data, dict) else 'N/A'}")
                
                # LOG: Check trajectories
                if idx == 0:
                    print(f"\n=== TRAJECTORIES (scenario {idx}) ===")
                    print(f"Number of agents: {trajectories.shape[0]}")
                    print(f"Number of timesteps: {trajectories.shape[1]}")
                    print(f"Trajectory shape: {trajectories.shape}")
                
                return {
                    'scenario_id': scenario.get('scenario_id', f'scenario_{idx}'),
                    'trajectories': trajectories,
                    'ego_agent_id': scenario.get('ego_agent_id', 0),  # Store ego agent ID
                    'map_data': map_data,  # Set to None to avoid pickling issues
                }
            
            # Handle other formats (legacy support)
            elif isinstance(scenario, dict) and 'history' in scenario:
                # Legacy format with history and future
                history = np.array(scenario['history'])
                future = np.array(scenario['future'])
                
                # Combine and format
                if history.ndim == 2:
                    history = history.reshape(1, -1, history.shape[-1])
                if future.ndim == 2:
                    future = future.reshape(1, -1, future.shape[-1])
                
                # This would need more processing...
                return None
            
            return None
        
        except Exception as e:
            print(f"Error formatting scenario {idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_tfrecord_file(self, tfrecord_file: str) -> List[Dict]:
        """Parse a single tfrecord file with better error handling"""
        scenarios = []
        error_count = 0
        max_errors = 100  # Stop after too many errors
        
        try:
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            
            for raw_data in dataset:
                if error_count >= max_errors:
                    print(f"Too many errors in {tfrecord_file}, skipping remaining examples")
                    break
                
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_data.numpy())
                    
                    scenario = self._parse_example(example, len(scenarios))
                    if scenario is not None:
                        scenarios.append(scenario)
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:  # Only print first 10 errors
                        pass  # Silent skip
        
        except Exception as e:
            print(f"Error parsing {tfrecord_file}: {e}")
        
        if error_count > 0:
            print(f"Skipped {error_count} invalid examples from {tfrecord_file}")
        
        return scenarios
    
    def _parse_example(self, example: tf.train.Example, idx: int) -> Optional[Dict]:
        """Parse a single TFExample with robust error handling"""
        try:
            # Extract scenario ID (with fallback)
            scenario_id = f'scenario_{idx}'
            if 'scenario_id' in example.features.feature:
                feature = example.features.feature['scenario_id']
                if feature.bytes_list.value:
                    try:
                        scenario_id = feature.bytes_list.value[0].decode('utf-8')
                    except:
                        pass
            
            # Extract all features
            feature_dict = {}
            
            for key in example.features.feature.keys():
                feature = example.features.feature[key]
                
                try:
                    if feature.float_list.value:
                        feature_dict[key] = np.array(feature.float_list.value, dtype=np.float32)
                    elif feature.bytes_list.value:
                        # Try to decode as float array
                        try:
                            feature_dict[key] = np.frombuffer(
                                feature.bytes_list.value[0], dtype=np.float32
                            )
                        except:
                            # Keep as bytes
                            feature_dict[key] = feature.bytes_list.value
                except:
                    continue
            
            # Extract trajectories
            trajectories = self._extract_trajectories(feature_dict)
            
            if trajectories is None or trajectories.size == 0:
                return None
            
            return {
                'scenario_id': scenario_id,
                'trajectories': trajectories,
                'map_data': None,
            }
        
        except Exception as e:
            return None  # Silent skip
    
    def _extract_trajectories(self, feature_dict: Dict) -> Optional[np.ndarray]:
        """Extract agent trajectories from feature dictionary"""
        # Try to find trajectory data in various formats
        num_agents = config.DATA_CONFIG['num_agents']
        num_timesteps = config.DATA_CONFIG['total_timesteps']
        
        # Initialize with zeros
        trajectories = np.zeros((num_agents, num_timesteps, 7), dtype=np.float32)
        
        # Try common feature names
        # This is a placeholder - adjust based on actual Waymo format
        # You may need to inspect your tfrecord files to find exact feature names
        
        return trajectories
    
    def __len__(self) -> int:
        return len(self.scenarios)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single scenario"""
        scenario = self.scenarios[idx]
        
        # Extract trajectories
        trajectories = scenario['trajectories']  # [num_agents, timesteps, 7]
        
        # Split into past, current, and future
        past_length = config.DATA_CONFIG['past_timesteps']
        current_length = config.DATA_CONFIG['current_timesteps']
        future_length = config.DATA_CONFIG['future_timesteps']
        current_timestep = past_length  # Timestep 10 is the "current" moment
        
        # Use ego agent if available, otherwise first valid agent
        ego_agent_id = scenario.get('ego_agent_id', 0)
        agent_idx = ego_agent_id
        
        # Ensure agent_idx is valid
        if agent_idx >= trajectories.shape[0]:
            # Fall back to first valid agent
            agent_idx = 0
            for i in range(min(trajectories.shape[0], config.DATA_CONFIG['num_agents'])):
                if np.any(np.abs(trajectories[i, :, 0]) > 1e-6):
                    agent_idx = i
                    break
        
        agent_traj = trajectories[agent_idx].copy()  # [timesteps, 7]
        
        # Ensure we have enough timesteps
        total_needed = past_length + current_length + future_length
        if agent_traj.shape[0] < total_needed:
            # Pad with last value
            padded = np.zeros((total_needed, 7), dtype=np.float32)
            padded[:agent_traj.shape[0]] = agent_traj
            padded[agent_traj.shape[0]:] = agent_traj[-1]
            agent_traj = padded
        
        # CRITICAL: Normalize coordinates relative to current position
        # This transforms to a local coordinate system centered at timestep 10
        current_pos = agent_traj[current_timestep, :2].copy()  # [x, y] at current timestep
        current_heading = agent_traj[current_timestep, 2]  # heading at current timestep
        
        # Normalize positions: subtract current position and rotate by -current_heading
        # This makes the current position (0, 0) with heading 0
        positions = agent_traj[:, :2] - current_pos  # Translate to origin
        
        # Rotate to align current heading with x-axis
        cos_h = np.cos(-current_heading)
        sin_h = np.sin(-current_heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        positions_rotated = positions @ rotation_matrix.T
        
        # Normalize headings relative to current heading
        headings = agent_traj[:, 2] - current_heading
        # Normalize to [-pi, pi]
        headings = np.arctan2(np.sin(headings), np.cos(headings))
        
        # Velocities are already in global frame, need to rotate them too
        velocities_global = agent_traj[:, 3:5]  # [vx, vy] in global frame
        velocities_rotated = velocities_global @ rotation_matrix.T
        
        # Accelerations (already computed, rotate them too)
        accelerations_global = agent_traj[:, 5:7]  # [ax, ay] in global frame
        accelerations_rotated = accelerations_global @ rotation_matrix.T
        
        # Create normalized trajectory
        normalized_traj = np.zeros_like(agent_traj)
        normalized_traj[:, 0:2] = positions_rotated
        normalized_traj[:, 2] = headings
        normalized_traj[:, 3:5] = velocities_rotated
        normalized_traj[:, 5:7] = accelerations_rotated
        
        # Split timeline
        history = normalized_traj[:past_length]  # [past_length, 7]
        current = normalized_traj[past_length:past_length + current_length]  # [current_length, 7]
        future = normalized_traj[past_length + current_length:past_length + current_length + future_length]  # [future_length, 7]
        
        # Extract positions for ground truth (x, y only) - already normalized
        future_xy = future[:, :2]  # [future_length, 2]
        
        # Extract features for model input - already normalized
        history_features = history  # [past_length, 7]
        
        # Normalize all agent trajectories for visualization
        all_agent_trajs_normalized = []
        num_valid_agents = 0
        for i in range(min(trajectories.shape[0], config.DATA_CONFIG['num_agents'])):
            if i == agent_idx:
                all_agent_trajs_normalized.append(normalized_traj)
                num_valid_agents += 1
            else:
                other_traj = trajectories[i].copy()
                # Check if agent has valid data (not all zeros)
                if np.any(np.abs(other_traj[:, :2]) > 1e-3):
                    num_valid_agents += 1
                    if other_traj.shape[0] < total_needed:
                        padded = np.zeros((total_needed, 7), dtype=np.float32)
                        padded[:other_traj.shape[0]] = other_traj
                        padded[other_traj.shape[0]:] = other_traj[-1] if other_traj.shape[0] > 0 else 0
                        other_traj = padded
                    
                    # Normalize using same transformation as ego agent
                    other_positions = other_traj[:, :2] - current_pos
                    other_positions_rotated = other_positions @ rotation_matrix.T
                    other_headings = other_traj[:, 2] - current_heading
                    other_headings = np.arctan2(np.sin(other_headings), np.cos(other_headings))
                    other_velocities_rotated = other_traj[:, 3:5] @ rotation_matrix.T
                    other_accelerations_rotated = other_traj[:, 5:7] @ rotation_matrix.T
                    
                    other_normalized = np.zeros_like(other_traj)
                    other_normalized[:, 0:2] = other_positions_rotated
                    other_normalized[:, 2] = other_headings
                    other_normalized[:, 3:5] = other_velocities_rotated
                    other_normalized[:, 5:7] = other_accelerations_rotated
                    all_agent_trajs_normalized.append(other_normalized)
                else:
                    # Invalid agent, add zeros
                    other_normalized = np.zeros((total_needed, 7), dtype=np.float32)
                    all_agent_trajs_normalized.append(other_normalized)
        
        if idx == 0:
            print(f"\n=== AGENT TRAJECTORIES (scenario {idx}) ===")
            print(f"Total agents in trajectories: {trajectories.shape[0]}")
            print(f"Valid agents (non-zero): {num_valid_agents}")
            print(f"Normalized trajectories shape: {np.array(all_agent_trajs_normalized).shape}")
        
        # Normalize map_data to local coordinate system
        normalized_map_data = None
        raw_map_data = scenario.get('map_data')
        if raw_map_data is not None:
            try:
                # Convert to dict if needed
                if hasattr(raw_map_data, '__dict__'):
                    map_dict = {k: v for k, v in raw_map_data.__dict__.items() if not k.startswith('_')}
                elif hasattr(raw_map_data, 'to_dict'):
                    map_dict = raw_map_data.to_dict()
                elif isinstance(raw_map_data, dict):
                    map_dict = raw_map_data
                else:
                    map_dict = None
                
                if map_dict is not None:
                    normalized_map_data = {}
                    # Normalize lanes, road_edges, crosswalks
                    for map_key in ['lanes', 'road_edges', 'crosswalks']:
                        if map_key in map_dict:
                            map_features = map_dict[map_key]
                            if isinstance(map_features, (list, np.ndarray)):
                                normalized_features = []
                                for feature in map_features:
                                    if isinstance(feature, np.ndarray) and feature.ndim == 2 and feature.shape[1] >= 2:
                                        # Normalize feature points using same transformation as trajectories
                                        feature_positions = feature[:, :2] - current_pos
                                        feature_rotated = feature_positions @ rotation_matrix.T
                                        normalized_features.append(feature_rotated)
                                    elif isinstance(feature, list) and len(feature) > 0:
                                        feature_array = np.array(feature)
                                        if feature_array.ndim == 2 and feature_array.shape[1] >= 2:
                                            feature_positions = feature_array[:, :2] - current_pos
                                            feature_rotated = feature_positions @ rotation_matrix.T
                                            normalized_features.append(feature_rotated)
                                normalized_map_data[map_key] = normalized_features
                            else:
                                normalized_map_data[map_key] = map_features
                    
                    # Copy other map fields as-is
                    for key, value in map_dict.items():
                        if key not in ['lanes', 'road_edges', 'crosswalks']:
                            normalized_map_data[key] = value
            except Exception as e:
                print(f"Warning: Could not normalize map_data: {e}")
                normalized_map_data = raw_map_data
        
        return {
            'scenario_id': scenario.get('scenario_id', f'scenario_{idx}'),
            'history': torch.FloatTensor(history_features),
            'current': torch.FloatTensor(current),
            'future': torch.FloatTensor(future_xy),
            'full_trajectory': torch.FloatTensor(normalized_traj),
            'agent_idx': agent_idx,
            'map_data': normalized_map_data,  # Now normalized to local coordinates
            'all_agent_trajectories': np.array(all_agent_trajs_normalized),  # [num_agents, timesteps, 7]
        }


def get_data_loaders(
    data_root: Path = None,
    batch_size: int = None,
    train_split: float = 0.8,
    max_scenarios: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        data_root: Root directory containing data files
        batch_size: Batch size for DataLoader
        train_split: Fraction of data for training
        max_scenarios: Maximum scenarios to load
    
    Returns:
        train_loader, val_loader
    """
    if data_root is None:
        data_root = config.DATA_ROOT
    
    if batch_size is None:
        batch_size = config.TRAIN_CONFIG['batch_size']
    
    # Check for pkl files first
    train_pkl = data_root / "training_processed.pkl"
    val_pkl = data_root / "validation_processed.pkl"
    
    # Create datasets
    if train_pkl.exists():
        print(f"Using training pkl file: {train_pkl}")
        train_dataset = WaymoDataset(pkl_file=str(train_pkl), max_scenarios=max_scenarios)
    else:
        # Fall back to tfrecord files
        tfrecord_files = sorted(list(data_root.glob("*.tfrecord*")))
        train_files = [str(f) for f in tfrecord_files if 'training' in f.name.lower()]
        if not train_files:
            # Split manually
            split_idx = int(len(tfrecord_files) * train_split)
            train_files = [str(f) for f in tfrecord_files[:split_idx]]
        
        if not train_files:
            raise FileNotFoundError(f"No training data files found in {data_root}")
        
        print(f"Using {len(train_files)} training tfrecord files")
        train_dataset = WaymoDataset(
            data_files=train_files,
            use_cache=config.DATA_CONFIG['cache_preprocessed'],
            max_scenarios=max_scenarios
        )
    
    if val_pkl.exists():
        print(f"Using validation pkl file: {val_pkl}")
        val_dataset = WaymoDataset(pkl_file=str(val_pkl), max_scenarios=max_scenarios // 4 if max_scenarios else None)
    else:
        # Fall back to tfrecord files
        tfrecord_files = sorted(list(data_root.glob("*.tfrecord*")))
        val_files = [str(f) for f in tfrecord_files if 'validation' in f.name.lower()]
        if not val_files:
            # Split manually
            split_idx = int(len(tfrecord_files) * train_split)
            val_files = [str(f) for f in tfrecord_files[split_idx:]]
        
        if not val_files:
            raise FileNotFoundError(f"No validation data files found in {data_root}")
        
        print(f"Using {len(val_files)} validation tfrecord files")
        val_dataset = WaymoDataset(
            data_files=val_files,
            use_cache=config.DATA_CONFIG['cache_preprocessed'],
            max_scenarios=max_scenarios // 4 if max_scenarios else None
        )
    
    # Create data loaders
    # Use num_workers=0 on Windows to avoid multiprocessing pickling issues
    import platform
    use_multiprocessing = platform.system() != 'Windows'
    num_workers = config.DATA_CONFIG['num_workers'] if use_multiprocessing else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.DATA_CONFIG['pin_memory'] if torch.cuda.is_available() and num_workers > 0 else False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.DATA_CONFIG['pin_memory'] if torch.cuda.is_available() and num_workers > 0 else False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader"""
    # Stack tensors
    history = torch.stack([item['history'] for item in batch])
    current = torch.stack([item['current'] for item in batch])
    future = torch.stack([item['future'] for item in batch])
    
    scenario_ids = [item['scenario_id'] for item in batch]
    agent_indices = [item['agent_idx'] for item in batch]
    
    return {
        'history': history,
        'current': current,
        'future': future,
        'scenario_ids': scenario_ids,
        'agent_indices': agent_indices,
    }


if __name__ == "__main__":
    # Test data loading
    print("Testing Waymo Data Loader...")
    
    try:
        train_loader, val_loader = get_data_loaders(max_scenarios=10)
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"\nBatch shapes:")
            print(f"  History: {batch['history'].shape}")
            print(f"  Current: {batch['current'].shape}")
            print(f"  Future: {batch['future'].shape}")
            break
        
        print("\nData loading test completed successfully!")
    
    except Exception as e:
        print(f"Error during data loading test: {e}")
        import traceback
        traceback.print_exc()
