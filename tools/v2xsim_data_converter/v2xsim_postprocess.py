import os
import json
import copy
import argparse
import pickle
import mmcv
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from nuscenes.nuscenes import NuScenes

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def process_sequence_data(
    input_pkl_path: str,
    output_pkl_path: str,
    dataset_root: str,
    nusc_version: str = "v1.0-trainval",
    verbose: bool = True
) -> dict:
    """
    Process sequential data and apply valid flag filtering.
    
    Args:
        input_pkl_path: Path to the input pkl file.
        output_pkl_path: Path to save the output pkl file.
        dataset_root: NuScenes dataset root directory.
        nusc_version: Dataset version, default 'v1.0-trainval'.
        verbose: Whether to display progress bar, default True.
    """
    # Load raw data
    data_infos = mmcv.load(input_pkl_path)
    sequence_data = data_infos["infos"]
    
    # Initialize NuScenes dataset object
    nusc = NuScenes(version=nusc_version, dataroot=dataset_root)
    
    # Container for processed data
    sequence_data_new = []
    dict_sample_token2list_instance_token = {}
    
    # Configure progress bar
    iterator = tqdm(sequence_data, desc=f"Processing frames ({os.path.basename(input_pkl_path)})") if verbose else sequence_data
    
    # Process frame by frame
    for frame_data in iterator:
        # Update LiDAR paths (replacing sweeps with sweeps_pcd)
        frame_data["lidar_path"] = frame_data["lidar_path"].replace("sweeps", "sweeps_pcd").replace(".bin", "")
        if "lidar_path_1delay" in frame_data:
            frame_data["lidar_path_1delay"] = frame_data["lidar_path_1delay"].replace("sweeps", "sweeps_pcd").replace(".bin", "")
        if "lidar_path_2delay" in frame_data:
            frame_data["lidar_path_2delay"] = frame_data["lidar_path_2delay"].replace("sweeps", "sweeps_pcd").replace(".bin", "")
        
        # Apply valid flag filtering
        mask = copy.deepcopy(frame_data["valid_flag"])
        filter_fields = [
            "gt_boxes", "gt_names", "gt_velocity",
            "num_lidar_pts", "num_radar_pts",
            "valid_flag", "gt_inds", "gt_ins_tokens",
            "fut_traj", "fut_traj_valid_mask",
            "visibility_tokens"
        ]
        
        # Batch process field filtering
        for field in filter_fields:
            if field in frame_data and len(frame_data[field]) == len(mask):
                frame_data[field] = frame_data[field][mask]
            else:
                raise KeyError(f"Invalid field {field} or dimension mismatch in token: {frame_data['token']}")
        
        sequence_data_new.append(frame_data)
        dict_sample_token2list_instance_token[frame_data["token"]] = frame_data["gt_ins_tokens"]
    
    # Update and save data
    data_infos["infos"] = sequence_data_new
    mmcv.dump(data_infos, output_pkl_path)

    return dict_sample_token2list_instance_token


def filter_sample_annotations(
    input_anno_path,
    output_anno_path,
    sample_token_dict,
    verbose=True
):
    """
    Strictly filter sample annotation data.
    
    Args:
        input_anno_path: Path to input sample_annotation.json.
        output_anno_path: Path to save the filtered json.
        sample_token_dict: Dictionary mapping sample_token to a list of valid instance_tokens.
        verbose: Whether to display progress bar.
        
    Returns:
        Dictionary of filtering statistics.
    """
    # ==================== Data Preprocessing ====================
    # Convert dictionary values to sets to improve query performance
    validation_map = {
        sample_token: set(instance_tokens)
        for sample_token, instance_tokens in sample_token_dict.items()
    }
    
    # ==================== Load Raw Data ====================
    with open(input_anno_path, 'r') as f:
        annotations = json.load(f)
    
    # ==================== Execute Filtering ====================
    stats = defaultdict(int)
    filtered = []
    
    iterator = tqdm(annotations, desc="Filtering annotations") if verbose else annotations
    
    for anno in iterator:
        stats['total'] += 1
        
        # Check field existence
        if 'sample_token' not in anno or 'instance_token' not in anno:
            stats['missing_field'] += 1
            continue
            
        sample_token = anno['sample_token']
        instance_token = anno['instance_token']
        
        # Sample-level filtering
        if sample_token not in validation_map:
            stats['invalid_sample'] += 1
            continue
            
        # Instance-level filtering
        if instance_token not in validation_map[sample_token]:
            stats['invalid_instance'] += 1
            continue
            
        # Passed all checks
        filtered.append(anno)
        stats['kept'] += 1
    
    # ==================== Secondary Filtering: Clean prev/next pointers ====================
    if filtered:
        # Build set of valid tokens
        valid_tokens = {anno["token"] for anno in filtered}
        
        # Clean invalid pointers
        stats['invalid_prev'] = 0
        stats['invalid_next'] = 0
        for anno in filtered:
            current_prev = anno.get("prev", "")
            if current_prev and current_prev not in valid_tokens:
                anno["prev"] = ""
                stats['invalid_prev'] += 1
            current_next = anno.get("next", "")
            if current_next and current_next not in valid_tokens:
                anno["next"] = ""
                stats['invalid_next'] += 1
    
    # ==================== Save Results ====================
    with open(output_anno_path, 'w') as f:
        json.dump(filtered, f, indent=4)
    
    return dict(stats)

if __name__ == '__main__':
    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(f'{current_folder_path}/../..')
    
    parser = argparse.ArgumentParser(description="Post-process V2X-Sim sequence data and annotations.")
    parser.add_argument('dataset_name', type=str, help="Target dataset name (e.g., V2X-Sim-full-id1)")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_root = f"datasets/{dataset_name}"
    info_dir = f"data/infos/{dataset_name}"
    anno_path = f"{dataset_root}/v1.0-trainval/sample_annotation.json"

    print(f"========================================================")
    print(f"Starting Post-Processing for: {dataset_name}")
    print(f"========================================================")

    # 1. Process Train set
    dict_train = process_sequence_data(
        input_pkl_path=f"{info_dir}/nuscenes_infos_temporal_train.pkl",
        output_pkl_path=f"{info_dir}/nuscenes_infos_temporal_train.pkl",
        dataset_root=dataset_root
    )

    # 2. Process Val set
    dict_val = process_sequence_data(
        input_pkl_path=f"{info_dir}/nuscenes_infos_temporal_val.pkl",
        output_pkl_path=f"{info_dir}/nuscenes_infos_temporal_val.pkl",
        dataset_root=dataset_root
    )

    # 3. Combine dictionaries for annotation filtering
    dict_train.update(dict_val)

    # 4. Filter sample_annotation.json
    stats = filter_sample_annotations(
        input_anno_path=anno_path,
        output_anno_path=anno_path,
        sample_token_dict=dict_train
    )
    
    print("\nPost-Processing Complete!")
    print(f"Annotation Filtering Stats: {stats}")