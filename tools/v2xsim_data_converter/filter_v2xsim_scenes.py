import os
import json
import argparse
import pickle
import mmcv
from tqdm import tqdm

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def write_json(new_dict, path_json):
    with open(path_json, "w") as dump_f:
        json.dump(new_dict, dump_f, indent=4)

def process_sequence_data(
    input_pkl_path: str,
    output_pkl_path: str,
    input_json_path: str,
    verbose: bool = True
) -> None:
    """
    Process sequential data and filter frames based on valid scene annotations.
    
    Args:
        input_pkl_path: Path to the input .pkl file containing sequence data.
        output_pkl_path: Path to save the filtered .pkl file.
        input_json_path: Path to the .json file mapping valid scenes.
        verbose: Whether to display the progress bar, default True.
    """
    # Load raw data
    data_infos = mmcv.load(input_pkl_path)
    sequence_data = data_infos["infos"]

    # Load valid scenes mapping
    dict_valid_scene2info = load_json(input_json_path)
    list_valid_scenes = list(dict_valid_scene2info.keys())
    
    if verbose:
        print(f"Loaded {len(list_valid_scenes)} valid scenes from {os.path.basename(input_json_path)}")
    
    # Container for processed data
    sequence_data_new = []
    
    # Configure progress bar
    iterator = tqdm(sequence_data, desc=f"Filtering {os.path.basename(input_pkl_path)}") if verbose else sequence_data
    
    # Process frame by frame
    for frame_data in iterator:
        # Extract scene ID from LiDAR path
        scene_id = f'scene_{frame_data["lidar_path"].split("/")[-1].split("_")[1]}'
        if scene_id in list_valid_scenes:
            sequence_data_new.append(frame_data)
    
    # Update and save data
    data_infos["infos"] = sequence_data_new
    if verbose:
        print(f"Frames before filtering: {len(sequence_data)}")
        print(f"Frames after filtering:  {len(sequence_data_new)}\n")
        
    mmcv.dump(data_infos, output_pkl_path)


if __name__ == "__main__":
    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(f'{current_folder_path}/../..')
    
    parser = argparse.ArgumentParser(description="Filter V2X-Sim sequential data for specific cooperative scenarios.")
    parser.add_argument('dataset_name', type=str, help="Target dataset name (e.g., V2X-Sim-full-id1)")
    parser.add_argument('coop_type', type=str, choices=['v2v', 'v2i'], help="Cooperative scenario type (v2v or v2i)")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    coop_type = args.coop_type
    
    info_dir = f"data/infos/{dataset_name}"
    
    print("========================================================")
    print(f"Filtering valid scenes for: {dataset_name} ({coop_type.upper()})")
    print("========================================================")

    # Process both train and val splits
    for split in ["train", "val"]:
        input_pkl = f"{info_dir}/nuscenes_infos_temporal_{split}.pkl"
        output_pkl = f"{info_dir}/nuscenes_infos_temporal_{split}_{coop_type}.pkl"
        input_json = f"tools/v2xsim_data_converter/scene2info_{coop_type}_{split}.json"
        
        if os.path.exists(input_pkl) and os.path.exists(input_json):
            process_sequence_data(input_pkl, output_pkl, input_json)
        else:
            print(f"[Warning] Missing required files for {split} split. Skipping...")
            
    print("Scene filtering complete!\n")