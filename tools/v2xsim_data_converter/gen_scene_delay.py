import os
import json
import argparse
from tqdm import tqdm

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def write_json(new_dict, path_json):
    with open(path_json, "w") as dump_f:
        json.dump(new_dict, dump_f, indent=4)

def gen_scene_delay(scene_json_path, sample_json_path, sample_data_json_path):
    print(f"Applying temporal shift to {os.path.dirname(scene_json_path)}...")
    scene_info = load_json(scene_json_path)
    sample_info = load_json(sample_json_path)
    sample_data_info = load_json(sample_data_json_path)
    
    dict_sample_token2info = {}
    for i in sample_info:
        if i["token"] in dict_sample_token2info:
            print("Warning: Duplicate sample token found")
        dict_sample_token2info[i["token"]] = i
        
    scene_info_new = []
    sample_token_del = []
    sample_token_first = []
    
    for j in tqdm(scene_info, desc="Processing scenes"):
        first_sample_token_old = j["first_sample_token"] 
        sample_token_del.append(first_sample_token_old)
        
        # Shift 1 frame
        first_sample_token_new = dict_sample_token2info[first_sample_token_old]["next"]
        sample_token_del.append(first_sample_token_new)
        
        # Shift 2 frames
        first_sample_token_new = dict_sample_token2info[first_sample_token_new]["next"]
        sample_token_first.append(first_sample_token_new)
        
        j["first_sample_token"] = first_sample_token_new
        scene_info_new.append(j)
        
    sample_info_new = []
    for i in tqdm(sample_info, desc="Processing samples"):
        if i["token"] in sample_token_del:
            continue
        if i["token"] in sample_token_first:
            i["prev"] = ""
        sample_info_new.append(i)
        
    sample_data_info_new = []
    for i in tqdm(sample_data_info, desc="Processing sample_data"):
        if i["sample_token"] in sample_token_del:
            continue
        if i["sample_token"] in sample_token_first:
            i["prev"] = ""
        sample_data_info_new.append(i)

    # Overwrite the original files safely
    write_json(scene_info_new, scene_json_path)
    write_json(sample_info_new, sample_json_path)
    write_json(sample_data_info_new, sample_data_json_path)
    print("Temporal shift completed successfully!\n")

if __name__ == "__main__":
    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(f'{current_folder_path}/../..')
    
    parser = argparse.ArgumentParser(description="Shift temporal sequences for delay configurations.")
    parser.add_argument('target_dir', type=str, help="Target directory (e.g., datasets/V2X-Sim-full-v2v-delay)")
    args = parser.parse_args()

    v1_0_dir = os.path.join(args.target_dir, "v1.0-trainval")
    
    gen_scene_delay(
        os.path.join(v1_0_dir, "scene.json"),
        os.path.join(v1_0_dir, "sample.json"),
        os.path.join(v1_0_dir, "sample_data.json")
    )