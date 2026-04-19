import os
import json
import argparse
import copy
from tqdm import tqdm


def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json


def write_json(new_dict, path_json):
    with open(path_json, "w") as dump_f:
        json.dump(new_dict, dump_f)

def gen_frame_sequence_mapping_coop(input_data_info):
    dict_frame2sequence_veh = {}
    dict_frame2sequence_inf = {}
    dict_frame2id_inf = {}
    dict_id2frame_inf = {}
    for i in tqdm(range(len(input_data_info))):
        dict_frame2sequence_veh[input_data_info[i]["vehicle_frame"]] = input_data_info[i]["vehicle_sequence"]
        dict_frame2sequence_inf[input_data_info[i]["infrastructure_frame"]] = input_data_info[i]["infrastructure_sequence"]
        if input_data_info[i]["infrastructure_frame"] in dict_frame2id_inf.keys():
            print(input_data_info[i]["infrastructure_frame"])
        dict_frame2id_inf[input_data_info[i]["infrastructure_frame"]] = i
        dict_id2frame_inf[i] = input_data_info[i]["infrastructure_frame"]
    # print(len(dict_frame2id_inf.keys()))
    # print(len(dict_frame2id_inf.keys()))
    return dict_frame2sequence_veh, dict_frame2sequence_inf, dict_frame2id_inf, dict_id2frame_inf


def gen_delay_k_data_info_coop(input_data_info, dict_frame2sequence_inf, dict_frame2id_inf, dict_id2frame_inf, delay_k):
    output_data_info = []
    data_info = copy.deepcopy(input_data_info)
    for i in tqdm(data_info):
        origin_frame_inf = i["infrastructure_frame"]
        target_id_inf = dict_frame2id_inf[origin_frame_inf] - delay_k
        if target_id_inf not in dict_id2frame_inf.keys():
            continue
        target_frame_inf = dict_id2frame_inf[target_id_inf]
        if target_frame_inf not in dict_frame2sequence_inf.keys() or dict_frame2sequence_inf[target_frame_inf] != dict_frame2sequence_inf[origin_frame_inf]:
            continue
        # 重构frame和sequence
        i["vehicle_frame"] = f'{int(i["vehicle_frame"])+delay_k*100000:06d}'
        i["infrastructure_frame"] = f'{int(target_frame_inf)+delay_k*100000:06d}'
        i["vehicle_sequence"] = f'{int(i["vehicle_sequence"])+delay_k*100:04d}'
        i["infrastructure_sequence"] = f'{int(i["infrastructure_sequence"])+delay_k*100:04d}'
        output_data_info.append(i)
    return output_data_info


def gen_vic_data_info_coop(input_data_info_path, output_path):
    input_data_info = read_json(input_data_info_path)
    output_data_info = []
    output_data_info.extend(input_data_info)

    _, dict_frame2sequence_inf, dict_frame2id_inf, dict_id2frame_inf = gen_frame_sequence_mapping_coop(input_data_info)
    
    data_info_1 = gen_delay_k_data_info_coop(input_data_info, dict_frame2sequence_inf, dict_frame2id_inf, dict_id2frame_inf, 1)
    output_data_info.extend(data_info_1)
    data_info_2 = gen_delay_k_data_info_coop(input_data_info, dict_frame2sequence_inf, dict_frame2id_inf, dict_id2frame_inf, 2)
    output_data_info.extend(data_info_2)
    data_info_3 = gen_delay_k_data_info_coop(input_data_info, dict_frame2sequence_inf, dict_frame2id_inf, dict_id2frame_inf, 3)
    output_data_info.extend(data_info_3)
    write_json(output_data_info, f'{output_path}/data_info_0_1_2_3.json')
    write_json(data_info_1, f'{output_path}/data_info_1.json')
    write_json(data_info_2, f'{output_path}/data_info_2.json')
    write_json(data_info_3, f'{output_path}/data_info_3.json')


def gen_vic_data_info(input_data_info_path, output_path):
    input_data_info = read_json(input_data_info_path)
    output_data_info = []
    output_data_info.extend(input_data_info)

    for delay_k in range(1, 4):
        data_info = copy.deepcopy(input_data_info)
        for i in tqdm(data_info):
            i["frame_id"] = str(delay_k) + i["frame_id"][1:]
            i["sequence_id"] = i["sequence_id"][:1] + str(delay_k) + i["sequence_id"][2:]
        output_data_info.extend(data_info)
        write_json(data_info, f'{output_path}/data_info_{delay_k}.json')
    write_json(output_data_info, f'{output_path}/data_info_0_1_2_3.json')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate all data_info.json files for V2X dataset.')
    parser.add_argument('--base-dir', type=str, default="./datasets/V2X-Seq-SPD", 
                        help='Base directory containing vehicle-side, infrastructure-side, and cooperative folders')

    args = parser.parse_args()
    base_dir = args.base_dir

    v2x_sides = ["vehicle-side", "infrastructure-side", "cooperative"]

    for side in v2x_sides:
        input_data_info_path = os.path.join(base_dir, side, "data_info.json")
        output_path = os.path.join(base_dir, side)

        if not os.path.exists(input_data_info_path):
            print(f"\n[Warning] Input file not found: {input_data_info_path}. Skipping...")
            continue
        
        print(f"\n========== Processing {side} ==========")
        
        if side == "cooperative":
            gen_vic_data_info_coop(input_data_info_path, output_path)
        else:
            gen_vic_data_info(input_data_info_path, output_path)
            
    print("\nAll JSON files for vehicle-side, infrastructure-side, and cooperative have been generated successfully!")