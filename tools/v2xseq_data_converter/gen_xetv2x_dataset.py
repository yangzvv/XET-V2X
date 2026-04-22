import os
import json
import argparse
from tqdm import tqdm


def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json


def write_json(new_dict, path_json):
    with open(path_json, "w") as dump_f:
        json.dump(new_dict, dump_f)


def create_directory_structure(output_dataset_path):
    os.system(f'rm -rf {output_dataset_path}')
    directory_structure = [
        'vehicle-side/label/lidar',
        'vehicle-side/label/camera',
        'infrastructure-side/label/virtuallidar',
        'infrastructure-side/label/camera',
        'cooperative/label'
    ]

    for directory in directory_structure:
        os.makedirs(os.path.join(output_dataset_path, directory), exist_ok=True)


def update_label_file(input_label_file_path, output_label_file_path, token, operation=None):
    label_info = read_json(input_label_file_path)
    if operation:
        for i in label_info:
            if i["token"] == token:
                i["type"] = operation[0]
                i["track_id"] == operation[1]
                break
    else:
        for i in label_info:
            if i["token"] == token:
                label_info.remove(i)
                break
    write_json(label_info, output_label_file_path)


def update_label_from_json(input_dataset_path, output_dataset_path, updata_label_info_path="./tools/v2xseq_data_converter/update_label_info.json"):
    update_label_info = read_json(updata_label_info_path)

    list_veh_update_info = update_label_info["vehicle-side"]
    for i in list_veh_update_info:
        update_label_file(
            f'{input_dataset_path}/vehicle-side/label/lidar/{i["frame"]}.json',
            f'{output_dataset_path}/vehicle-side/label/lidar/{i["frame"]}.json',
            i["token"],
            i["operation"]
        )

    list_inf_update_info = update_label_info["infrastructure-side"]
    for i in list_inf_update_info:
        update_label_file(
            f'{input_dataset_path}/infrastructure-side/label/virtuallidar/{i["frame"]}.json',
            f'{output_dataset_path}/infrastructure-side/label/virtuallidar/{i["frame"]}.json',
            i["token"],
            i["operation"]
        )


def gen_sequence_data_info(input_dataset_path, output_dataset_path, list_sequences, freq, delay):
    if not os.path.exists(output_dataset_path):
        create_directory_structure(output_dataset_path)

    input_json_name = 'data_info.json' if delay == 0 else f'data_info_{delay}.json'

    input_veh_data_info_path = os.path.join(input_dataset_path, f'vehicle-side/{input_json_name}')
    input_inf_data_info_path = os.path.join(input_dataset_path, f'infrastructure-side/{input_json_name}')
    input_coop_data_info_path = os.path.join(input_dataset_path, f'cooperative/{input_json_name}')
    
    output_veh_data_info_path = os.path.join(output_dataset_path, 'vehicle-side/data_info.json')
    output_inf_data_info_path = os.path.join(output_dataset_path, 'infrastructure-side/data_info.json')
    output_coop_data_info_path = os.path.join(output_dataset_path, 'cooperative/data_info.json')

    input_veh_data_info = read_json(input_veh_data_info_path)
    input_inf_data_info = read_json(input_inf_data_info_path)
    input_coop_data_info = read_json(input_coop_data_info_path)
    list_output_veh_data_info = []
    list_output_inf_data_info = []
    list_output_coop_data_info = []

    dict_veh_frame2info = {veh_info["frame_id"]: veh_info for veh_info in input_veh_data_info}
    dict_inf_frame2info = {inf_info["frame_id"]: inf_info for inf_info in input_inf_data_info}
    dict_coop_frame2info = {coop_info["vehicle_frame"]: coop_info for coop_info in input_coop_data_info}

    dict_coop_sequence2frames = {}
    for i in input_coop_data_info:
        if i["vehicle_sequence"] not in dict_coop_sequence2frames.keys():
            dict_coop_sequence2frames[i["vehicle_sequence"]] = []
        dict_coop_sequence2frames[i["vehicle_sequence"]].append(i["vehicle_frame"])

    interval = 10 // freq
    for sequence in list_sequences:
        # print("sequence:", sequence)
        if sequence not in dict_coop_sequence2frames:
            continue
        seq_frames = list(sorted(dict_coop_sequence2frames[sequence]))
        list_interval = []
        for i in range(int(seq_frames[0]), int(seq_frames[-1]) + 1, interval):
            list_interval.append(i)
        list_interval.append(list_interval[-1] + interval)

        c = 0
        seq_frames_new = []
        for veh_frame in seq_frames:
            for j in range(c, len(list_interval) - 2):
                if int(veh_frame) >= list_interval[j + 1]:
                    # print(veh_frame)
                    c += 1
            if list_interval[c] <= int(veh_frame) < list_interval[c + 1]:
                seq_frames_new.append(veh_frame)
                list_output_coop_data_info.append(dict_coop_frame2info[veh_frame])
                c += 1

    print("total example frames:", len(list_output_coop_data_info))
    for k in tqdm(list_output_coop_data_info):
        list_output_veh_data_info.append(dict_veh_frame2info[k["vehicle_frame"]])
        list_output_inf_data_info.append(dict_inf_frame2info[k["infrastructure_frame"]])

    write_json(list_output_veh_data_info, output_veh_data_info_path)
    write_json(list_output_inf_data_info, output_inf_data_info_path)
    write_json(list_output_coop_data_info, output_coop_data_info_path)


def filt_label(input_label_file, output_label_file):
    label_info = read_json(input_label_file)
    new_label_info = []
    list_track_id = []
    for i in label_info:
        if i["track_id"] not in list_track_id:
            new_label_info.append(i)
            list_track_id.append(i["track_id"])
        else:
            pass # print("repeat track id:", i["track_id"])
    write_json(new_label_info, output_label_file)


def filt_label_coop(input_label_file, output_label_file):
    label_info = read_json(input_label_file)
    new_label_info = []
    list_track_id = []
    a = 0
    for i in label_info:
        if i["from_side"]!="inf":
            a += 1
            continue
        if i["track_id"] not in list_track_id:
            new_label_info.append(i)
            list_track_id.append(i["track_id"])
    write_json(new_label_info, output_label_file)


def copy_dataset(input_dataset_path, ln_input_dataset_path, output_dataset_path, update_label):
    full_src_path = os.path.normpath(os.path.join(os.getcwd(), ln_input_dataset_path))
    veh_data_info = read_json(os.path.join(output_dataset_path, 'vehicle-side/data_info.json'))
    inf_data_info = read_json(os.path.join(output_dataset_path, 'infrastructure-side/data_info.json'))
    coop_data_info = read_json(os.path.join(output_dataset_path, 'cooperative/data_info.json'))

    if update_label:
        update_label_from_json(input_dataset_path, output_dataset_path)

    for i in tqdm(veh_data_info, desc="Copying vehicle-side"):
        os.system(f"cp -f {input_dataset_path}/vehicle-side/{i['label_camera_std_path']} {output_dataset_path}/vehicle-side/label/camera/")
        os.system(f"cp -f {input_dataset_path}/vehicle-side/{i['label_lidar_std_path']} {output_dataset_path}/vehicle-side/label/lidar/")
        # filt_label(f"datasets/SPD-label-clean/cooperative-vehicle-infrastructure/vehicle-side/{i['label_lidar_std_path']}", f"{output_dataset_path}/vehicle-side/{i['label_lidar_std_path']}")
    os.system(f"ln -s {full_src_path}/vehicle-side/calib {output_dataset_path}/vehicle-side/calib")
    os.system(f"ln -s {full_src_path}/vehicle-side/image {output_dataset_path}/vehicle-side/image")
    os.system(f"ln -s {full_src_path}/vehicle-side/velodyne_180deg {output_dataset_path}/vehicle-side/velodyne")

    for j in tqdm(inf_data_info, desc="Copying infrastructure-side"):
        os.system(f"cp {input_dataset_path}/infrastructure-side/{j['label_camera_std_path']} {output_dataset_path}/infrastructure-side/label/camera/")
        os.system(f"cp {input_dataset_path}/infrastructure-side/{j['label_lidar_std_path']} {output_dataset_path}/infrastructure-side/label/virtuallidar/")
        # filt_label(f"datasets/SPD-label-clean/cooperative-vehicle-infrastructure/infrastructure-side/{j['label_lidar_std_path']}", f"{output_dataset_path}/infrastructure-side/{j['label_lidar_std_path']}")
    os.system(f"ln -s {full_src_path}/infrastructure-side/calib {output_dataset_path}/infrastructure-side/calib")
    os.system(f"ln -s {full_src_path}/infrastructure-side/image {output_dataset_path}/infrastructure-side/image")
    os.system(f"ln -s {full_src_path}/infrastructure-side/velodyne {output_dataset_path}/infrastructure-side/velodyne")

    for k in tqdm(coop_data_info, desc="Copying cooperative"):
        os.system(f"cp -f {input_dataset_path}/cooperative/label/{int(k['vehicle_frame'][1:]):06d}.json {output_dataset_path}/cooperative/label/")
        # os.system(f"cp -f datasets/SPD-label-clean/cooperative-vehicle-infrastructure/cooperative/new_label/{int(k['vehicle_frame'][1:]):06d}.json {output_dataset_path}/cooperative/label/")
    os.system(f"ln -s {full_src_path}/vehicle-side/calib {output_dataset_path}/cooperative/calib")
    os.system(f"ln -s {full_src_path}/vehicle-side/image {output_dataset_path}/cooperative/image")
    os.system(f"ln -s {full_src_path}/vehicle-side/velodyne_180deg {output_dataset_path}/cooperative/velodyne")
    os.system(f"ln -s {full_src_path}/maps {output_dataset_path}/maps")


if __name__ == "__main__":
    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(f'{current_folder_path}/../..')
    curDirectory = os.getcwd()
    print("Working Directory:", curDirectory)

    parser = argparse.ArgumentParser(description='Generate and copy dataset files.')
    parser.add_argument('--input', type=str, help='Input dataset path')
    parser.add_argument('--ln-input', type=str, default='', help='Soft link input dataset path')
    parser.add_argument('--output', type=str, help='Output dataset path')
    parser.add_argument('--sequences', nargs='+', help='List of sequences to process')
    parser.add_argument('--update-label', action='store_true', default=False, help='Whether to update labels.')
    parser.add_argument('--freq', type=int, default=2, help='sample frequency.')
    parser.add_argument('--delay', type=int, default=0, help='Delay frame index (e.g. 0, 1, 2, 3)')
    
    args = parser.parse_args()

    input_dataset_path = args.input
    output_dataset_path = args.output
    list_sequences = args.sequences
    update_label = args.update_label
    freq = args.freq
    delay = args.delay

    if not args.ln_input:
        ln_input_dataset_path = input_dataset_path
    else:
        ln_input_dataset_path = args.ln_input

    create_directory_structure(output_dataset_path)
    gen_sequence_data_info(input_dataset_path, output_dataset_path, list_sequences, freq, delay)
    copy_dataset(input_dataset_path, ln_input_dataset_path, output_dataset_path, update_label)
    