import os
import json
import math
import pickle
import copy
import mmcv
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pypcd.pypcd import PointCloud
current_folder_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{current_folder_path}/../..')
curDirectory = os.getcwd()
print(curDirectory)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def process_sequence_data(
    input_pkl_path: str,
    output_pkl_path: str,
    dataset_root: str,
    nusc_version: str = "v1.0-trainval",
    verbose: bool = True
) -> None:
    """
    处理时序数据并应用有效标记过滤
    
    参数：
    input_pkl_path: 输入pkl文件路径
    output_pkl_path: 输出pkl文件路径
    dataset_root: NuScenes数据集根目录
    nusc_version: 数据集版本号, 默认v1.0-trainval
    verbose: 是否显示进度条, 默认True
    """
    # 加载原始数据
    data_infos = mmcv.load(input_pkl_path)
    sequence_data = data_infos["infos"]
    
    # 初始化NuScenes数据集对象
    nusc = NuScenes(version=nusc_version, dataroot=dataset_root)
    
    # 创建处理容器
    sequence_data_new = []
    dict_sample_token2list_instance_token = {}
    
    # 配置进度条
    iterator = tqdm(sequence_data, desc="Processing frames") if verbose else sequence_data
    
    # 逐帧处理
    for frame_data in iterator:
        # 更新LiDAR路径
        frame_data["lidar_path"] = frame_data["lidar_path"].replace("sweeps", "sweeps_pcd").replace(".bin", "")
        if "lidar_path_1delay" in frame_data.keys():
            frame_data["lidar_path_1delay"] = frame_data["lidar_path_1delay"].replace("sweeps", "sweeps_pcd").replace(".bin", "")
        if "lidar_path_2delay" in frame_data.keys():
            frame_data["lidar_path_2delay"] = frame_data["lidar_path_2delay"].replace("sweeps", "sweeps_pcd").replace(".bin", "")
        
        # 应用有效标记过滤
        mask = copy.deepcopy(frame_data["valid_flag"])
        filter_fields = [
            "gt_boxes", "gt_names", "gt_velocity",
            "num_lidar_pts", "num_radar_pts",
            "valid_flag", "gt_inds", "gt_ins_tokens",
            "fut_traj", "fut_traj_valid_mask",
            "visibility_tokens"
        ]
        
        # 批量处理字段过滤
        for field in filter_fields:
            if field in frame_data and len(frame_data[field]) == len(mask):
                frame_data[field] = frame_data[field][mask]
            else:
                raise KeyError(f"Invalid field {field} or dimension mismatch")
        
        sequence_data_new.append(frame_data)

        dict_sample_token2list_instance_token[frame_data["token"]] = frame_data["gt_ins_tokens"]
    
    # 更新并保存数据
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
    严格过滤样本标注数据
    
    参数：
    input_anno_path: 输入sample_annotation.json路径
    output_anno_path: 输出文件路径
    sample_token_dict: {sample_token: [valid_instance_tokens]}
    verbose: 是否显示进度条
    
    返回统计字典：
    {
        'total': 总处理条目数,
        'kept': 保留条目数,
        'invalid_sample': 样本不匹配数,
        'invalid_instance': 实例不匹配数,
        'missing_field': 字段缺失数
    }
    """
    # ==================== 数据预处理 ====================
    # 将字典值转换为集合提升查询性能
    validation_map = {
        sample_token: set(instance_tokens)
        for sample_token, instance_tokens in sample_token_dict.items()
    }
    
    # ==================== 加载原始数据 ====================
    with open(input_anno_path, 'r') as f:
        annotations = json.load(f)
    
    # ==================== 执行过滤 ====================
    stats = defaultdict(int)
    filtered = []
    
    # 迭代处理每个标注条目
    iterator = annotations
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(annotations, desc="Filtering annotations")
    
    for anno in iterator:
        stats['total'] += 1
        
        # 字段存在性检查
        if 'sample_token' not in anno or 'instance_token' not in anno:
            stats['missing_field'] += 1
            continue
            
        sample_token = anno['sample_token']
        instance_token = anno['instance_token']
        
        # 样本级过滤
        if sample_token not in validation_map:
            stats['invalid_sample'] += 1
            continue
            
        # 实例级过滤
        if instance_token not in validation_map[sample_token]:
            stats['invalid_instance'] += 1
            continue
            
        # 通过所有检查
        filtered.append(anno)
        stats['kept'] += 1
    
    # ==================== 二次过滤：清理prev和next指针 ====================
    if filtered:
        # 构建有效token集合
        valid_tokens = {anno["token"] for anno in filtered}
        
        # 清理无效的next指针
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
    
    # ==================== 保存结果 ====================
    with open(output_anno_path, 'w') as f:
        json.dump(filtered, f, indent=4)
    
    return dict(stats)


if __name__ == '__main__':
    # V2X-Sim-full-id1
    # dict_sample_token2list_instance_token_train = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-id1/nuscenes_infos_temporal_train.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-id1/nuscenes_infos_temporal_train.pkl",
    #     dataset_root="datasets/V2X-Sim-full-id1"
    # )

    # dict_sample_token2list_instance_token_val = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-id1/nuscenes_infos_temporal_val.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-id1/nuscenes_infos_temporal_val.pkl",
    #     dataset_root="datasets/V2X-Sim-full-id1"
    # )

    # dict_sample_token2list_instance_token_train.update(dict_sample_token2list_instance_token_val)

    # stats = filter_sample_annotations(
    #     input_anno_path="datasets/V2X-Sim-full-id1/v1.0-trainval/sample_annotation.json",
    #     output_anno_path="datasets/V2X-Sim-full-id1/v1.0-trainval/sample_annotation.json",
    #     sample_token_dict=dict_sample_token2list_instance_token_train
    # )

    # V2X-Sim-full-id2
    # dict_sample_token2list_instance_token_train = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-id2/nuscenes_infos_temporal_train.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-id2/nuscenes_infos_temporal_train.pkl",
    #     dataset_root="datasets/V2X-Sim-full-id2"
    # )

    # dict_sample_token2list_instance_token_val = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-id2/nuscenes_infos_temporal_val.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-id2/nuscenes_infos_temporal_val.pkl",
    #     dataset_root="datasets/V2X-Sim-full-id2"
    # )

    # dict_sample_token2list_instance_token_train.update(dict_sample_token2list_instance_token_val)

    # stats = filter_sample_annotations(
    #     input_anno_path="datasets/V2X-Sim-full-id2/v1.0-trainval/sample_annotation.json",
    #     output_anno_path="datasets/V2X-Sim-full-id2/v1.0-trainval/sample_annotation.json",
    #     sample_token_dict=dict_sample_token2list_instance_token_train
    # )

    # V2X-Sim-full-id0
    # dict_sample_token2list_instance_token_train = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-id0/nuscenes_infos_temporal_train.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-id0/nuscenes_infos_temporal_train.pkl",
    #     dataset_root="datasets/V2X-Sim-full-id0"
    # )

    # dict_sample_token2list_instance_token_val = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-id0/nuscenes_infos_temporal_val.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-id0/nuscenes_infos_temporal_val.pkl",
    #     dataset_root="datasets/V2X-Sim-full-id0"
    # )

    # dict_sample_token2list_instance_token_train.update(dict_sample_token2list_instance_token_val)

    # stats = filter_sample_annotations(
    #     input_anno_path="datasets/V2X-Sim-full-id0/v1.0-trainval/sample_annotation.json",
    #     output_anno_path="datasets/V2X-Sim-full-id0/v1.0-trainval/sample_annotation.json",
    #     sample_token_dict=dict_sample_token2list_instance_token_train
    # )

    # V2X-Sim-full-v2v
    # dict_sample_token2list_instance_token_train = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-v2v/nuscenes_infos_temporal_train.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-v2v/nuscenes_infos_temporal_train.pkl",
    #     dataset_root="datasets/V2X-Sim-full-v2v"
    # )

    # dict_sample_token2list_instance_token_val = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-v2v/nuscenes_infos_temporal_val.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-v2v/nuscenes_infos_temporal_val.pkl",
    #     dataset_root="datasets/V2X-Sim-full-v2v"
    # )

    # dict_sample_token2list_instance_token_train.update(dict_sample_token2list_instance_token_val)

    # stats = filter_sample_annotations(
    #     input_anno_path="datasets/V2X-Sim-full-v2v/v1.0-trainval/sample_annotation.json",
    #     output_anno_path="datasets/V2X-Sim-full-v2v/v1.0-trainval/sample_annotation.json",
    #     sample_token_dict=dict_sample_token2list_instance_token_train
    # )

    # V2X-Sim-full-v2i
    # dict_sample_token2list_instance_token_train = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-v2i/nuscenes_infos_temporal_train.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-v2i/nuscenes_infos_temporal_train.pkl",
    #     dataset_root="datasets/V2X-Sim-full-v2i"
    # )

    # dict_sample_token2list_instance_token_val = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-v2i/nuscenes_infos_temporal_val.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-v2i/nuscenes_infos_temporal_val.pkl",
    #     dataset_root="datasets/V2X-Sim-full-v2i"
    # )

    # dict_sample_token2list_instance_token_train.update(dict_sample_token2list_instance_token_val)

    # stats = filter_sample_annotations(
    #     input_anno_path="datasets/V2X-Sim-full-v2i/v1.0-trainval/sample_annotation.json",
    #     output_anno_path="datasets/V2X-Sim-full-v2i/v1.0-trainval/sample_annotation.json",
    #     sample_token_dict=dict_sample_token2list_instance_token_train
    # )

    # V2X-Sim-full-v2v-delay
    # dict_sample_token2list_instance_token_train = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-v2v-delay/nuscenes_infos_temporal_train.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-v2v-delay/nuscenes_infos_temporal_train.pkl",
    #     dataset_root="datasets/V2X-Sim-full-v2v-delay"
    # )

    # dict_sample_token2list_instance_token_val = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-v2v-delay/nuscenes_infos_temporal_val.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-v2v-delay/nuscenes_infos_temporal_val.pkl",
    #     dataset_root="datasets/V2X-Sim-full-v2v-delay"
    # )

    # dict_sample_token2list_instance_token_train.update(dict_sample_token2list_instance_token_val)

    # stats = filter_sample_annotations(
    #     input_anno_path="datasets/V2X-Sim-full-v2v-delay/v1.0-trainval/sample_annotation.json",
    #     output_anno_path="datasets/V2X-Sim-full-v2v-delay/v1.0-trainval/sample_annotation.json",
    #     sample_token_dict=dict_sample_token2list_instance_token_train
    # )

    # V2X-Sim-full-v2i-delay
    # dict_sample_token2list_instance_token_train = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-v2i-delay/nuscenes_infos_temporal_train.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-v2i-delay/nuscenes_infos_temporal_train.pkl",
    #     dataset_root="datasets/V2X-Sim-full-v2i-delay"
    # )

    # dict_sample_token2list_instance_token_val = process_sequence_data(
    #     input_pkl_path="data/infos/V2X-Sim-full-v2i-delay/nuscenes_infos_temporal_val.pkl",
    #     output_pkl_path="data/infos/V2X-Sim-full-v2i-delay/nuscenes_infos_temporal_val.pkl",
    #     dataset_root="datasets/V2X-Sim-full-v2i-delay"
    # )

    # dict_sample_token2list_instance_token_train.update(dict_sample_token2list_instance_token_val)

    # stats = filter_sample_annotations(
    #     input_anno_path="datasets/V2X-Sim-full-v2i-delay/v1.0-trainval/sample_annotation.json",
    #     output_anno_path="datasets/V2X-Sim-full-v2i-delay/v1.0-trainval/sample_annotation.json",
    #     sample_token_dict=dict_sample_token2list_instance_token_train
    # )

    pass
