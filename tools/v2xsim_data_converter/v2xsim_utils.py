import json
from tqdm import tqdm
import os
import numpy as np
import argparse

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=2)

def process_instance(v2xsim_instances, v2xsim_sample_annotations):
    instance_to_anns = {inst['token']: [] for inst in v2xsim_instances}
    for annotation in v2xsim_sample_annotations:
        if annotation['instance_token'] in instance_to_anns:
            instance_to_anns[annotation['instance_token']].append(annotation)
    
    v2xsim_instances_new = []
    v2xsim_sample_annotations_new = []
    
    for instance in tqdm(v2xsim_instances):
        anns = instance_to_anns.get(instance['token'], [])

        if not anns:
            continue

        instance['first_annotation_token'] = anns[0]['token']
        instance['last_annotation_token'] = anns[-1]['token']
        instance['nbr_annotations'] = len(anns)

        anns[0]['prev'] = ""
        anns[-1]['next'] = ""
        
        v2xsim_sample_annotations_new.extend(anns)
            
        v2xsim_instances_new.append(instance)

    return v2xsim_instances_new, v2xsim_sample_annotations_new

# def process_instance(v2xsim_instances, v2xsim_sample_annotations):
#     annotations_mappings = {}
#     for annotation in v2xsim_sample_annotations:
#         annotations_mappings[annotation['token']] = annotation
    
#     for instance in tqdm(v2xsim_instances):
#         instance_token = instance['token']
#         first_annotation_token = instance['first_annotation_token']
#         nbr_annotations = instance['nbr_annotations']

#         current_annotation_token = first_annotation_token
#         for ii in range(nbr_annotations - 2):
#             current_annotation_token = annotations_mappings[current_annotation_token]['next']
        
#         current_annotation_token_next = annotations_mappings[current_annotation_token]['next']
#         if current_annotation_token_next not in annotations_mappings.keys():
#             print("There is wrong instance token: ", instance_token)
#             instance['nbr_annotations'] = nbr_annotations - 1
#             instance['last_annotation_token'] = current_annotation_token

#             annotations_mappings[current_annotation_token]['next'] = ''

#     v2xsim_instances_new = v2xsim_instances
#     v2xsim_sample_annotations_new = []
#     for key in annotations_mappings.keys():
#         v2xsim_sample_annotations_new.append(annotations_mappings[key])

#     return v2xsim_instances_new, v2xsim_sample_annotations_new


def process_sample_annotations(v2xsim_sample_annotations):
    for annotation in tqdm(v2xsim_sample_annotations):
        annotation['num_lidar_pts'] = 1
        annotation['num_radar_pts'] = 1

        if not all (np.array(annotation['size'])) > 0.0:
            if not annotation['size'][0] > 0.0:
                annotation['size'][0] = 1.0
            else:
                print("There is a except case!", annotation)
                os._exit()

    return v2xsim_sample_annotations


def process_category(v2xsim_categories):
    category_mappings = {
        "noise": "noise",
        "animal": "noise",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.personal_mobility": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "human.pedestrian.stroller": "pedestrian",
        "human.pedestrian.wheelchair": "pedestrian",
        "movable_object.barrier": "noise",
        "movable_object.debris": "noise",
        "movable_object.pushable_pullable": "noise",
        "movable_object.trafficcone": "traffic_cone",
        # "static_object.bicycle_rack": "bicycle",
        "static_object.bicycle_rack": "noise",
        # "vehicle.bicycle": "bicycle",
        "vehicle.bicycle": "noise",
        "vehicle.bus.bendy": "car",
        "vehicle.bus.rigid": "car",
        "vehicle.car": "car",
        "vehicle.construction": "car",
        "vehicle.emergency.ambulance": "car",
        "vehicle.emergency.police": "car",
        # "vehicle.motorcycle": "bicycle",
        "vehicle.motorcycle": "noise",
        "vehicle.trailer": "car",
        "vehicle.truck": "car",
        "flat.driveable_surface": "noise",
        "flat.other": "noise",
        "flat.sidewalk": "noise",
        "flat.terrain": "noise",
        "static.manmade": "noise",
        "static.other": "noise",
        "static.vegetation": "noise",
        "vehicle.ego": "noise"
    }

    for category in v2xsim_categories:
        category["name"] = category_mappings[category["name"]]

    return v2xsim_categories


def process_timestamp(v2xsim_sample, v2xsim_sample_data, v2xsim_ego_pose):
    if v2xsim_sample[0]['timestamp'] < 1000:
        sample_timestamp_mappings = {}
        egopose_timestamp_mappings = {}
        for spl_data in tqdm(v2xsim_sample_data):
            scene_num = int(spl_data['filename'].split('/')[-1].split('_')[1])
            new_timestamp = (spl_data['timestamp'] + scene_num * 1000) / 5.0 * 1e6
            spl_data['timestamp'] = new_timestamp

            sample_timestamp_mappings[spl_data['sample_token']] = new_timestamp
            egopose_timestamp_mappings[spl_data['ego_pose_token']] = new_timestamp
        for spl in tqdm(v2xsim_sample):
            spl['timestamp'] = sample_timestamp_mappings[spl['token']]
        for pose in tqdm(v2xsim_ego_pose):
            pose['timestamp'] = egopose_timestamp_mappings[pose['token']]

    return v2xsim_sample, v2xsim_sample_data, v2xsim_ego_pose


def generate_split_data():
    split_data_path = "data/split_datas/v2x-sim-split.json"
    
    split_data = {}
    split_data['batch_split'] = {}
    split_data['batch_split']['train'] = []
    split_data['batch_split']['val'] = []
    split_data['batch_split']['test'] = []

    for ii in range(0, 100, 10):
        split_data['batch_split']['train'].append('scene_' + str(ii + 1))
        split_data['batch_split']['train'].append('scene_' + str(ii + 2))
        split_data['batch_split']['train'].append('scene_' + str(ii + 3))
        split_data['batch_split']['train'].append('scene_' + str(ii + 4))
        split_data['batch_split']['train'].append('scene_' + str(ii + 10))

        split_data['batch_split']['val'].append('scene_' + str(ii + 5))
        split_data['batch_split']['val'].append('scene_' + str(ii + 6))

        split_data['batch_split']['test'].append('scene_' + str(ii + 7))
        split_data['batch_split']['test'].append('scene_' + str(ii + 8))
        split_data['batch_split']['test'].append('scene_' + str(ii + 9))
    
    write_json(split_data, split_data_path)


def process_log(v2xsim_log):
    for log in v2xsim_log:
        log['location'] = "yizhuang06"

    return v2xsim_log

def process_filter_scene(v2xsim_scene, v2xsim_sample, v2xsim_sample_data,
                        v2xsim_instances, v2xsim_sample_annotations, v2xsim_ego_pose):
    
    # select part scenes
    selective_scenes = ['ce0d35zgspb9w90ytv0x9ik8bb6a1z7h', '695k6w086m10h029b97n2zr7016ospc3', 'paz0y66keum5hs518ll29juicrhyu1s7', 
                        '8fvf8y39zpu77129u39j897si62jd98e', 'ofw501ws5eyia0341m1f83ofwf51w9q8', '60j43l02d6o8116439h54s69p4f6xr36', 
                        'kpxu2j6ys749dj339f2bs10fyrr8nd3d', 'dbi6p7n8v723c7t1s6z7830u4399514l', 'wgh352t5i6vv1z7b7c0770v5yb8iht4m', 
                        'xj8d1xde704mat5806hzp4fkkc9j031p', '956g5g9f59ul9f20s643am7j2r200gog', '117y6rn0i23449rgej23ojcgforc6t15', 
                        '63wb52l1h3zk47376xt88p8q8m12r445', '9a3c97190juavvw27i7799i724zrb71d', 't77afy2s6zcwvu19vxn3529x78j2fr4f', 
                        'w775a716af12zn8emh435663k775odr0', 'df4e23v89n2je9f518j18z9eouc7z8s1', '13h2lg915198yajbcrp24p94206dnpme', 
                        '5k7r5o14x9w6781ap73em40f109w3gur', '7n1723102odv30zc975z6cfuwda04wy5', '23l36l6d78y811p4b2q2w3wu8gey4jaf', 
                        'l4n2p7ns26x6p25247ycq809dlh718y0', 't7qk2sei8w64j41f8rozde15dqcgbmc3', 'a2s507q88a9o1c9678ye63jg69n35s3c', 
                        '3a2ya6lsr12x48t72k0v4c3e59247e49', 'jgdmz50t29is9d81ymwu323f3ui1dkws', 'l08s5i5b8u5z5ai4ul3tdkwsq2nh47lx', 
                        'fz522mr7d311t14g1cy877c3px8q1m50', 'uk7vc473z5ntv864p5ggb51jfnp0zqq0', '9uqxa40p9ovhyg2747u7dydi2bqj2aq4', 
                        'g5rbk3kn54g549d40246s0j8v3uyd5lc', '7rwb6x0ee1053vcm402e2d6ca7116a1a', 'c0i0kq6ka5mnocvkc3x24u7o02c93b09', 
                        '4073497k0219jf3m988gd64y2puj84o2', 'u26od88ox40vqmvmg74iezu7ksjj0710', 'mn21v085w3j945gmh325h8zofu20wltw', 
                        'tdy7sk803p0gv2yvmv1504y28gg6r723', '09j8qd2oxcrd1ptv6n0136bb5835v126', 'u3742eh2t999otum4ltppcp31e91t336', 
                        'gnynqs1qt50x4zx93c6w5m7kh4e7ob86', '762huvy2w431yi0vv0feo050zivdzz8o', 'k08lh35b9m08z9vafslrvk056fo452xp', 
                        'pk11xd1sk9kshz8e60f6erh02453iu31', '7u8ufbol1w3i9p7fp683qzl6gc17yn05', 'd85xh10849nncc3uqjj5ay8v9n404541', 
                        '31s6agg90pj3g89q613m4vsc8w8flarp', '906r9j2f4tu0a476if9007bve201n52q', '35l3tjf77yww80g32k1y908bvkpso38p', 
                        '1704k89771hza60137681v6j15bcrxgy', 'njq63lfzn3bqju5fpn54c06i13lf7457', '57q2s5p0o2m2w578egn5sloy1q5p7ze1', 
                        '8m59q56x8wy5gse96fms136i5q7ffal7', '7x63922373gi9r5421bgfhfr44f49e67', 'puy1nx7o23f4172rqud84e7lz3r04y0v', 
                        'c30f60s0xf2fik77e2p87545940719h2', 'v5114i4srn2v1nj63fr71yydl8r573x2', 'g9xo00roih298xm136yy8im70sfz51o5', 
                        'w031r4773ybhss0lc57ia92290ca879w', 'i9072arqzunple195j6cz418b725fc49', 'q13ef47vm080a7hgv7384u6ybhuq2a0a', 
                        'a0de0mchuul976kvw3e69o23id7g49u0', '4hwq75oul322vmms30fzp30jrd45w815', '4z0cgu8212dxr3641gq99qcm0991ys6g', 
                        '265snydmw714u49qua391n3840c31p95', '4lm52c2c47awen0o0u1e9r7i7c4ffm4c', '72253sbs523k16h1ur4hv92b2c9470b8', 
                        'fvvxlr82ccj71j955qb1ym9146br86b0', 'gjk69078nsexj567k99e546sux035145', 'h3y9z0ape66d76ei3zxgu81382my0fvc', 
                        'v7fe955db4a1g87695q7201lmiby7746']

    v2xsim_sample_new = []
    v2xsim_sample_new_tokens = []
    for sample in tqdm(v2xsim_sample):
        if sample['scene_token'] in selective_scenes:
            v2xsim_sample_new.append(sample)
            v2xsim_sample_new_tokens.append(sample['token'])
    
    v2xsim_sample_data_new = []
    v2xsim_ego_pose_new_tokens = []
    for sample_data in tqdm(v2xsim_sample_data):
        if sample_data['sample_token'] in v2xsim_sample_new_tokens:
            v2xsim_sample_data_new.append(sample_data)
            v2xsim_ego_pose_new_tokens.append(sample_data['ego_pose_token'])
    
    v2xsim_scene_new = []
    for scene in tqdm(v2xsim_scene):
        if scene['token'] in selective_scenes:
            v2xsim_scene_new.append(scene)
    
    v2xsim_sample_annotations_new = []
    v2xsim_instances_new_tokens = []
    for sample_annotation in tqdm(v2xsim_sample_annotations):
        if sample_annotation['sample_token'] in v2xsim_sample_new_tokens:
            v2xsim_sample_annotations_new.append(sample_annotation)
            if sample_annotation['instance_token'] not in v2xsim_instances_new_tokens:
                v2xsim_instances_new_tokens.append(sample_annotation['instance_token'])

    v2xsim_instances_new = []
    for instance in tqdm(v2xsim_instances):
        if instance['token'] in v2xsim_instances_new_tokens:
            v2xsim_instances_new.append(instance)

    v2xsim_ego_pose_token_mappings = {}
    for ego_pose in tqdm(v2xsim_ego_pose):
        v2xsim_ego_pose_token_mappings[ego_pose['token']] = ego_pose
    v2xsim_ego_pose_new = []
    for ego_pose_token in tqdm(v2xsim_ego_pose_new_tokens):
        v2xsim_ego_pose_new.append(v2xsim_ego_pose_token_mappings[ego_pose_token])

    return v2xsim_scene_new, v2xsim_sample_new, v2xsim_sample_data_new, v2xsim_instances_new, v2xsim_sample_annotations_new, v2xsim_ego_pose_new

parser = argparse.ArgumentParser(description='V2X-Sim Dataset Preprocess')
parser.add_argument(
    '--root-path',
    type=str,
    default='./datasets/V2X-Sim-2.0-mini',
    help='specify the root path of dataset')
args = parser.parse_args()

if __name__ == '__main__':
    v2xsim_data_root = args.root_path

    v2xsim_scene_path = os.path.join(v2xsim_data_root, "v1.0-trainval/scene.json")
    v2xsim_sample_path = os.path.join(v2xsim_data_root, "v1.0-trainval/sample.json")
    v2xsim_sample_data_path = os.path.join(v2xsim_data_root, "v1.0-trainval/sample_data.json")
    v2xsim_instances_path = os.path.join(v2xsim_data_root, "v1.0-trainval/instance.json")
    v2xsim_sample_annotations_path = os.path.join(v2xsim_data_root, "v1.0-trainval/sample_annotation.json")
    v2xsim_ego_pose_path = os.path.join(v2xsim_data_root, "v1.0-trainval/ego_pose.json")
    v2xsim_log_path = os.path.join(v2xsim_data_root, "v1.0-trainval/log.json")

    v2xsim_scene = load_json(v2xsim_scene_path)
    v2xsim_sample = load_json(v2xsim_sample_path)
    v2xsim_sample_data = load_json(v2xsim_sample_data_path)
    v2xsim_instances = load_json(v2xsim_instances_path)
    v2xsim_sample_annotations = load_json(v2xsim_sample_annotations_path)
    v2xsim_ego_pose = load_json(v2xsim_ego_pose_path)
    v2xsim_log = load_json(v2xsim_log_path)

    print("Start Processing Filtering Scenes:")
    v2xsim_scene, v2xsim_sample, v2xsim_sample_data, v2xsim_instances,v2xsim_sample_annotations, v2xsim_ego_pose = process_filter_scene(v2xsim_scene, v2xsim_sample, v2xsim_sample_data,
                                                                v2xsim_instances, v2xsim_sample_annotations, v2xsim_ego_pose)

    print("Start Processing Instances:")
    v2xsim_instances, v2xsim_sample_annotations = process_instance(v2xsim_instances, v2xsim_sample_annotations)

    print("Start Processing Sample Annotations:")
    v2xsim_sample_annotations = process_sample_annotations(v2xsim_sample_annotations)

    print("Start Processing TimeStamps:")
    v2xsim_sample, v2xsim_sample_data, v2xsim_ego_pose = process_timestamp(v2xsim_sample, v2xsim_sample_data, v2xsim_ego_pose)

    print("Start Processing Log:")
    v2xsim_log = process_log(v2xsim_log)

    write_json(v2xsim_scene, v2xsim_scene_path)
    write_json(v2xsim_sample, v2xsim_sample_path)
    write_json(v2xsim_sample_data, v2xsim_sample_data_path)
    write_json(v2xsim_instances, v2xsim_instances_path)
    write_json(v2xsim_sample_annotations, v2xsim_sample_annotations_path)
    write_json(v2xsim_ego_pose, v2xsim_ego_pose_path)
    write_json(v2xsim_log, v2xsim_log_path)