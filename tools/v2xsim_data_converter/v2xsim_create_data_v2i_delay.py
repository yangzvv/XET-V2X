import argparse
from os import path as osp
import sys
from tools.v2xsim_data_converter import v2xsim_nuscenes_converter_v2i_delay as nuscenes_converter
sys.path.append('..')


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10,
                       coop=False,
                       mono = True
                       ):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    # nuscenes_converter.create_nuscenes_infos(
    #     root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps, use_can_bus=False)
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps, use_can_bus=False, mono=mono, coop=coop)

    if version == 'v1.0-test':
        info_test_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version, mono=mono)
    else:
        info_train_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_train_path, version=version, mono=mono)
        nuscenes_converter.export_2d_annotation(
            root_path, info_val_path, version=version, mono=mono)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./datasets/V2X-Sim-2.0-id1',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/infos/V2X-Sim-2.0-id1',
    required=False,
    help='name of info pkl')
parser.add_argument(
    '--coop',
    action='store_true',
    help='whether generate cooperative data'
)
parser.add_argument(
    '--mono',
    action='store_true',
    help='whether generate mono data'
)
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            coop=args.coop,
            mono=args.mono
            )
        # test_version = f'{args.version}-test'
        # nuscenes_data_prep(
        #     root_path=args.root_path,
        #     can_bus_root_path=args.canbus,
        #     info_prefix=args.extra_tag,
        #     version=test_version,
        #     dataset_name='NuScenesDataset',
        #     out_dir=args.out_dir,
        #     max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            mono=args.mono
            )