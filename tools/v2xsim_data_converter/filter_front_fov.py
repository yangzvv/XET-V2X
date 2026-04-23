import os
import argparse
import numpy as np
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from pypcd.pypcd import PointCloud

def filter_front(points: np.ndarray, fov_deg: float = 45.0) -> np.ndarray:
    """
    Filter the original point cloud to retain points within a specified frontal field of view (FOV).
    
    Args:
        points: (4, N) numpy array representing [x, y, z, intensity].
        fov_deg: Half-angle of the FOV in degrees.
        
    Returns:
        Filtered point cloud array in the same (4, N) format.
    """
    x = points[0, :]
    y = points[1, :]
    angles = np.arctan2(y, x)  # Range: [-pi, pi], positive is left, negative is right
    fov_rad = np.deg2rad(fov_deg)
    
    mask = np.logical_and(angles >= -fov_rad, angles <= fov_rad)
    return points[:, mask]

def save_as_pcd(points: np.ndarray, output_path: str):
    """
    Save the point cloud as a standard .pcd file with proper headers using pypcd.
    This strictly matches the format expected by mmcv/mmdet3d dataloaders.
    
    Args:
        points: (4, N) numpy array.
        output_path: Destination path for the .pcd file.
    """
    points_transposed = points.T  # Shape becomes [N, 4]
    num_points = points_transposed.shape[0]
    
    # 1. Build Standard PCD Header
    header = {
        'version': .7,
        'fields': ['x', 'y', 'z', 'intensity'],
        'size': [4, 4, 4, 4],          # 4 bytes for float32
        'type': ['F', 'F', 'F', 'F'],  # F = Float
        'count': [1, 1, 1, 1],
        'width': num_points,
        'height': 1,                   # 1 means unorganized point cloud
        'viewpoint': [0, 0, 0, 1, 0, 0, 0],
        'points': num_points,
        'data': 'binary'               # Explicitly define data type
    }

    # 2. Convert to numpy record array format required by pypcd
    pc_data = np.core.records.fromarrays(
        points,  # Use original [4, N] format for fromarrays
        names=header['fields'],
        formats=['f4', 'f4', 'f4', 'f4']
    )
    
    # 3. Instantiate and save
    pcd = PointCloud(header, pc_data)
    # Using 'binary' compression drastically speeds up downstream PyTorch dataloading
    pcd.save_pcd(output_path, compression='binary')


def process_folder(input_dir: str, output_dir: str, fov_deg: float = 45.0, apply_filter: bool = True):
    """
    Iterate through all .pcd.bin or .pcd files in input_dir, optionally apply the frontal 
    point cloud filter, and save the results to output_dir.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a valid directory: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Support common point cloud extensions
    exts = ['.pcd.bin', '.pcd']
    files = []
    for ext in exts:
        files.extend([os.path.join(input_dir, fn) for fn in os.listdir(input_dir) if fn.endswith(ext)])

    print(f"Found {len(files)} point cloud files. Starting processing...")

    for src_path in tqdm(files, desc="Processing Point Clouds"):
        fname = os.path.basename(src_path).replace('.pcd.bin', '.pcd')
        dst_path = os.path.join(output_dir, fname)

        # 1. Load point cloud
        pc = LidarPointCloud.from_file(src_path)
        
        # 2. Filter points if required
        if apply_filter:
            pts_filt = filter_front(pc.points, fov_deg)
        else:
            pts_filt = pc.points
            
        # 3. Save as standard .pcd with header (Fixed the bug here)
        save_as_pcd(pts_filt, dst_path)

    print("Processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch process point clouds to retain only the frontal region within a specified FOV."
        )
    parser.add_argument(
        'input_dir', type=str,
        help="Input directory containing .pcd.bin or .pcd files."
        )
    parser.add_argument(
        'output_dir', type=str,
        help="Output directory (will be created automatically if it doesn't exist)."
        )
    parser.add_argument(
        '--fov', type=float, default=45.0,
        help="Half-angle of the FOV in degrees (default: 45.0). Total FOV = 2 * fov."
        )
    parser.add_argument(
        '--filter_front',
        action='store_true',
        help='Whether to apply the frontal FOV filter.'
        )
    
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.fov, args.filter_front)
    
    # Note: V2X-Sim only processes id_1 and id_2; id_0 does not require LiDAR filtering.
    # Example usage:
    # python tools/v2xsim_data_converter/filter_front_fov.py ./datasets/V2X-Sim-2.0/sweeps/LIDAR_TOP_id_1 ./datasets/V2X-Sim-2.0/sweeps_90deg/LIDAR_TOP_id_1 --fov 45.0 --filter_front
    # python tools/v2xsim_data_converter/filter_front_fov.py ./datasets/V2X-Sim-2.0/sweeps/LIDAR_TOP_id_2 ./datasets/V2X-Sim-2.0/sweeps_90deg/LIDAR_TOP_id_2 --fov 45.0 --filter_front
    # python tools/v2xsim_data_converter/filter_front_fov.py ./datasets/V2X-Sim-2.0/sweeps/LIDAR_TOP_id_0 ./datasets/V2X-Sim-2.0/sweeps_90deg/LIDAR_TOP_id_0
