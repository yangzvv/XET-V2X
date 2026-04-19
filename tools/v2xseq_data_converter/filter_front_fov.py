import os
import argparse
import numpy as np
from pypcd import pypcd
from tqdm import tqdm

def filter_front(points_data: np.ndarray, fov_deg: float = 45.0) -> np.ndarray:
    """
    Filters a point cloud to retain points within a specified frontal field of view (FOV).
    
    Args:
        points_data: Structured numpy array containing 'x' and 'y' fields.
        fov_deg: Half-angle of the FOV in degrees (e.g., 45.0 means +/- 45 degrees).
        
    Returns:
        Boolean mask indicating which points are within the FOV.
    """
    x = points_data['x']
    y = points_data['y']
    angles = np.arctan2(y, x)  # Range: [-pi, pi]
    fov_rad = np.deg2rad(fov_deg)
    
    mask = np.logical_and(angles >= -fov_rad, angles <= fov_rad)
    return mask

def process_folder(input_dir: str, output_dir: str, fov_deg: float = 45.0):
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a valid directory: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    files = [os.path.join(input_dir, fn) for fn in os.listdir(input_dir) if fn.endswith('.pcd')]
    print(f"Found {len(files)} .pcd files. Starting processing...")

    for src_path in tqdm(files):
        fname = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, fname)

        # 1. Read the point cloud
        pc = pypcd.PointCloud.from_path(src_path)

        # 2. Extract structured data
        data = pc.pc_data

        # 3. Filter data
        mask = filter_front(data, fov_deg)
        filtered_data = data[mask]

        # 4. Update the PointCloud object
        pc.pc_data = filtered_data
        pc.points = filtered_data.shape[0]
        pc.width = filtered_data.shape[0]

        # 5. Save while preserving original metadata/compression
        pc.save_pcd(dst_path, compression=pc.data)

    print("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch process point clouds to retain only points within a frontal FOV."
    )
    parser.add_argument('input_dir', type=str, help="Directory containing input .pcd files.")
    parser.add_argument('output_dir', type=str, help="Directory to save filtered .pcd files.")
    parser.add_argument('--fov', type=float, default=45.0, help="Half-angle of the FOV in degrees (default: 45.0).")
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.fov)

    # Example usage:
    # python tools/v2xseq_data_converter/filter_front_fov.py ./datasets/V2X-Seq-SPD/vehicle-side/velodyne ./datasets/V2X-Seq-SPD/vehicle-side/velodyne_180deg --fov 90.0