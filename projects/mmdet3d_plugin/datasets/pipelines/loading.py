import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from einops import rearrange
from mmdet3d.datasets.pipelines import LoadAnnotations3D ,LoadPointsFromMultiSweeps
from mmdet3d.datasets.pipelines import LoadPointsFromFile
from mmdet3d.core.points import BasePoints, get_points_type
import os
from pypcd import pypcd
from nuscenes.utils.data_classes import LidarPointCloud


@PIPELINES.register_module()
class LoadMultiViewImageFromFilesInCeph(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', file_client_args=dict(backend='disk'), img_root=''):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = mmcv.FileClient(**self.file_client_args)
        self.img_root = img_root

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (list of str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        images_multiView = []
        filename = results['img_filename']
        for img_path in filename:
            img_path = os.path.join(self.img_root, img_path)
            if self.file_client_args['backend'] == 'petrel':
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes)
            elif self.file_client_args['backend'] == 'disk':
                img = mmcv.imread(img_path, self.color_type)
            images_multiView.append(img)
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            #[mmcv.imread(name, self.color_type) for name in filename], axis=-1)
            images_multiView, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class XETV2XLoadMultiViewPointsFromFile(LoadPointsFromFile):
    def __init__(self, coord_type, load_dim=6, use_dim=..., shift_height=False, use_color=False, file_client_args=dict(backend='disk'),pts_root=''):

        super().__init__(coord_type, load_dim, use_dim, shift_height, use_color, file_client_args)

        self.pts_root = pts_root
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points
    
    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """       
        if len(results["pc_filename"]) == 1:
            pts_filename_veh = os.path.join(self.pts_root.replace("cooperative", "vehicle-side"), results["pc_filename"][0])
            points_veh = pypcd.PointCloud.from_path(pts_filename_veh)
            points_veh = points_veh.pc_data
            points_veh_np = np.zeros((len(points_veh),self.load_dim))
            points_veh_np[:,0] = points_veh['x']
            points_veh_np[:,1] = points_veh['y']
            points_veh_np[:,2] = points_veh['z']
            points_veh_np[:,3] = points_veh['intensity']
            points_veh = points_veh_np[:, self.use_dim]
            attribute_dims_veh = None
            if self.shift_height:
            # vehicle side
                floor_height_veh = np.percentile(points_veh[:, 2], 0.99)
                height_veh = points_veh[:, 2] - floor_height_veh
                points_veh = np.concatenate(
                    [points_veh[:, :3],
                    np.expand_dims(height_veh, 1), points_veh[:, 3:]], 1)
                attribute_dims_veh = dict(height=3)
            if self.use_color:
                assert len(self.use_dim) >= 6
                # vehicle side
                if attribute_dims_veh is None:
                    attribute_dims_veh = dict()
                attribute_dims_veh.update(
                    dict(color=[
                        points_veh.shape[1] - 3,
                        points_veh.shape[1] - 2,
                        points_veh.shape[1] - 1,
                    ]))
            points_class = get_points_type(self.coord_type)
            points_veh = points_class(points_veh, points_dim=points_veh.shape[-1], attribute_dims=attribute_dims_veh)
            results['points_veh'] = points_veh
            results['pseudoimg_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                ]
            results['pseudoimg_ori_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                ]
            # Set initial values for default meta_keys
            results['pseudoimg_pad_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                ]
            results['pseudoimg_scale_factor'] = 1.0
        else:
            pts_filename_veh = os.path.join(self.pts_root.replace("cooperative", "vehicle-side"), results["pc_filename"][0])
            pts_filename_inf = os.path.join(self.pts_root.replace("cooperative", "infrastructure-side"), results["pc_filename"][1])
            points_veh = pypcd.PointCloud.from_path(pts_filename_veh)
            points_inf = pypcd.PointCloud.from_path(pts_filename_inf)
            points_veh = points_veh.pc_data
            points_inf = points_inf.pc_data

            points_veh_np = np.zeros((len(points_veh),self.load_dim))
            points_veh_np[:,0] = points_veh['x']
            points_veh_np[:,1] = points_veh['y']
            points_veh_np[:,2] = points_veh['z']
            points_veh_np[:,3] = points_veh['intensity']
            points_veh = points_veh_np[:, self.use_dim]
        

            points_inf_np = np.zeros((len(points_inf),self.load_dim))
            points_inf_np[:,0] = points_inf['x']
            points_inf_np[:,1] = points_inf['y']
            points_inf_np[:,2] = points_inf['z']
            points_inf_np[:,3] = points_inf['intensity']
            points_inf = points_inf_np[:, self.use_dim]

            attribute_dims_veh = None
            attribute_dims_inf = None

            if self.shift_height:
                # vehicle side
                floor_height_veh = np.percentile(points_veh[:, 2], 0.99)
                height_veh = points_veh[:, 2] - floor_height_veh
                points_veh = np.concatenate(
                    [points_veh[:, :3],
                    np.expand_dims(height_veh, 1), points_veh[:, 3:]], 1)
                attribute_dims_veh = dict(height=3)
                # infrastructure side
                floor_height_inf = np.percentile(points_inf[:, 2], 0.99)
                height_inf = points_inf[:, 2] - floor_height_inf
                points_inf = np.concatenate(
                    [points_inf[:, :3],
                    np.expand_dims(height_inf, 1), points_inf[:, 3:]], 1)
                attribute_dims_inf = dict(height=3)

            if self.use_color:
                assert len(self.use_dim) >= 6
                # vehicle side
                if attribute_dims_veh is None:
                    attribute_dims_veh = dict()
                attribute_dims_veh.update(
                    dict(color=[
                        points_veh.shape[1] - 3,
                        points_veh.shape[1] - 2,
                        points_veh.shape[1] - 1,
                    ]))
                # infrastructure side
                if attribute_dims_inf is None:
                    attribute_dims_inf = dict()
                attribute_dims_inf.update(
                    dict(color=[
                        points_inf.shape[1] - 3,
                        points_inf.shape[1] - 2,
                        points_inf.shape[1] - 1,
                    ]))

            points_class = get_points_type(self.coord_type)
            points_veh = points_class(points_veh, points_dim=points_veh.shape[-1], attribute_dims=attribute_dims_veh)
            points_inf = points_class(points_inf, points_dim=points_inf.shape[-1], attribute_dims=attribute_dims_inf)

            results['points_veh'] = points_veh
            results['points_inf'] = points_inf
            results['pseudoimg_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                    (
                        results["pc_range_inf"][4]-results["pc_range_inf"][1], 
                        results["pc_range_inf"][3]-results["pc_range_inf"][0],
                        results["pc_range_inf"][5]-results["pc_range_inf"][2]
                    ),
                ]
            results['pseudoimg_ori_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                    (
                        results["pc_range_inf"][4]-results["pc_range_inf"][1], 
                        results["pc_range_inf"][3]-results["pc_range_inf"][0],
                        results["pc_range_inf"][5]-results["pc_range_inf"][2]
                    ),
                ]
            # Set initial values for default meta_keys
            results['pseudoimg_pad_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                    (
                        results["pc_range_inf"][4]-results["pc_range_inf"][1], 
                        results["pc_range_inf"][3]-results["pc_range_inf"][0],
                        results["pc_range_inf"][5]-results["pc_range_inf"][2]
                    ),
                ]
            results['pseudoimg_scale_factor'] = 1.0

        return results


@PIPELINES.register_module()
class XETV2XV2XSimLoadMultiViewPointsFromFile(LoadPointsFromFile):
    def __init__(self, coord_type, load_dim=6, use_dim=..., shift_height=False, use_color=False, file_client_args=dict(backend='disk'),pts_root=''):

        super().__init__(coord_type, load_dim, use_dim, shift_height, use_color, file_client_args)

        self.pts_root = pts_root
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points
    
    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        if len(results["pc_filename"]) == 1:
            pts_filename_veh = os.path.join(self.pts_root, results["pc_filename"][0])

            try:
                points_veh = LidarPointCloud.from_file(pts_filename_veh).points
            except Exception as e:
                raise RuntimeError(f"Failed to load point clouds: {e}")
            
            # points_veh 的形状为 (4, N)，分别表示 [x, y, z, intensity]，需要转置为 (N, 4)
            points_veh = points_veh.T

            # 转换为 numpy 格式，并处理维度
            points_veh_np = np.zeros((points_veh.shape[0], self.load_dim))
            points_veh_np[:, 0] = points_veh[:, 0]  # x
            points_veh_np[:, 1] = points_veh[:, 1]  # y
            points_veh_np[:, 2] = points_veh[:, 2]  # z
            points_veh_np[:, 3] = points_veh[:, 3]  # intensity
            points_veh = points_veh_np[:, self.use_dim]
            attribute_dims_veh = None

            if self.shift_height:
                # vehicle side
                floor_height_veh = np.percentile(points_veh[:, 2], 0.99)
                height_veh = points_veh[:, 2] - floor_height_veh
                points_veh = np.concatenate(
                    [points_veh[:, :3],
                    np.expand_dims(height_veh, 1), points_veh[:, 3:]], 1)
                attribute_dims_veh = dict(height=3)

            if self.use_color:
                assert len(self.use_dim) >= 6
                # vehicle side
                if attribute_dims_veh is None:
                    attribute_dims_veh = dict()
                attribute_dims_veh.update(
                    dict(color=[
                        points_veh.shape[1] - 3,
                        points_veh.shape[1] - 2,
                        points_veh.shape[1] - 1,
                    ]))

            points_class = get_points_type(self.coord_type)
            points_veh = points_class(points_veh, points_dim=points_veh.shape[-1], attribute_dims=attribute_dims_veh)

            results['points_veh'] = points_veh
            results['pseudoimg_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                ]
            results['pseudoimg_ori_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                ]
            # Set initial values for default meta_keys
            results['pseudoimg_pad_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                ]
            results['pseudoimg_scale_factor'] = 1.0
            
        else:
            pts_filename_veh = os.path.join(self.pts_root, results["pc_filename"][0])
            pts_filename_inf = os.path.join(self.pts_root, results["pc_filename"][1])     
        
        
            # 使用 nuScenes devkit 加载点云数据
            try:
                points_veh = LidarPointCloud.from_file(pts_filename_veh).points
                points_inf = LidarPointCloud.from_file(pts_filename_inf).points
            except Exception as e:
                raise RuntimeError(f"Failed to load point clouds: {e}")
            
            # points_veh 和 points_inf 的形状为 (4, N)，分别表示 [x, y, z, intensity]，需要转置为 (N, 4)
            points_veh = points_veh.T
            points_inf = points_inf.T

            # 转换为 numpy 格式，并处理维度
            points_veh_np = np.zeros((points_veh.shape[0], self.load_dim))
            points_veh_np[:, 0] = points_veh[:, 0]  # x
            points_veh_np[:, 1] = points_veh[:, 1]  # y
            points_veh_np[:, 2] = points_veh[:, 2]  # z
            points_veh_np[:, 3] = points_veh[:, 3]  # intensity
            # points_veh_np[:,4] = points_veh['timestamp']
            # results["timestamp_veh"] = points_veh['timestamp'][0]
            # results["timestamp"] = points_veh['timestamp']
            points_veh = points_veh_np[:, self.use_dim]
            
            points_inf_np = np.zeros((points_inf.shape[0], self.load_dim))
            points_inf_np[:, 0] = points_inf[:, 0]  # x
            points_inf_np[:, 1] = points_inf[:, 1]  # y
            points_inf_np[:, 2] = points_inf[:, 2]  # z
            points_inf_np[:, 3] = points_inf[:, 3]  # intensity
            points_inf = points_inf_np[:, self.use_dim]

            # points_veh = self._load_points(pts_filename_veh)
            # points_inf = self._load_points(pts_filename_inf)
            # points = points.reshape(-1, self.load_dim)
            # points_veh = points_veh.reshape(-1, self.load_dim)
            # points_inf = points_inf.reshape(-1, self.load_dim)
            # # points = points[:, self.use_dim]
            # points_veh = points_veh[:, self.use_dim]
            # points_inf = points_inf[:, self.use_dim]
            # attribute_dims = None
            attribute_dims_veh = None
            attribute_dims_inf = None

            if self.shift_height:
                # vehicle side
                floor_height_veh = np.percentile(points_veh[:, 2], 0.99)
                height_veh = points_veh[:, 2] - floor_height_veh
                points_veh = np.concatenate(
                    [points_veh[:, :3],
                    np.expand_dims(height_veh, 1), points_veh[:, 3:]], 1)
                attribute_dims_veh = dict(height=3)
                # infrastructure side
                floor_height_inf = np.percentile(points_inf[:, 2], 0.99)
                height_inf = points_inf[:, 2] - floor_height_inf
                points_inf = np.concatenate(
                    [points_inf[:, :3],
                    np.expand_dims(height_inf, 1), points_inf[:, 3:]], 1)
                attribute_dims_inf = dict(height=3)

            if self.use_color:
                assert len(self.use_dim) >= 6
                # vehicle side
                if attribute_dims_veh is None:
                    attribute_dims_veh = dict()
                attribute_dims_veh.update(
                    dict(color=[
                        points_veh.shape[1] - 3,
                        points_veh.shape[1] - 2,
                        points_veh.shape[1] - 1,
                    ]))
                # infrastructure side
                if attribute_dims_inf is None:
                    attribute_dims_inf = dict()
                attribute_dims_inf.update(
                    dict(color=[
                        points_inf.shape[1] - 3,
                        points_inf.shape[1] - 2,
                        points_inf.shape[1] - 1,
                    ]))

            points_class = get_points_type(self.coord_type)
            points_veh = points_class(points_veh, points_dim=points_veh.shape[-1], attribute_dims=attribute_dims_veh)
            points_inf = points_class(points_inf, points_dim=points_inf.shape[-1], attribute_dims=attribute_dims_inf)

            # results['filename'] = results["pc_filename"]
            results['points_veh'] = points_veh
            results['points_inf'] = points_inf
            # results['points'] = [points_veh, points_inf]
            # unravel to list, see `DefaultFormatBundle` in formating.py
            # which will transpose each image separately and then stack into array
            # results['img'] = [points_veh, points_inf]
            results['pseudoimg_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                    (
                        results["pc_range_inf"][4]-results["pc_range_inf"][1], 
                        results["pc_range_inf"][3]-results["pc_range_inf"][0],
                        results["pc_range_inf"][5]-results["pc_range_inf"][2]
                    ),
                ]
            results['pseudoimg_ori_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                    (
                        results["pc_range_inf"][4]-results["pc_range_inf"][1], 
                        results["pc_range_inf"][3]-results["pc_range_inf"][0],
                        results["pc_range_inf"][5]-results["pc_range_inf"][2]
                    ),
                ]
            # Set initial values for default meta_keys
            results['pseudoimg_pad_shape'] = [
                    (
                        results["pc_range_veh"][4]-results["pc_range_veh"][1], 
                        results["pc_range_veh"][3]-results["pc_range_veh"][0],
                        results["pc_range_veh"][5]-results["pc_range_veh"][2]
                    ),
                    (
                        results["pc_range_inf"][4]-results["pc_range_inf"][1], 
                        results["pc_range_inf"][3]-results["pc_range_inf"][0],
                        results["pc_range_inf"][5]-results["pc_range_inf"][2]
                    ),
                ]
            results['pseudoimg_scale_factor'] = 1.0

        return results


@PIPELINES.register_module()
class XETV2XLoadAnnotations3D(LoadAnnotations3D):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """
    def __init__(self,
                 with_ins_inds_3d=False,
                 ins_inds_add_1=False,  # NOTE: make ins_inds start from 1, not 0
                 **kwargs):
        super().__init__(**kwargs)
        self.with_ins_inds_3d = with_ins_inds_3d
        self.ins_inds_add_1 = ins_inds_add_1
  
    def _load_ins_inds_3d(self, results):
        ann_gt_inds = results['ann_info']['gt_inds'].copy() # TODO: note here

        # NOTE: Avoid gt_inds generated twice
        results['ann_info'].pop('gt_inds')
        
        if self.ins_inds_add_1:
            ann_gt_inds += 1
        results['gt_inds'] = ann_gt_inds
        return results

    def __call__(self, results):
        results = super().__call__(results)
        
        if self.with_ins_inds_3d:
            results = self._load_ins_inds_3d(results)
        
        return results

    def __repr__(self):
        repr_str = super().__repr__()
        indent_str = '    '
        repr_str += f'{indent_str}with_ins_inds_3d={self.with_ins_inds_3d}, '
        
        return repr_str