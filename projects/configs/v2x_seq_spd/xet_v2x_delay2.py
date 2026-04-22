_base_ = ["../_base_/datasets/nus-3d.py", "../_base_/default_runtime.py"]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range_inf = [0, -51.2, -5.0, 102.4, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
length = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
height = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
output_shape = [height, length]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

#class_range for eva
class_range={
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "construction_vehicle": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    "traffic_cone": 30,
    "barrier": 30
}

use_inf=True
use_camera=True
use_lidar=True
use_pc_veh=True
use_pc_inf=True
input_modality = dict(
    use_camera=use_camera, use_lidar=use_lidar, use_inf=use_inf, use_radar=False, use_map=False, use_external=True
)
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)

# NOTE: You can change queue_length from 5 to 3 to save GPU memory, but at risk of performance drop.
queue_length = 5  # each sequence contains `queue_length` frames.

### traj prediction args ###
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True


# Other settings
train_gt_iou_threshold = 0.3

model = dict(
    type="XETV2X",
    # pointcloud
    use_pc_veh=use_lidar and use_pc_veh,
    pc_range=point_cloud_range,
    voxel_layer=dict(
        max_num_points=100,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)
        ),
    voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        ),
    middle_encoder=dict(
        type="PointPillarsScatter",
        in_channels=64,
        output_shape=output_shape
        ),
    pc_backbone=dict(
        type="SECOND",
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]
        ),
    # pc_neck=dict(type="SECONDFPN", in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[256, 256, 256]),
    pc_neck=dict(
        type="FPN",
        in_channels=[64, 128, 256],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
        ),
    freeze_voxel_encoder=False,
    freeze_middle_encoder=False,
    freeze_pc_backbone=False,
    freeze_pc_neck=False,
    freeze_pc_bn=False,
    # pointcloud inf
    use_pc_inf=use_lidar and use_pc_inf,
    pc_range_inf=point_cloud_range_inf,
    voxel_layer_inf=dict(
        max_num_points=100,
        point_cloud_range=point_cloud_range_inf,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)
        ),
    voxel_encoder_inf=dict(
        type="PillarFeatureNet",
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range_inf,
        ),
    # image
    use_img=use_camera,
    use_grid_mask=True,
    img_backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(
            type="DCNv2",
            deform_groups=1,
            fallback_on_stride=False
            ),
        stage_with_dcn=(False, False, True, True),
        ),
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
        ),
    freeze_img_backbone=False,
    freeze_img_neck=False,
    freeze_img_bn=False,
    # temporal
    queue_length=queue_length,
    embed_dims=_dim_,
    num_query=900,
    num_classes=10,
    video_test_mode=True,
    gt_iou_threshold=train_gt_iou_threshold,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    #miss_tolerance=21,
    qim_args=dict(
        qim_type="QIMBase",
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
        ),
    mem_args=dict(
        memory_bank_type="MemoryBank",
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
        ),
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=10,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
            ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_past_traj_weight=0.0,
        ),  # loss cfg for tracking
    # bev
    pts_bbox_head=dict(
        type="XETV2XBEVFormerTrackHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        past_steps=past_steps,
        fut_steps=fut_steps,
        transformer=dict(
            type="XETV2XPerceptionTransformer",
            num_feature_levels=4,
            num_cams=2,
            num_lidars=2,
            use_cams_embeds=True,
            use_lidars_embeds=True,
            use_multimodal_embeds=True,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            use_timestamps=False,
            embed_dims=_dim_,
            encoder=dict(
                type="XETV2XBEVFormerEncoder",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                num_cams=2,
                num_lidars=2,
                use_calib_error_offset=True,
                use_cam_calib_error_offset=True,
                use_lidar_calib_error_offset=True,
                transformerlayers=dict(
                    type="XETV2XBEVFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention",
                            embed_dims=_dim_,
                            num_levels=1
                            ),
                        dict(
                            type="V2XSpatialCrossAttention",
                            num_cams=2,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                                ),
                            embed_dims=_dim_,
                            ),
                        dict(
                            type="V2XSpatialCrossAttention",
                            num_cams=2,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                                ),
                            embed_dims=_dim_,
                            ),
                        ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                        ),
                    ),
                ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                            ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                            ),
                        ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                        ),
                    ),
                ),
            ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
            ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
            ),
        loss_bbox=dict(
            type="L1Loss",
            loss_weight=0.25
            ),
        loss_iou=dict(
            type="GIoULoss",
            loss_weight=0.0
            ),
        ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost",
                    weight=0.0
                    ),
                pc_range=point_cloud_range,
                ),
            )
        ),
    )

file_client_args = dict(backend="disk")

dataset_type = "XETV2XSeqDataset"
data_root = "./datasets/V2X-Seq-SPD-Delay-2/cooperative/"
info_root = "./data/infos/V2X-Seq-SPD-Delay-2/cooperative/"
ann_file_train = info_root + f"spd_infos_temporal_train.pkl"
ann_file_val = info_root + f"spd_infos_temporal_val.pkl"
ann_file_test = info_root + f"spd_infos_temporal_val.pkl"

# for eval data
split_datas_file = "./data/split_datas/cooperative-split-data-spd-delay.json"

train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True, file_client_args=file_client_args, img_root=data_root),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="XETV2XLoadMultiViewPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
        pts_root = data_root
        ),
    dict(
        type="XETV2XPointsRangeFilter", 
        point_cloud_range=point_cloud_range,
        point_cloud_range_inf=point_cloud_range_inf,
        ),
    dict(
        type="XETV2XLoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_ins_inds_3d=True,  # ins_inds
        ins_inds_add_1=True,  # ins_inds start from 1
        ),
    dict(type="XETV2XObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="XETV2XObjectNameFilterTrack", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="XETV2XDefaultFormatBundle3D", class_names=class_names),
    dict(
        type="XETV2XCollect3D",
        use_camera=use_camera,
        use_lidar=use_lidar,
        keys=(
            "img",
            "points_veh",
            "points_inf",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_inds",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            "gt_past_traj",
            "gt_past_traj_mask",
            "gt_sdc_bbox",
            "gt_sdc_label",
            "gt_lane_labels",
            "gt_lane_bboxes",
            "gt_lane_masks",
            ),
        img_meta_keys=(
            'filename', 'lidar2img', 'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 
            'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 
            'pts_filename', 'scene_token', 'can_bus', "timestamps"
            ),
        pc_meta_keys=(
            'pc_filename', 'lidar2pseudoimg', 'pseudoimg_shape', 'pseudoimg_ori_shape', 'pseudoimg_pad_shape', 
            'pseudoimg_scale_factor', 'box_mode_3d', 'box_type_3d', 'sample_idx', 'prev_idx', 'next_idx', 
            'pts_filename', 'scene_token', 'can_bus', "timestamps"
            )
        ),
    ]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True, file_client_args=file_client_args, img_root=data_root),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="XETV2XLoadMultiViewPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
        pts_root=data_root
        ),
    dict(
        type="XETV2XPointsRangeFilter", 
        point_cloud_range=point_cloud_range,
        point_cloud_range_inf=point_cloud_range_inf,
        ),
    dict(
        type="XETV2XLoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_ins_inds_3d=True,  # ins_inds
        ins_inds_add_1=True,  # ins_inds start from 1
        ),
    dict(type="XETV2XObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="XETV2XObjectNameFilterTrack", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="XETV2XDefaultFormatBundle3D", class_names=class_names),
    dict(
        type="XETV2XCollect3D",
        use_camera=use_camera,
        use_lidar=use_lidar,
        keys=(
            "img",
            "points_veh",
            "points_inf",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_inds",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            "gt_past_traj",
            "gt_past_traj_mask",
            "gt_sdc_bbox",
            "gt_sdc_label",
            "gt_lane_labels",
            "gt_lane_bboxes",
            "gt_lane_masks",
            ),
        img_meta_keys=(
            'filename', 'lidar2img', 'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 
            'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 
            'pts_filename', 'scene_token', 'can_bus', "timestamps"
            ),
        pc_meta_keys=(
            'pc_filename', 'lidar2pseudoimg', 'pseudoimg_shape', 'pseudoimg_ori_shape', 'pseudoimg_pad_shape', 
            'pseudoimg_scale_factor', 'box_mode_3d', 'box_type_3d', 'sample_idx', 'prev_idx', 'next_idx', 
            'pts_filename', 'scene_token', 'can_bus', "timestamps"
            )
        ),
    ]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        split_datas_file=split_datas_file,
        v2x_side='vehicle_side',
        class_range=class_range,
        pc_range_veh=point_cloud_range,
        pc_range_inf=point_cloud_range_inf
        ),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=['det', 'track'],
        split_datas_file=split_datas_file,
        v2x_side='vehicle_side',
        class_range=class_range,
        pc_range_veh=point_cloud_range,
        pc_range_inf=point_cloud_range_inf
        ),
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        eval_mod=['det', 'track'],
        split_datas_file=split_datas_file,
        v2x_side='vehicle_side',
        class_range=class_range,
        pc_range_veh=point_cloud_range,
        pc_range_inf=point_cloud_range_inf
        ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
    )
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            # "pc_backbone": dict(lr_mult=0.1),
            }
        ),
    weight_decay=0.01,
    )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

total_epochs = 10
# evaluation = dict(interval=5, pipeline=test_pipeline)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
    )
checkpoint_config = dict(interval=1)
load_from = "ckpts/bevformer_r101_dcn_24ep.pth"

find_unused_parameters = True
