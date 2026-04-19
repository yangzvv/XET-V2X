#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import copy
import math
import os 
# from .uniad_track import  UniADTrack
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models import build_loss
from einops import rearrange
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models.builder import build_backbone, build_neck, build_voxel_encoder, build_middle_encoder
import mmcv
from ..dense_heads.track_head_plugin import MemoryBank, QueryInteractionModule, Instances, RuntimeTrackerBase
from ..modules import CrossAgentSparseInteraction
from mmdet3d.ops import Voxelization
from torch.nn import functional as F

@DETECTORS.register_module()
class XETVIC(MVXTwoStageDetector):
    def __init__(
        self,
        # pointcloud
        use_pc_veh=True,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_layer=None,
        voxel_encoder=None,
        middle_encoder=None,
        pc_backbone=None,
        pc_neck=None,
        freeze_voxel_encoder=False,
        freeze_middle_encoder=False,
        freeze_pc_backbone=False,
        freeze_pc_neck=False,
        freeze_pc_bn=False,
        # pointcloud inf
        use_pc_inf=False,
        pc_range_inf=[0, -51.2, -5.0, 102.4, 51.2, 3.0],
        voxel_layer_inf=None,
        voxel_encoder_inf=None,
        # image
        use_img=False,
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_img_bn=False,
        # temporal
        queue_length=3,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        video_test_mode=False,
        gt_iou_threshold=0.0,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        miss_tolerance=5,
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=False,
            fp_ratio=0.3,
            random_drop=0.1,
            ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
            ),
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
            ),
        # bev
        pts_bbox_head=None,
        freeze_bev_encoder=False,
        # criterion
        loss_cfg=None,
        task_loss_weight=dict(
            track=1.0,
            ),
        # model training and testing settings
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        **kwargs,
        ):
        super(XETVIC, self).__init__(
            # img_backbone=img_backbone,
            # img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            )
        self.pc_range = pc_range

        # >>> pointcloud
        self.use_pc_veh = use_pc_veh
        self.use_pc_inf = use_pc_inf
        if use_pc_veh:
            self.voxel_layer = Voxelization(**voxel_layer)
            self.voxel_encoder = build_voxel_encoder(voxel_encoder)
            if freeze_voxel_encoder:
                if freeze_pc_bn:
                    self.voxel_encoder.eval()
                for param in self.voxel_encoder.parameters():
                    param.requires_grad = False
        if use_pc_inf:
            self.pc_range_inf = pc_range_inf
            self.voxel_layer_inf = Voxelization(**voxel_layer_inf)
            self.voxel_encoder_inf = build_voxel_encoder(voxel_encoder_inf)
            if freeze_voxel_encoder:
                if freeze_pc_bn:
                    self.voxel_encoder_inf.eval()
                for param_inf in self.voxel_encoder_inf.parameters():
                    param_inf.requires_grad = False
        if middle_encoder:
            self.middle_encoder = build_middle_encoder(middle_encoder)
            if freeze_middle_encoder:
                if freeze_pc_bn:
                    self.middle_encoder.eval()
                for param in self.middle_encoder.parameters():
                    param.requires_grad = False
        if pc_backbone:
            self.pc_backbone = build_backbone(pc_backbone)
            if freeze_pc_backbone:
                if freeze_pc_bn:
                    self.pc_backbone.eval()
                for param in self.pc_backbone.parameters():
                    param.requires_grad = False
        if pc_neck:
            self.neck = build_neck(pc_neck)
            if freeze_pc_neck:
                if freeze_pc_bn:
                    self.neck.eval()
                for param in self.neck.parameters():
                    param.requires_grad = False
        # <<<
        
        # >>> image
        self.use_img = use_img
        self.use_grid_mask = use_grid_mask
        if use_img:
            if img_backbone:
                self.img_backbone = build_backbone(img_backbone)
                if freeze_img_backbone:
                    if freeze_img_bn:
                        self.img_backbone.eval()
                    for param in self.img_backbone.parameters():
                        param.requires_grad = False
            if img_neck is not None:
                self.img_neck = build_neck(img_neck)
                if freeze_img_neck:
                    if freeze_img_bn:
                        self.img_neck.eval()
                    for param in self.img_neck.parameters():
                        param.requires_grad = False
            if use_grid_mask:
                self.grid_mask = GridMask(
                    True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
                    )
        # <<<
        
        # >>> temporal
        self.queue_length = queue_length
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.video_test_mode = video_test_mode
        self.gt_iou_threshold = gt_iou_threshold
        assert self.video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.query_embedding = nn.Embedding(self.num_query+1, self.embed_dims * 2)   # the final one is ego query, which constantly models ego-vehicle
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )  # hyper-param for removing inactive queries

        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        bbox_coder['pc_range'] = pc_range
        bbox_coder['post_center_range'] = post_center_range
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )
        self.test_track_instances = None
        # <<<

        # >>> bev
        self.bev_h = self.pts_bbox_head.bev_h
        self.bev_w = self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder
        # <<<

        # >>> criterion
        self.criterion = build_loss(loss_cfg)
        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == {'track'}   
        # <<<
        
        # >>> matrix
        self.l2g_r_mat = None
        self.l2g_t = None
        # <<<

        # self.idx_v2x_sim = 0  # yzw 2025
        
    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        if img is None:
            return None
        assert img.dim() == 5
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B//len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)  # yzw 901 512
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz  # yzw: x, y, z, l, w, h, sinθ, cosθ, vx, vy
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.query = query

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )

        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embedding.weight.device)

    def velo_update(
        self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
    ):
        """
        Args:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
            velocity (Tensor): (num_query, 2). m/s
                in lidar frame. vx, vy
            global2lidar (np.Array) [4,4].
        Outs:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
        """
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = reference_points + velo_pad * time_delta

        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2

        g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)

        ref_pts = ref_pts @ g2l_r

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0]
        )
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1]
        )
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2]
        )

        ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts

    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res = res.contiguous()
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def voxelize_inf(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res = res.contiguous()
            res_voxels, res_coors, res_num_points = self.voxel_layer_inf(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)

        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances.to(device)
    
    def extract_pc_feat_queue(self, pc_queue, len_queue=None):
        num_lidars = len(pc_queue)
        if len_queue is not None:
            pc_feats_list = []
            for i in range(len_queue):
                pc_feats_list_single = []
                if num_lidars == 1:
                    pc_feats = self.extract_pc_feat(pc=[pc_queue[0][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list_single.append(pc_feats)
                elif num_lidars == 2:
                    pc_feats = self.extract_pc_feat(pc=[pc_queue[0][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list_single.append(pc_feats)
                    pc_feats_inf = self.extract_pc_feat_inf(pc=[pc_queue[1][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list_single.append(pc_feats_inf)
                # for pc_queue_vi in pc_queue:
                #     pc_feats = self.extract_pc_feat(pc=[pc_queue_vi[i][:,0:4]])  # bs, 1, C, H, W
                #     pc_feats_list_single.append(pc_feats)
                # pc_feats_list.append(pc_feats_list_vi)

                pc_feats_list_single_combined = []
                for lvl in range(len(pc_feats_list_single[0])):
                    pc_feats_single_qn_combined = torch.cat([pc_feats_list_single[j][lvl] for j in range(len(pc_feats_list_single))], dim=1)
                    pc_feats_single_qn_combined = torch.unsqueeze(pc_feats_single_qn_combined, dim=1)
                    pc_feats_list_single_combined.append(pc_feats_single_qn_combined)
                pc_feats_list.append(pc_feats_list_single_combined)
            pc_feats_list_combined = []
            for i in range(len(pc_feats_list[0])):
                pc_feats_single_q_combined = torch.cat([pc_feats_list[j][i] for j in range(len_queue)], dim=1)
                pc_feats_list_combined.append(pc_feats_single_q_combined)
            return pc_feats_list_combined
        else:
            pc_feats_list = []
            len_queue = len(pc_queue[0])
            for i in range(len_queue):
                if num_lidars == 1:
                    pc_feats = self.extract_pc_feat(pc=[pc_queue[0][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list.append(pc_feats)
                elif num_lidars == 2:
                    pc_feats = self.extract_pc_feat(pc=[pc_queue[0][i][:,0:4]])  # bs, 1, C, H, W
                    # if self.idx_v2x_sim < 100:  # yzw 2025
                    #     torch.save(pc_feats, f'/data/kongzhi/yangzhenwei/coding-projects/XET-VIC/datasets/V2X-Sim-PC-Feats/feature_pyramid_{self.idx_v2x_sim}.pt')  # yzw 2025
                    #     self.idx_v2x_sim += 1  # yzw 2025
                    pc_feats_list.append(pc_feats)
                    pc_feats_inf = self.extract_pc_feat_inf(pc=[pc_queue[1][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list.append(pc_feats_inf)
                # for pc_queue_vi in pc_queue:
                #     pc_feats = self.extract_pc_feat(pc=[pc_queue_vi[i][:,0:4]])  # bs, 1, C, H, W
                #     pc_feats_list.append(pc_feats)
            pc_feats_list_combined = []
            for i in range(len(pc_feats_list[0])):
                pc_feats_combined = torch.cat([pc_feats_list[j][i] for j in range(len(pc_feats_list))], dim=1)
                pc_feats_list_combined.append(pc_feats_combined)
            return pc_feats_list_combined
    
    def extract_pc_feat(self, pc):
        """Extract features of pointclouds."""
        voxels, num_points, coors = self.voxelize(pc)
        voxel_features = self.voxel_encoder(voxels, num_points, coors) # (10600,4)
        batch_size = coors[-1, 0].item() + 1  
        x = self.middle_encoder(voxel_features, coors, batch_size) #([1, 64, 512, 512])
        x = self.pc_backbone(x) # ([1, 64, 256, 256]),(1,128,128,128),(1,256,64,64)
        if self.with_neck:
            x = list(self.neck(x)) # ([1, 3*256, 256, 256])  NCHW
        for i in range(len(x)):
            x[i] = x[i].unsqueeze(0)
        return x
    
    def extract_pc_feat_inf(self, pc):
        """Extract features of pointclouds."""
        voxels, num_points, coors = self.voxelize_inf(pc)
        voxel_features = self.voxel_encoder_inf(voxels, num_points, coors) # (10600,4)
        batch_size = coors[-1, 0].item() + 1  
        x = self.middle_encoder(voxel_features, coors, batch_size) #([1, 64, 512, 512])
        x = self.pc_backbone(x) # ([1, 64, 256, 256]),(1,128,128,128),(1,256,64,64)
        if self.with_neck:
            x = list(self.neck(x)) # ([1, 3*256, 256, 256])  NCHW
        for i in range(len(x)):
            x[i] = x[i].unsqueeze(0)
        return x
    
    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict
    
    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict['boxes_3d']
        out["sdc_scores_3d"] = result_dict['scores_3d']
        out["sdc_track_scores"] = result_dict['track_scores']
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out
    
    def upsample_bev_if_tiny(self, outs_track):
        if outs_track["bev_embed"].size(0) == 100 * 100:
            # For tiny model
            # bev_emb
            bev_embed = outs_track["bev_embed"] # [10000, 1, 256]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(bev_embed, '(h w) b c -> b c h w', h=h, w=w)  # [1, 256, 100, 100]
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)  # [1, 256, 200, 200]
            bev_embed = rearrange(bev_embed, 'b c h w -> (h w) b c')
            outs_track["bev_embed"] = bev_embed

            # prev_bev
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if self.training:
                    #  [1, 10000, 256]
                    prev_bev = rearrange(prev_bev, 'b (h w) c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> b (h w) c')
                    outs_track["prev_bev"] = prev_bev
                else:
                    #  [10000, 1, 256]
                    prev_bev = rearrange(prev_bev, '(h w) b c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> (h w) b c')
                    outs_track["prev_bev"] = prev_bev

            # bev_pos
            bev_pos  = outs_track["bev_pos"]  # [1, 256, 100, 100]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track
    
    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        # bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask)[0]
        bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # filter out sleep querys
        if instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]

    def forward(self, return_loss=True, **kwargs):
        """
            Calls either forward_train or forward_test depending on whether return_loss=True.
            Note this setting will change the expected inputs. 
            When `return_loss=True`, img and img_metas are single-nested (i.e. torch.Tensor and list[dict]), 
            and when `resturn_loss=False`, img and img_metas should be double nested (i.e.  list[torch.Tensor], list[list[dict]]), 
            with the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=("img", "points_veh", "points_inf"))
    def forward_train(
        self,
        img=None,
        img_metas=None,
        points_veh=None,
        points_inf=None,
        pc_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        gt_instance=None,
        **kwargs,
        ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
            gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
            gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
            gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
            gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
            gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
            gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.
            
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        losses = dict()
        pc = []
        if self.use_pc_veh:
            pc.append(points_veh)
        if self.use_pc_inf:
            pc.append(points_inf)
        pc = pc if pc else None
        losses_track, outs_track = self.forward_track_train(img, pc, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, pc_metas, timestamp)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)
        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)
        return losses
    
    @auto_fp16(apply_to=("img", "pc"))
    def forward_track_train(self,
                            img,
                            pc,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            gt_past_traj,
                            gt_past_traj_mask,
                            gt_inds,
                            gt_sdc_bbox,
                            gt_sdc_label,
                            l2g_t,
                            l2g_r_mat,
                            img_metas,
                            pc_metas,
                            timestamp):
        """Forward funciton
        Args:
        Returns:
        """
        track_instances = self._generate_empty_tracks()
        if self.use_img:
            num_frame = img.size(1)
            device = img.device
        elif self.use_pc_veh or self.use_pc_inf:
            num_frame = len(pc[0][0])
            device = pc[0][0][0].device
        # init gt instances!
        gt_instances_list = []
        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(device)
            # normalize gt bboxes here!
            boxes = normalize_bbox(boxes, self.pc_range)
            sd_boxes = gt_sdc_bbox[0][i].tensor.to(device)
            sd_boxes = normalize_bbox(sd_boxes, self.pc_range)
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = gt_inds[0][i]
            gt_instances.past_traj = gt_past_traj[0][i].float()
            gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()
            gt_instances.sdc_boxes = torch.cat([sd_boxes for _ in range(boxes.shape[0])], dim=0)  # boxes.shape[0] sometimes 0
            gt_instances.sdc_labels = torch.cat([gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0)
            gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        out = dict()
        for i in range(num_frame):
            if self.use_img:
                prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
                prev_img_metas = copy.deepcopy(img_metas)
                single_img = torch.stack([img_[i] for img_ in img], dim=0)
                single_metas_img = [copy.deepcopy(img_metas[0][i])]
                used_metas = single_metas_img
            else:
                prev_img = None
                prev_img_metas = None
                single_img = None
                single_metas_img = None
            if self.use_pc_veh or self.use_pc_inf:
                prev_pc = []
                prev_pc_metas = copy.deepcopy(pc_metas)
                single_pc = []
                single_pc_metas = [copy.deepcopy(pc_metas[0][i])]
                used_metas = single_pc_metas
                for pc_vi in pc:
                    if pc_vi is None:
                        continue
                    prev_pc_vi = pc_vi[0][:i] if i != 0 else pc_vi[0][:1]
                    single_pc_vi = torch.stack([pc_[i] for pc_ in pc_vi], dim=0)
                    prev_pc.append(prev_pc_vi)
                    single_pc.append(single_pc_vi)
            else:
                prev_pc = None
                prev_pc_metas = None
                single_pc = None
                single_pc_metas = None 

            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else: 
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []
            frame_res = self._forward_single_frame_train(
                single_img,
                prev_img,
                single_metas_img,
                prev_img_metas,
                single_pc,
                prev_pc,
                single_pc_metas,
                prev_pc_metas,
                used_metas,
                track_instances,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes
            )
            # all_query_embeddings: len=dec nums, N*256
            # all_matched_idxes: len=dec nums, N*2
            track_instances = frame_res["track_instances"]

        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                    "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        out.update({k: frame_res[k] for k in get_keys})
        
        losses = self.criterion.losses_dict
        return losses, out
    
    def _forward_single_frame_train(
        self,
        img,
        prev_img,
        img_metas,
        prev_img_metas,
        pc,
        prev_pc,
        pc_metas,
        prev_pc_metas,
        used_metas,
        track_instances,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None
        ):
        """
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """
        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(
            img,
            img_metas,
            prev_img,
            prev_img_metas,
            pc, 
            pc_metas,
            prev_pc,
            prev_pc_metas,
        )

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=used_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
        if l2g_r2 is not None:
            # Update ref_pts for next frame considering each agent's velocity
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
            )
        else:
            ref_pts = last_ref_pts[0]

        # track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
        track_instances.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances_list.append(track_instances)
        
        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
            track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300,past_steps, 2]

            out["track_instances"] = track_instances
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])   # Not used
        
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        out.update(self.select_active_track_query(track_instances, active_index, used_metas))
        # out.update(self.select_sdc_track_query(track_instances[900], img_metas))
        out.update(self.select_sdc_track_query(track_instances[-1], used_metas))
        # memory bank 
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        # Step-2 Update track instances using matcher

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        return out
    
    def forward_test(
        self,
        img=None,
        img_metas=None,
        points_veh = None,
        points_inf = None,
        pc_metas=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        rescale=False,
        **kwargs
        ):
        """Test function
        """
        if self.use_img:
            if not isinstance(img_metas, list):
                raise TypeError('img_metas must be a list, but got {}'.format(type(img_metas)))
            img = img if img is not None else []

            if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
                # the first sample of each scene is truncated
                self.prev_frame_info['prev_bev'] = None
            # update idx
            self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

            # do not use temporal information
            if not self.video_test_mode:
                self.prev_frame_info['prev_bev'] = None

            # Get the delta of ego position and angle between two timestamps.
            tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
            # first frame
            if self.prev_frame_info['scene_token'] is None:
                img_metas[0][0]['can_bus'][:3] = 0
                img_metas[0][0]['can_bus'][-1] = 0
            # following frames
            else:
                img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
                img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
            self.prev_frame_info['prev_pos'] = tmp_pos
            self.prev_frame_info['prev_angle'] = tmp_angle

            img_metas = img_metas[0]
            used_metas = img_metas

        else:
            img = None
            img_metas = None

        if self.use_pc_veh or self.use_pc_inf:
            if not isinstance(pc_metas, list):
                raise TypeError('pc_metas must be a list, but got {}'.format(type(pc_metas)))
            if self.use_pc_veh and self.use_pc_inf:
                pc = [points_veh, points_inf] 
            elif self.use_pc_veh:
                pc = [points_veh]
            elif self.use_pc_inf:
                pc = [points_inf]
            
            if pc_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
                # the first sample of each scene is truncated
                self.prev_frame_info['prev_bev'] = None
            # update idx
            self.prev_frame_info['scene_token'] = pc_metas[0][0]['scene_token']

            # do not use temporal information
            if not self.video_test_mode:
                self.prev_frame_info['prev_bev'] = None

            # Get the delta of ego position and angle between two timestamps.
            tmp_pos = copy.deepcopy(pc_metas[0][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(pc_metas[0][0]['can_bus'][-1])
            # first frame
            if self.prev_frame_info['scene_token'] is None:
                pc_metas[0][0]['can_bus'][:3] = 0
                pc_metas[0][0]['can_bus'][-1] = 0
            # following frames
            else:
                pc_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
                pc_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
            self.prev_frame_info['prev_pos'] = tmp_pos
            self.prev_frame_info['prev_angle'] = tmp_angle

            pc_metas = pc_metas[0]
            used_metas = pc_metas

        else:
            pc = None
            pc_metas = None

        
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(used_metas))]
        result_track = self.simple_test_track(img, img_metas, pc, pc_metas, used_metas, l2g_t, l2g_r_mat, timestamp)

        # Upsample bev for tiny model        
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        for i, res in enumerate(result):
            res['token'] = used_metas[i]['sample_idx']
            res.update(result_track[i])

        return result
    
    def simple_test_track(
        self,
        img=None,
        img_metas=None,
        pc=None,
        pc_metas=None,
        used_metas=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None
        ):
        """only support bs=1 and sequential input"""
        if self.use_img:
            bs = img.size(0)
        else:
            bs = len(pc[0])

        """ init track instances for first frame """
        if (self.test_track_instances is None or used_metas[0]["scene_token"][-2:] != self.scene_token[-2:]):
            self.timestamp = timestamp
            self.scene_token = used_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
        
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            pc,
            pc_metas,
            used_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]

        self.test_track_instances = track_instances
                
        results = [dict()]
        get_keys = ["bev_embed", "bev_pos", 
                    "track_query_embeddings", "track_bbox_results", 
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        results[0].update({k: frame_res[k] for k in get_keys})

        results = self._det_instances2results(track_instances_fordet, results, used_metas)
        return results

    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        pc,
        pc_metas,
        used_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None
        ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            # active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])                  
            active_inst.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(
            img=img,
            img_metas=img_metas,
            pc=pc, 
            pc_metas=pc_metas,
            prev_bev=prev_bev,  # yzw
            )

        det_output = self.pts_bbox_head.get_detections(
            bev_embed, 
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=used_metas,
        )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query 
        track_instances.obj_idxes[900] = -2
        """ update track base """
        self.track_base.update(track_instances, None)
       
        active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)    # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, used_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], used_metas))

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out
    
    def forward_dummy(
        self,
        img=None,
        points_veh=None,  # List[Tensor]: [torch.Size([B, N_veh, 5])], N_veh动态变化
        points_inf=None,  # List[Tensor]: [torch.Size([B, N_inf, 5])], N_inf动态变化
        l2g_t=None,       # Tensor: torch.Size([B, 3])
        l2g_r_mat=None,   # Tensor: torch.Size([B, 3, 3])
        timestamp=None,   # List[Tensor]: [torch.Size([B])]
        **kwargs
        ):
        """动态点云输入的简化前向传播，用于FLOPs计算
        
        Args:
            points_veh (List[Tensor]): 车端点云列表，每个元素形状为 (B, N_veh, 5)
            points_inf (List[Tensor]): 路端点云列表，每个元素形状为 (B, N_inf, 5)
            l2g_t (Tensor): 局部到全局平移，形状 (B, 3)
            l2g_r_mat (Tensor): 局部到全局旋转矩阵，形状 (B, 3, 3)
            timestamp (List[Tensor]): 时间戳列表，每个元素形状 (B,)

        Returns:
            Tensor: 主干网络输出（如BEV特征）
        """
        # ------------------------------
        # 1. 动态点云输入统一处理
        # ------------------------------
        batch_size = 1  # FLOPs计算默认单样本
        
        # 确保点云输入为Tensor（若为List则取首个样本）
        points_veh = points_veh[0] if isinstance(points_veh, list) else points_veh  # (B, N_veh, 5)
        points_inf = points_inf[0] if isinstance(points_inf, list) else points_inf  # (B, N_inf, 5)
        
        # 动态点云截断（确保不超过模型最大处理点数）
        max_points_veh = self.pcd_encoder_veh.max_num_points  # 例如 60000
        max_points_inf = self.pcd_encoder_inf.max_num_points  # 例如 45000
        points_veh = points_veh[:, :max_points_veh, :] if points_veh is not None else None
        points_inf = points_inf[:, :max_points_inf, :] if points_inf is not None else None

        # ------------------------------
        # 2. 虚拟元数据构造（适配XETVICCollect3D格式）
        # ------------------------------
        dummy_metas = [{
            'pc_filename': ['dummy_veh.pcd', 'dummy_inf.pcd'],
            'lidar2pseudoimg': torch.eye(4),  # 单位矩阵占位
            'pseudoimg_shape': (1024, 1024, 8),
            'pseudoimg_ori_shape': (1024, 1024, 8),
            'pseudoimg_pad_shape': (1024, 1024, 8),
            'pseudoimg_scale_factor': 1.0,
            'box_mode_3d': 0,  # LiDAR坐标系
            'box_type_3d': 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes',
            'sample_idx': '000000',
            'prev_idx': '',
            'next_idx': '000001',
            'pts_filename': 'dummy.bin',
            'scene_token': 'dummy_scene',
            'can_bus': torch.zeros(18),  # 符合XETVIC配置中的pc_meta_keys
            'timestamps': [0.0, 0.0]     # 时间戳对齐占位
        } for _ in range(batch_size)]

        # ------------------------------
        # 3. 多模态输入组合与预处理
        # ------------------------------
        # 组合点云输入（适配实际模型接口）
        pc = []
        if self.use_pc_veh and points_veh is not None:
            pc.append(points_veh)
        if self.use_pc_inf and points_inf is not None:
            pc.append(points_inf)
        pc = pc if len(pc) > 0 else None

        # 强制关闭时序依赖（确保单帧独立计算）
        original_video_mode = self.video_test_mode
        self.video_test_mode = False
        self.prev_frame_info['prev_bev'] = None

        # ------------------------------
        # 4. 核心计算路径（与FLOPs相关）
        # ------------------------------
        # 调用主干网络（适配实际返回结构）
        output = self.simple_test_track(
            img=img,
            img_metas=[dummy_metas] if self.use_img else None,
            pc=pc,
            pc_metas=[dummy_metas] if (self.use_pc_veh or self.use_pc_inf) else None,
            used_metas=dummy_metas,
            l2g_t=l2g_t,
            l2g_r_mat=l2g_r_mat,
            timestamp=timestamp
        )
        
        # 提取BEV特征（根据实际输出结构调整）
        bev_embed = output[0].get("bev_embed", None)
        if bev_embed is None:
            bev_embed = output[0]["head_cls"]  # 后备方案：假设输出包含分类头logits

        # 恢复原始模式（可选）
        self.video_test_mode = original_video_mode

        return bev_embed

    # def get_history_bev(self, imgs_queue, img_metas_list):
    #     """
    #     Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
    #     """
    #     self.eval()
    #     with torch.no_grad():
    #         prev_bev = None
    #         bs, len_queue, num_cams, C, H, W = imgs_queue.shape
    #         imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
    #         img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
    #         for i in range(len_queue):
    #             img_metas = [each[i] for each in img_metas_list]
    #             img_feats = [each_scale[:, i] for each_scale in img_feats_list]
    #             prev_bev, _ = self.pts_bbox_head.get_bev_features(
    #                 mlvl_feats=img_feats, 
    #                 img_metas=img_metas, 
    #                 prev_bev=prev_bev)
    #     self.train()
    #     return prev_bev

    # # Generate bev using bev_encoder in BEVFormer
    # def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
    #     if prev_img is not None and prev_img_metas is not None:
    #         assert prev_bev is None
    #         prev_bev = self.get_history_bev(prev_img, prev_img_metas)

    #     img_feats = self.extract_img_feat(img=imgs)
    #     if self.freeze_bev_encoder:
    #         with torch.no_grad():
    #             bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
    #                 mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
    #     else:
    #          bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
    #                 mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        
    #     if bev_embed.shape[1] == self.bev_h * self.bev_w:
    #         bev_embed = bev_embed.permute(1, 0, 2)
        
    #     assert bev_embed.shape[0] == self.bev_h * self.bev_w
    #     return bev_embed, bev_pos

    @auto_fp16(apply_to=("img", "prev_img", "pc", "prev_pc", "prev_bev"))
    def get_bevs(
        self, 
        img=None,
        img_metas=None,
        prev_img=None,
        prev_img_metas=None,
        pc=None, 
        pc_metas=None, 
        prev_pc=None, 
        prev_pc_metas=None, 
        prev_bev=None
        ):
        use_img_condition = self.use_img and prev_img is not None and prev_img_metas is not None
        use_pc_condition = (self.use_pc_veh or self.use_pc_inf) and prev_pc is not None and prev_pc_metas is not None
        if use_img_condition or use_pc_condition:
            assert prev_bev is None, "prev_bev should be None before generating history BEV."
            prev_bev = self.get_history_bev(prev_img, prev_img_metas, prev_pc, prev_pc_metas)
        mmdl_mlvl_feats = []
        mmdl_metas = []
        if self.use_img:
            img_feats = self.extract_img_feat(img=img)
            mmdl_mlvl_feats.append(img_feats)
            mmdl_metas.append(img_metas)
        if self.use_pc_veh or self.use_pc_inf:
            pc_feats = self.extract_pc_feat_queue(pc_queue=pc)
            mmdl_mlvl_feats.append(pc_feats)
            mmdl_metas.append(pc_metas)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                # bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(mlvl_feats=pc_feats, img_metas=pc_metas, prev_bev=prev_bev)
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mmdl_mlvl_feats=mmdl_mlvl_feats,
                    mmdl_metas=mmdl_metas, 
                    prev_bev=prev_bev
                    )
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                mmdl_mlvl_feats=mmdl_mlvl_feats,
                mmdl_metas=mmdl_metas, 
                prev_bev=prev_bev
                )
        
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos
    
    # def get_history_bev(self, pc_queue, img_metas_list):
    #     """
    #     Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
    #     """
    #     self.eval()
    #     with torch.no_grad():
    #         prev_bev = None
    #         len_queue = len(pc_queue[0])
    #         pc_feats_list = self.extract_pc_feat_queue(pc_queue, len_queue)
    #         for i in range(len_queue):
    #             img_metas = [each[i] for each in img_metas_list]
    #             pc_feats = [each_scale[:, i] for each_scale in pc_feats_list]
    #             # pc_feats = pc_feats_list[i]
    #             prev_bev, _ = self.pts_bbox_head.get_bev_features(mlvl_feats=pc_feats, img_metas=img_metas, prev_bev=prev_bev)
    #     self.train()
    #     return prev_bev

    def get_history_bev(self, img_queue, img_metas_list, pc_queue, pc_metas_list):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            mmdl_mlvl_feats_list = []
            mmdl_metas_list = []
            if self.use_img:
                bs, len_queue, num_cams, C, H, W = img_queue.shape
                img_queue = img_queue.reshape(bs * len_queue, num_cams, C, H, W)
                img_feats_list = self.extract_img_feat(img=img_queue, len_queue=len_queue)
                mmdl_mlvl_feats_list.append(img_feats_list)
                mmdl_metas_list.append(img_metas_list)
            if self.use_pc_veh or self.use_pc_inf:
                len_queue = len(pc_queue[0])
                pc_feats_list = self.extract_pc_feat_queue(pc_queue, len_queue)
                mmdl_mlvl_feats_list.append(pc_feats_list)
                mmdl_metas_list.append(pc_metas_list)
            for i in range(len_queue):
                mmdl_mlvl_feats = [[each_scale[:, i] for each_scale in feats_list] for feats_list in mmdl_mlvl_feats_list]
                mmdl_metas = [[each[i] for each in metas_list] for metas_list in mmdl_metas_list]
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mmdl_mlvl_feats=mmdl_mlvl_feats, 
                    mmdl_metas=mmdl_metas,
                    prev_bev=prev_bev
                    )
            # if self.use_img and (not self.use_pc_veh) and (not self.use_pc_inf):
            #     for i in range(len_queue):
            #         img_metas = [each[i] for each in img_metas_list]
            #         img_feats = [each_scale[:, i] for each_scale in img_feats_list]
            #         pc_metas = None
            #         pc_feats = None
            #         prev_bev, _ = self.pts_bbox_head.get_bev_features(img_mlvl_feats=img_feats, img_metas=img_metas, pc_mlvl_feats=pc_feats, pc_metas=pc_metas, prev_bev=prev_bev)
            # elif self.use_pc_veh or self.use_pc_inf and (not self.use_img):
            #     for i in range(len_queue):
            #         img_metas = None
            #         img_feats = None
            #         pc_metas = [each[i] for each in pc_metas_list]
            #         pc_feats = [each_scale[:, i] for each_scale in pc_feats_list]
            #         prev_bev, _ = self.pts_bbox_head.get_bev_features(img_mlvl_feats=img_feats, img_metas=img_metas, pc_mlvl_feats=pc_feats, pc_metas=pc_metas, prev_bev=prev_bev)
            # else:
            #     for i in range(len_queue):
            #         img_metas = [each[i] for each in img_metas_list]
            #         img_feats = [each_scale[:, i] for each_scale in img_feats_list]
            #         pc_metas = [each[i] for each in pc_metas_list]
            #         pc_feats = [each_scale[:, i] for each_scale in pc_feats_list]
            #         prev_bev, _ = self.pts_bbox_head.get_bev_features(img_mlvl_feats=img_feats, img_metas=img_metas, pc_mlvl_feats=pc_feats, pc_metas=pc_metas, prev_bev=prev_bev)
        self.train()
        return prev_bev
    
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict
    
    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """

        from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
        import mmcv
        from os import path as osp
        from mmcv.parallel import DataContainer as DC
        
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            # inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            # pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
            inds = result[batch_id]['scores_3d'].numpy() > 0.1
            pred_bboxes = result[batch_id]['boxes_3d'][inds]

            if len(pred_bboxes) <= 0:
                continue

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for convertion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)


def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
