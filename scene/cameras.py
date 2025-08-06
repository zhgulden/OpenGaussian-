#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getCameraToPixelMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, cx, cy, image, depth, gt_alpha_mask,
                 gt_sam_mask, gt_mask_feat,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        # modify -----
        self.cx = cx
        self.cy = cy
        # modify -----
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.data_on_gpu = True     # note
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # modify -----
        self.original_mask = gt_alpha_mask.to(self.data_device) if gt_alpha_mask is not None else None
        
        # modify -----
        self.original_sam_mask = gt_sam_mask.to(self.data_device) if gt_sam_mask is not None else None
        self.original_mask_feat = gt_mask_feat.to(self.data_device) if gt_mask_feat is not None else None
        self.pesudo_ins_feat = None
        self.pesudo_mask_bool = None
        self.cluster_masks = None
        self.bClusterOccur = None

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # The same as above but not transposed (for our use purposes)
        self.world_view_transform_no_t = torch.tensor(getWorld2View2(R, T, trans, scale)).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # The same as above but not transposed (for our use purposes)
        self.projection_matrix_no_t = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.intrinsic_matrix = getCameraToPixelMatrix(
            self.image_width, self.image_height, self.FoVx, self.FoVy, self.cx, self.cy)
        
        # Attribute to estimate visible splats
        self.depth_map = None
        
        # Pixel lookup table attributes
        self.unique_ids = None
        # Structure of shape H,W,num_ids holding aggregated count for each channel as for each ID (map is self.id_to_idx)
        self.pixel_value_tensor = None
        self.id_to_idx = None
        # self.idx_to_id = None
        self.id_range = None
        
    # modify -----
    def to_gpu(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor) and not attr.is_cuda:
                setattr(self, attr_name, attr.to('cuda'))
        self.data_on_gpu = True

    # modify -----
    def to_cpu(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor) and attr.is_cuda:
                setattr(self, attr_name, attr.to('cpu'))
        self.data_on_gpu = False

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
