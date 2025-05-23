# Copyright (c) OpenMMLab. All rights reserved.
import os, copy, math

import torch
import torch.nn as nn
from mmcv.cnn import Scale, ConvModule
from mmcv.runner import force_fp32
from mmrotate.models.dense_heads.rotated_anchor_free_head import RotatedAnchorFreeHead
from mmdet.core import multi_apply, reduce_mean

from mmrotate.models.builder import ROTATED_HEADS, build_loss
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmrotate.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma, gwd_loss

INF = 1e8


@ROTATED_HEADS.register_module()
class SemiPoint2RBoxV2Head(RotatedAnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    Compared with FCOS head, Rotated FCOS head add a angle branch to
    support rotated object detection.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        angle_version (str): Angle representations. Defaults to 'le90'.
        use_hbbox_loss (bool): If true, use horizontal bbox loss and
            loss_angle should not be None. Default to False.
        scale_angle (bool): If true, add scale to angle pred branch.
            Default to True.
        angle_coder (:obj:`ConfigDict` or dict): Config of angle coder.
        h_bbox_coder (dict): Config of horzional bbox coder,
            only used when use_hbbox_loss is True.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder. Defaults
            to 'DistanceAnglePointCoder'.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_angle (:obj:`ConfigDict` or dict, Optional): Config of angle loss.

    Example:
        >>> self = RotatedFCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, angle_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 strides = [8],
                 regress_ranges = [(-1, 1e8)],
                 center_sampling = True,
                 center_sample_radius = 0.75,
                 angle_version = 'le90',
                 edge_loss_start_iter = 60000,
                 joint_angle_start_iter = 10000,
                 voronoi_type = 'gaussian-orientation',
                 voronoi_thres = dict(
                     default=[0.994, 0.005],
                     override=(([2, 11], [0.999, 0.6]),
                               ([7, 8, 10, 14], [0.95, 0.005]))),
                 square_cls = [1, 9, 11],
                 edge_loss_cls = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
                 post_process = {11: 1.2},
                 bbox_coder = dict(type='DistanceAnglePointCoder'),
                 angle_coder = dict(
                    type='PSCCoder',
                    angle_version='le90',
                    dual_freq=False,
                    num_step=3,
                    thr_mod=0),
                 loss_cls = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox = dict(
                     type='mmdet.L1Loss', loss_weight=0.01),
                 loss_overlap = dict(
                     type='GaussianOverlapLoss', loss_weight=100.0),
                 loss_voronoi = dict(
                     type='GaussianVoronoiLoss', loss_weight=50.0),
                 loss_bbox_edg = dict(
                     type='EdgeLoss', loss_weight=10.0),
                 loss_bbox_syn = dict(
                     type='RotatedIoULoss', loss_weight=1.0),
                 loss_ss=dict(
                    type='mmdet.SmoothL1Loss', loss_weight=0.2, beta=0.1),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(
                             type='Normal',
                             name='conv_cls',
                             std=0.01,
                             bias_prob=0.01), 
                         dict(
                             type='Normal',
                             name='conv_gate',
                             std=0.01,
                             bias_prob=0.01)]),
                 **kwargs):
        self.angle_coder = build_bbox_coder(angle_coder)
        super().__init__(
            num_classes,
            in_channels,
            strides=strides,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.angle_version = angle_version
        self.edge_loss_start_iter = edge_loss_start_iter
        self.joint_angle_start_iter = joint_angle_start_iter
        self.voronoi_thres = voronoi_thres
        self.voronoi_type = voronoi_type
        self.square_cls = square_cls
        self.edge_loss_cls = edge_loss_cls
        self.post_process = post_process
        self.loss_ss = build_loss(loss_ss)
        self.loss_bbox_syn = build_loss(loss_bbox_syn)
        self.loss_overlap = build_loss(loss_overlap)
        self.loss_voronoi = build_loss(loss_voronoi)
        self.loss_bbox_edg = build_loss(loss_bbox_edg)
            
    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        # self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        
    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is \
            num_points * 4.
            - centernesses (list[Tensor]): centerness for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
        """
        cls_feat = x[0]
        reg_feat = x[0]

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        angle_pred = self.conv_angle(reg_feat)

        # Gaussian sig_x, sig_y, p
        sig_x = bbox_pred[:, 0].exp()
        sig_y = bbox_pred[:, 1].exp()
        dx = bbox_pred[:, 2].sigmoid() * 2 - 1  # (-1, 1)
        dy = bbox_pred[:, 3].sigmoid() * 2 - 1  # (-1, 1)
        bbox_pred = torch.stack((sig_x, sig_y, dx, dy), 1) * 8

        return (cls_score,), (bbox_pred,), (angle_pred,)
    

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def loss(
        self,
        cls_scores,
        bbox_preds,
        angle_preds,
        batch_gt_instances,
        batch_img_metas,
        batch_gt_instances_ignore = None,
    ):
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, each \
                is a 4D-tensor, the channel number is num_points * encode_size.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        # bbox_targets here is in format t,b,l,r
        # angle_targets is not coded here
        labels, bbox_targets, bid_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, angle_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_bid_targets = torch.cat(bid_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    # & (flatten_labels != 2)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        # loss_cls = self.loss_cls(
        #         flatten_cls_scores[pos_inds], flatten_labels[pos_inds], avg_factor=num_pos)
        loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_cls_scores = flatten_cls_scores[pos_inds].sigmoid()
        pos_labels = flatten_labels[pos_inds]
        pos_cls_scores = torch.gather(pos_cls_scores, 1, pos_labels[:, None])[:, 0]

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_bid_targets = flatten_bid_targets[pos_inds]

        self.vis = [None] * len(batch_gt_instances)  # For visual debug
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_labels = flatten_labels[pos_inds]

            pos_decoded_angle_preds = self.angle_coder.decode(
                pos_angle_preds, keepdim=True)
            if self.iter_count < self.joint_angle_start_iter:
                pos_decoded_angle_preds = pos_decoded_angle_preds.detach()
            square_mask = torch.zeros_like(pos_labels, dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, pos_labels == c)
            pos_decoded_angle_preds[square_mask] = 0

            pos_cent_preds = pos_points + pos_bbox_preds[:, 2:]
            pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)  # Key. targets[:, -1] must be zero
            pos_rbox_preds = torch.cat((pos_rbox_targets[:, :2], pos_bbox_preds[:, :2] * 2, pos_decoded_angle_preds), -1)

            cos_r = torch.cos(pos_decoded_angle_preds)
            sin_r = torch.sin(pos_decoded_angle_preds)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            pos_gaus_preds = R.matmul(torch.diag_embed(pos_bbox_preds[:, :2])).matmul(R.permute(0, 2, 1))

            # Synthetic objects
            pos_syn_mask = pos_bid_targets[:, 1] == 1
            if torch.any(pos_syn_mask):
                preds = pos_gaus_preds[pos_syn_mask].view(-1, 2, 2)
                targets = xy_wh_r_2_xy_sigma(pos_rbox_targets[pos_syn_mask])
                loss_bbox_syn = 5 * gwd_loss((None, preds.bmm(preds)), targets)
            else:
                loss_bbox_syn = pos_gaus_preds.new_tensor(0)

            loss_bbox = self.loss_bbox(
                pos_cent_preds[~pos_syn_mask], 
                pos_rbox_targets[~pos_syn_mask, :2], 
                avg_factor=num_pos)

            # Aggregate targets of the same instance based on their identical bid
            bid_with_view = pos_bid_targets[:, 3] + 0.5 * pos_bid_targets[:, 2]
            bid, idx = torch.unique(bid_with_view, return_inverse=True)
            
            # Generate a mask to eliminate bboxes without correspondence
            # (bcnt is supposed to be 3, for ori, rot, and flp)
            ins_bid_with_view = bid.new_zeros(*bid.shape).index_reduce_(
                0, idx, bid_with_view, 'amin', include_self=False)
            _, bidx, bcnt = torch.unique(
                ins_bid_with_view.long(),
                return_inverse=True,
                return_counts=True)
            bmsk = bcnt[bidx] == 2

            # Select instances by batch
            ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_bid_targets[:, 3], 'amin', include_self=False)
            
            ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_bid_targets[:, 0], 'amin', include_self=False)
            
            ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_labels, 'amin', include_self=False)
            
            ins_gaus_preds = pos_gaus_preds.new_zeros(
                *bid.shape, 4).index_reduce_(
                    0, idx, pos_gaus_preds.view(-1, 4), 'mean',
                    include_self=False).view(-1, 2, 2)
            
            ins_rbox_preds = pos_rbox_preds.new_zeros(
                *bid.shape, pos_rbox_preds.shape[-1]).index_reduce_(
                    0, idx, pos_rbox_preds, 'mean',
                    include_self=False)
            
            ins_rbox_targets = pos_rbox_targets.new_zeros(
                *bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(
                    0, idx, pos_rbox_targets, 'mean',
                    include_self=False)

            ori_mu_all = ins_rbox_targets[:, 0:2]
            loss_bbox_vor = ori_mu_all.new_tensor(0)
            loss_bbox_ovl = ori_mu_all.new_tensor(0)
            batched_rbox = []
            loss_bbox_edg = ori_mu_all.new_tensor(0)
            for batch_id in range(len(batch_gt_instances)):
                group_mask = (ins_batch == batch_id) & (ins_bids != 0)
                # Overlap and Voronoi Losses
                mu = ori_mu_all[group_mask]
                sigma = ins_gaus_preds[group_mask]
                label = ins_labels[group_mask]
                if len(mu) >= 2:
                    loss_bbox_ovl += self.loss_overlap((mu, sigma.bmm(sigma)))
                if len(mu) >= 1:
                    pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
                    neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
                    if 'override' in self.voronoi_thres.keys():
                        for item in self.voronoi_thres['override']:
                            for cls in item[0]:
                                pos_thres[cls] = item[1][0]
                                neg_thres[cls] = item[1][1]
                    loss_bbox_vor += self.loss_voronoi((mu, sigma.bmm(sigma)),
                                                       label, self.images[batch_id],
                                                       pos_thres, neg_thres,
                                                       voronoi=self.voronoi_type)
                    self.vis[batch_id] = self.loss_voronoi.vis
            
            #  Batched RBox for Edge Loss
            if self.iter_count >= self.edge_loss_start_iter:
                for batch_id in range(len(batch_gt_instances)):
                    group_mask = (ins_batch == batch_id) & (ins_bids != 0)
                    rbox = ins_rbox_preds[group_mask]
                    label = ins_labels[group_mask]
                    edge_loss_mask = torch.zeros_like(label, dtype=torch.bool)
                    for c in self.edge_loss_cls:
                        edge_loss_mask = torch.logical_or(edge_loss_mask, label == c)
                    batched_rbox.append(rbox[edge_loss_mask])
                loss_bbox_edg = self.loss_bbox_edg(batched_rbox, self.edges)
            
            loss_bbox_ovl = loss_bbox_ovl / len(batch_gt_instances)
            loss_bbox_vor = loss_bbox_vor / len(batch_gt_instances)
            loss_bbox_edg = loss_bbox_edg / len(batch_gt_instances)

            pair_gaus_preds = ins_gaus_preds[bmsk].view(-1, 2, 2, 2)
            pair_labels = ins_labels[bmsk].view(-1, 2)[:, 0]
            square_mask = torch.zeros_like(pair_labels, dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, pair_labels == c)
            
            pair_cls_scores = torch.empty(
                *bid.shape, device=bid.device).index_reduce_(
                    0, idx, pos_cls_scores, 'mean',
                    include_self=False)[bmsk].view(-1, 2)
            
            pair_angle_preds = torch.empty(
                *bid.shape, pos_angle_preds.shape[-1],
                device=bid.device).index_reduce_(
                    0, idx, pos_angle_preds, 'mean',
                    include_self=False)[bmsk].view(-1, 2,
                                                pos_angle_preds.shape[-1])
            pair_angle_preds = self.angle_coder.decode(
                    pair_angle_preds, keepdim=True)
                                   
            # Self-supervision
            ss_info = batch_img_metas[0]['ss']
            valid = pair_cls_scores[:, 1] > 0.1
            bbox_area = pair_gaus_preds[:, 0, 0, 0] * pair_gaus_preds[:, 0, 1, 1] * 4
            sca = ss_info[1] if ss_info[0] == 'sca' else 1
            valid = torch.logical_and(valid, bbox_area > 24 ** 2)
            valid = torch.logical_and(valid, bbox_area * sca > 24 ** 2)
            valid = torch.logical_and(valid, bbox_area < 512 ** 2)
            valid = torch.logical_and(valid, bbox_area * sca < 512 ** 2)
            
            if torch.any(valid):
                ori_preds = pair_gaus_preds[valid, 0]
                trs_preds = pair_gaus_preds[valid, 1]
                square_mask = square_mask[valid]
                ori_angle = pair_angle_preds[valid, 0]
                trs_angle = pair_angle_preds[valid, 1]

                if ss_info[0] == 'rot':
                    rot = ori_preds.new_tensor(ss_info[1])
                    cos_r = torch.cos(rot)
                    sin_r = torch.sin(rot)
                    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
                    ori_preds = R.matmul(ori_preds).matmul(R.permute(0, 2, 1))
                    loss_ss = gwd_loss((None, ori_preds.bmm(ori_preds)), (None, trs_preds.bmm(trs_preds)))
                    d_ang = trs_angle - ori_angle - ss_info[1]
                    d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
                    d_ang[square_mask] = 0
                    loss_ssa = self.loss_ss(d_ang, torch.zeros_like(d_ang))
                elif ss_info[0] == 'flp':
                    ori_preds = ori_preds * ori_preds.new_tensor((1, -1, -1, 1)).reshape(2, 2)
                    loss_ss = gwd_loss((None, ori_preds.bmm(ori_preds)), (None, trs_preds.bmm(trs_preds)))
                    d_ang = trs_angle + ori_angle
                    d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
                    d_ang[square_mask] = 0
                    loss_ssa = self.loss_ss(d_ang, torch.zeros_like(d_ang))
                else:
                    sca = ori_preds.new_tensor(ss_info[1])
                    ori_preds = ori_preds * sca
                    loss_ss = gwd_loss((None, ori_preds.bmm(ori_preds)), (None, trs_preds.bmm(trs_preds)))
                    d_ang = trs_angle - ori_angle
                    d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
                    d_ang[square_mask] = 0
                    loss_ssa = self.loss_ss(d_ang, torch.zeros_like(d_ang))
                loss_ss = self.loss_ss.loss_weight * loss_ss
            else:
                loss_ss = pos_bbox_preds.new_tensor(0)
                loss_ssa = 0 * pos_angle_preds.sum()
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_bbox_vor = pos_bbox_preds.sum()
            loss_bbox_ovl = pos_bbox_preds.sum()
            loss_bbox_edg = pos_bbox_preds.sum()
            loss_ss = pos_bbox_preds.sum()
            loss_ssa = pos_angle_preds.sum()
            loss_bbox_syn = pos_bbox_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_vor=loss_bbox_vor,
            loss_bbox_ovl=loss_bbox_ovl,
            loss_bbox_edg=loss_bbox_edg,
            loss_ss=loss_ss,
            loss_ssa=loss_ssa,
            loss_bbox_syn=loss_bbox_syn
            )

    def get_targets(self, points, batch_gt_instances):
        """Compute regression, classification and centerness targets for points
        in multiple images.
        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
        Returns:
            tuple: Targets of each level.
            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                level.
            - concat_lvl_angle_targets (list[Tensor]): Angle targets of \
                each level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, bid_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        bid_targets_list = [
            bid_targets.split(num_points, 0)
            for bid_targets in bid_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bid_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            bid_targets = torch.cat(
                [bid_targets[i] for bid_targets in bid_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_bid_targets.append(bid_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_bid_targets)

    def _get_targets_single(
            self, gt_instances, points,
            regress_ranges,
            num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances['labels'])
        gt_bboxes = gt_instances['bboxes']
        gt_labels = gt_instances['labels']
        gt_bids = gt_instances['bids']

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bids.new_zeros((num_points, 4))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
        
        offset = points - gt_ctr
        w, h = gt_wh[..., 0].clone(), gt_wh[..., 1].clone()

        center_r = torch.clamp((w * h).sqrt() / 64, 1, 5)[..., None]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        # inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            # inside_center_bbox_mask = (abs(offset) < stride * center_r).all(dim=-1)
            # inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
            #                                         inside_gt_bbox_mask)
            inside_gt_bbox_mask = (abs(offset) < stride * center_r).all(dim=-1)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]
        bid_targets = gt_bids[min_area_inds]
        bbox_targets = torch.cat((bbox_targets, angle_targets), -1)

        return labels, bbox_targets, bid_targets


    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   img_metas,
                   cfg=None,
                   rescale=None,
                   batch_gt_instances=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            angle_preds (list[Tensor]): Box angle for each scale level \
                with shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the 6-th
                column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if self.training:
                det_bboxes = self._get_bboxes_single_pseudo(cls_score_list,
                                                            bbox_pred_list,
                                                            angle_pred_list,
                                                            mlvl_points, img_shape,
                                                            scale_factor, cfg, rescale, batch_gt_instances[img_id])
            else:
                det_bboxes = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    angle_pred_list,
                                                    mlvl_points, img_shape,
                                                    scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list


    def _get_bboxes_single_pseudo(self,
                                cls_scores,
                                bbox_preds,
                                angle_preds,
                                mlvl_points,
                                img_shape,
                                scale_factor,
                                cfg,
                                rescale=False,
                                batch_gt_instances=None):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        gt_instances = batch_gt_instances
        if self.training:
            scale_factor = [1, 1]
        else:
            scale_factor = scale_factor
        gt_bboxes = gt_instances['bboxes']
        gt_labels = gt_instances['labels']
        gt_pos = (gt_bboxes[:, 0:2] / self.strides[0] * scale_factor[1]).long()

        cls_score, bbox_pred, angle_pred = cls_scores[0], bbox_preds[0], angle_preds[0]
        H, W = cls_score.shape[1:3]

        gt_valid_mask = (0 <= gt_pos[:, 0]) & (gt_pos[:, 0] < W) & (0 <= gt_pos[:, 1]) & (gt_pos[:, 1] < H)
        gt_idx = gt_pos[:, 1] * W + gt_pos[:, 0]
        gt_idx = gt_idx.clamp(0, cls_score[0].numel() - 1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)[gt_idx]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)[gt_idx]

        angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)[gt_idx]
        decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
        
        bboxes = torch.cat((gt_bboxes[:, 0:2], bbox_pred[:, :2] * 2, decoded_angle), -1)

        bboxes[~gt_valid_mask, 2:] = 0
        bboxes[:, 2:4] = bboxes[:, 2:4] / scale_factor[1]

        for id in self.post_process.keys():
            bboxes[gt_labels == id, 2:4] *= self.post_process[id]
        for id in self.square_cls:
            bboxes[gt_labels == id, -1] = 0

        bboxes = torch.cat((bboxes, torch.ones_like(bboxes[:, :1])), dim=1)

        return bboxes, gt_labels



    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           batch_gt_instances=None):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, angle_pred, points in zip(
                cls_scores, bbox_preds, angle_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
            angle_pred = self.angle_coder.decode(angle_pred, keepdim=True)
            # bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * 1).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                angle_pred = angle_pred[topk_inds, :]   # add
                scores = scores[topk_inds, :]
            bboxes = torch.cat((points + bbox_pred[:, 2:], bbox_pred[:, :2] * 2, angle_pred), -1)   # add
            # bboxes = self.bbox_coder.decode(
            #     points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img)
        for id in self.square_cls:
            det_bboxes[det_labels == id, 4] = 0
        return det_bboxes, det_labels

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds, angle_preds):
        """This function will be used in S2ANet, whose num_anchors=1."""
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        # device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            angle_pred = angle_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1)
            angle_pred = angle_pred.reshape(num_imgs, -1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)

            points = mlvl_points[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(points, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list