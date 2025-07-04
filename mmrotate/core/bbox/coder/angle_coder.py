# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder
from torch import Tensor

from ..builder import ROTATED_BBOX_CODERS



@ROTATED_BBOX_CODERS.register_module()
class CSLCoder(BaseBBoxCoder):
    """Circular Smooth Label Coder.

    `Circular Smooth Label (CSL)
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_ .

    Args:
        angle_version (str): Angle definition.
        omega (float, optional): Angle discretization granularity.
            Default: 1.
        window (str, optional): Window function. Default: gaussian.
        radius (int/float): window radius, int type for
            ['triangle', 'rect', 'pulse'], float type for
            ['gaussian']. Default: 6.
    """

    def __init__(self, angle_version, omega=1, window='gaussian', radius=6):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        assert window in ['gaussian', 'triangle', 'rect', 'pulse']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45}
        self.angle_offset = self.angle_offset_dict[angle_version]
        self.omega = omega
        self.window = window
        self.radius = radius
        self.coding_len = int(self.angle_range // omega)

    def encode(self, angle_targets):
        """Circular Smooth Label Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The csl encoding of angle offset for each
                scale level. Has shape (num_anchors * H * W, coding_len)
        """

        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.coding_len)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long % self.coding_len
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = torch.exp(-torch.pow(base_radius_range, 2) /
                                     (2 * self.radius**2))

        else:
            raise NotImplementedError

        if isinstance(smooth_value, torch.Tensor):
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)

        return smooth_label.scatter(1, radius_range, smooth_value)

    def decode(self, angle_preds):
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset
                for each scale level.
                Has shape (num_anchors * H * W, coding_len)

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)
        """
        angle_cls_inds = torch.argmax(angle_preds, dim=1)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)


@ROTATED_BBOX_CODERS.register_module()
class PSCCoder(BaseBBoxCoder):
    """Phase-Shifting Coder.

    `Phase-Shifting Coder (PSC)
    <https://arxiv.org/abs/2211.06368>`.

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        dual_freq (bool, optional): Use dual frequency. Default: True.
        num_step (int, optional): Number of phase steps. Default: 3.
        thr_mod (float): Threshold of modulation. Default: 0.47.
    """

    def __init__(self,
                 angle_version: str,
                 dual_freq: bool = True,
                 num_step: int = 3,
                 thr_mod: float = 0.47):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.dual_freq = dual_freq
        self.num_step = num_step
        self.thr_mod = thr_mod
        if self.dual_freq:
            self.encode_size = 2 * self.num_step
        else:
            self.encode_size = self.num_step

        self.coef_sin = torch.tensor(
            tuple(
                torch.sin(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))
        self.coef_cos = torch.tensor(
            tuple(
                torch.cos(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Phase-Shifting Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
        """
        phase_targets = angle_targets * 2
        phase_shift_targets = tuple(
            torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
            for x in range(self.num_step))

        # Dual-freq PSC for square-like problem
        if self.dual_freq:
            phase_targets = angle_targets * 4
            phase_shift_targets += tuple(
                torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
                for x in range(self.num_step))

        return torch.cat(phase_shift_targets, axis=-1)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Phase-Shifting Decoder.

        Args:
            angle_preds (Tensor): The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """
        self.coef_sin = self.coef_sin.to(angle_preds)
        self.coef_cos = self.coef_cos.to(angle_preds)

        phase_sin = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_sin,
            dim=-1,
            keepdim=keepdim)
        phase_cos = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_cos,
            dim=-1,
            keepdim=keepdim)
        phase_mod = phase_cos**2 + phase_sin**2
        phase = -torch.atan2(phase_sin, phase_cos)  # In range [-pi,pi)

        if self.dual_freq:
            phase_sin = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_sin,
                dim=-1,
                keepdim=keepdim)
            phase_cos = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_cos,
                dim=-1,
                keepdim=keepdim)
            phase_mod = phase_cos**2 + phase_sin**2
            phase2 = -torch.atan2(phase_sin, phase_cos) / 2

            # Phase unwarpping, dual freq mixing
            # Angle between phase and phase2 is obtuse angle
            idx = torch.cos(phase) * torch.cos(phase2) + torch.sin(
                phase) * torch.sin(phase2) < 0
            # Add pi to phase2 and keep it in range [-pi,pi)
            phase2[idx] = phase2[idx] % (2 * math.pi) - math.pi
            phase = phase2

        # Set the angle of isotropic objects to zero
        phase[phase_mod < self.thr_mod] *= 0
        angle_pred = phase / 2
        return angle_pred
    
    
@ROTATED_BBOX_CODERS.register_module()
class PseudoAngleCoder(BaseBBoxCoder):
    """Pseudo Angle Coder."""

    encode_size = 1

    def encode(self, angle_targets: Tensor) -> Tensor:
        return angle_targets

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        if keepdim:
            return angle_preds
        else:
            return angle_preds.squeeze(-1)

@ROTATED_BBOX_CODERS.register_module()
class UCResolver(BaseBBoxCoder):
    """Unit Cycle Resolver.

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        mdim (int, optional): Dimension of mapping.
        loss_angle_restrict (:obj:`ConfigDict` or dict, Optional): Config of angle restrict loss.
    """

    def __init__(self,
                 angle_version: str,
                 mdim: int = 3,
                 invalid_thr: float=0.0,
                 loss_angle_restrict = None):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.mdim = mdim
        self.invalid_thr = invalid_thr
        self.loss_weight = loss_angle_restrict.loss_weight
        assert mdim >= 2

        self.encode_size = mdim
        if loss_angle_restrict is None:
            self.loss_angle_restrict = None
        else:
            #self.loss_angle_restrict = MODELS.build(loss_angle_restrict)
            self.loss_angle_restrict = torch.nn.L1Loss()

        self.coef_sin = torch.tensor([
                torch.sin(torch.tensor(2 * k * torch.pi / self.mdim))
                for k in range(self.mdim)])
        self.coef_cos = torch.tensor([
                torch.cos(torch.tensor(2 * k * torch.pi / self.mdim))
                for k in range(self.mdim)])
            
    def encode(self, angle_targets: Tensor) -> Tensor:
        """Unit Cycle Resolver.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Has shape (..., num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (..., num_anchors * H * W, encode_size)
        """
        angle_targets = angle_targets * 2

        if self.mdim > 2:
            encoded_targets = torch.cat([
                torch.cos(angle_targets + 2 * torch.pi * x / self.mdim)
                for x in range(self.mdim)], dim=-1)
        else:
            encoded_targets = torch.cat([
                torch.cos(angle_targets), torch.sin(angle_targets)], dim=-1)
            
        return encoded_targets

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Unit Cycle Resolver.

        Args:
            angle_preds (Tensor): The encoding state.
                Has shape (..., num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """

        self.coef_sin = self.coef_sin.to(angle_preds)
        self.coef_cos = self.coef_cos.to(angle_preds)

        if self.mdim > 2:
            predict_cos =   torch.sum(angle_preds * self.coef_cos, dim=-1, keepdim=keepdim)
            predict_sin = - torch.sum(angle_preds * self.coef_sin, dim=-1, keepdim=keepdim)
        else:
            predict_cos = angle_preds[..., 0, None]
            predict_sin = angle_preds[..., 1, None]

        theta = torch.atan2(predict_sin, predict_cos)
        
        if self.invalid_thr > 0:
            theta[predict_sin**2 + predict_cos**2 < (self.mdim/2)**2 * self.invalid_thr] *= 0

        return theta / 2

    def get_restrict_loss(self, angle_preds: Tensor) -> Tensor:
        """Unit Cycle Resolver.

        Args:
            angle_preds (Tensor): The encoding state.
                Has shape (..., num_anchors * H * W, encode_size)

        Returns:
            Tensor: Angle restrict loss.
                Has shape (1)
        """
        assert self.mdim <= 3

        d_angle_restrict = torch.sum(torch.pow(angle_preds, 2), dim=-1)
        d_angle_target = torch.ones_like(d_angle_restrict) * torch.tensor(self.mdim / 2)
        loss_angle_restrict = self.loss_angle_restrict(d_angle_restrict, d_angle_target)

        if self.mdim == 3:
            d_angle_restrict = torch.sum(angle_preds, dim=-1)
            d_angle_target = torch.zeros_like(d_angle_restrict)
            loss_angle_restrict += self.loss_angle_restrict(d_angle_restrict, d_angle_target)

        return loss_angle_restrict * self.loss_weight
