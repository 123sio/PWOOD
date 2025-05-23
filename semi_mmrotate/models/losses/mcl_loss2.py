import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import multi_apply
from mmrotate.models import ROTATED_LOSSES
import numpy as np
import sklearn.mixture as skm


@ROTATED_LOSSES.register_module()
class RotatedMCLLoss2(nn.Module):
    def __init__(self, cls_channels=16, thres = 0.015):
        super(RotatedMCLLoss2, self).__init__()
        self.cls_channels = cls_channels
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        self.thres =  thres

    def gmm_policy(self, scores, given_gt_thr=0.02, policy='high'):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.

        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (
                    scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr

    def pre_processing(self, logits, alone_angle=True):
        if alone_angle:
            cls_scores, bbox_preds, angle_preds = logits
            assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        else:
            cls_scores, bbox_preds = logits
            assert len(cls_scores) == len(bbox_preds)

        batch_size = cls_scores[0].shape[0]   
        img_logits_list = []
        for img_id in range(batch_size):
            img_logits = []
            img_logits.append(torch.cat([
                x[img_id].permute(1, 2, 0).reshape(-1, self.cls_channels) for x in cls_scores
            ], dim=0))

            if alone_angle:
                img_logits.append(torch.cat([
                    torch.cat([x[img_id], y[img_id]], dim=0).permute(1, 2, 0).reshape(-1, 5) for x, y in
                    zip(bbox_preds, angle_preds)
                ], dim=0))
            else:
                img_logits.append(torch.cat([
                    x[img_id].permute(1, 2, 0).reshape(-1, 5) for x in bbox_preds
                ], dim=0))


            img_logits_list.append(img_logits)

        return img_logits_list

    def loss_single(self, t_logits_list, s_logits_list):
        t_cls_scores, t_bbox_preds = tuple(t_logits_list)
        s_cls_scores, s_bbox_preds = tuple(s_logits_list)
        with torch.no_grad():
            teacher_probs = t_cls_scores.sigmoid()
            teacher_bboxes = t_bbox_preds

            # 计算置信度
            joint_confidences = teacher_probs
            max_vals = torch.max(joint_confidences, 1)[0]

            # 策略1：GMM动态阈值
            thres = self.gmm_policy(max_vals)
            valid_inds = torch.nonzero(max_vals > thres).squeeze(-1)
            selected_inds = valid_inds

            # 策略2：筛选出置信度大于 0.02 的点
            #print(self.thres)
            #valid_inds = torch.nonzero(max_vals > self.thres).squeeze(-1)
            #selected_inds = valid_inds

            # 策略3：如果 valid_inds 过少，直接使用它；否则，选择前 2000 个
            # if valid_inds.size(0) > 2000:
            #     topk_inds = torch.topk(max_vals[valid_inds], 2000)[1]  # 在 valid_inds 选出前 2000 个
            #     selected_inds = valid_inds[topk_inds]
            # else:
            #     selected_inds = valid_inds  # 置信度大于 0.02 的所有点

            weight_mask = torch.zeros_like(max_vals)
            weight_mask[selected_inds] = max_vals[selected_inds]
            b_mask = weight_mask > 0.

        if b_mask.sum() == 0:
            loss_cls = QFLv2(
                s_cls_scores.sigmoid(),
                teacher_probs,
                weight=max_vals,
                reduction="sum",
            ) / max_vals.sum()
            loss_bbox = torch.zeros(1).squeeze().to(max_vals.device)
        else:
            loss_cls = QFLv2(
                s_cls_scores.sigmoid(),
                teacher_probs,
                weight=weight_mask,
                reduction="sum",
            ) / weight_mask.sum()

            loss_bbox = (self.bbox_loss(
                s_bbox_preds[b_mask],
                teacher_bboxes[b_mask],
            ) * weight_mask[:, None][b_mask]).mean() * 10

        return loss_cls, loss_bbox
    
    def forward(self, teacher_logits, student_logits, img_metas=None, alone_angle=True):

        img_t_logits_list = self.pre_processing(teacher_logits, alone_angle)
        img_s_logits_list = self.pre_processing(student_logits, alone_angle)

        featmap_sizes = [featmap.size()[-2:] for featmap in teacher_logits[0]]

        # get labels and bbox_targets of each image
        losses_list = multi_apply(
                            self.loss_single,
                            img_t_logits_list,
                            img_s_logits_list)

        unsup_losses = dict(
            loss_cls=sum(losses_list[0]) / len(losses_list[0]),
            loss_bbox=sum(losses_list[1]) / len(losses_list[1])
        )

        return unsup_losses


def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0 
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape) 
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta) 
    pos = weight > 0

    # positive goes to teacher quality 
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos] 
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss