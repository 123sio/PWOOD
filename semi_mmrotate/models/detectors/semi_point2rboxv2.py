# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample

from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.single_stage import RotatedSingleStageDetector
from torchvision import transforms

from semi_mmrotate.models.third_parties.ted.ted import TED



def get_single_pattern(image, bbox, label, square_cls):
    if bbox[2] < 16 or bbox[3] < 16 or bbox[2] > 512 or bbox[3] > 512:
        raise

    def obb2poly(obb):
        cx, cy, w, h, t = obb
        dw, dh = (w - 1) / 2, (h - 1) / 2
        cost = np.cos(t)
        sint = np.sin(t)
        mrot = np.float32([[cost, -sint], [sint, cost]])
        poly = np.float32([[-dw, -dh], [dw, -dh], [dw, dh], [-dw, dh]])
        return np.matmul(poly, mrot.T) + np.float32([cx, cy])

    def get_pattern_gaussian(w, h, device):
        w, h = int(w), int(h)
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij')
        y = (y - h / 2) / (h / 2)
        x = (x - w / 2) / (w / 2)
        ox, oy = torch.randn(2, device=device).clip(-3, 3) * 0.15
        sx, sy = torch.rand(2, device=device) * 0.5 + 1
        z = torch.exp(-((x - ox) * sx)**2 - ((y - oy) * sy)**2) * 0.5 + 0.5
        return z

    cx, cy, w, h, t = bbox
    w, h = int(w), int(h)
    poly = obb2poly([cx, cy, w, h, t])

    pts1 = poly[0:3]
    pts2 = np.float32([[-1, -1], [1, -1], [1, 1]])
    M = cv2.getAffineTransform(pts1, pts2)
    M = np.concatenate((M, ((0, 0, 1),)), 0)

    H, W = image.shape[1:3]
    T = np.array([[2 / W, 0, -1],
                  [0, 2 / H, -1],
                  [0, 0, 1]])
    theta = T @ np.linalg.inv(M)
    theta = image.new_tensor(theta[:2, :])[None]
    grid = F.affine_grid(theta, [1, 3, h, w], align_corners=True)
    chip = F.grid_sample(image[None], grid, align_corners=True)[0]

    alpha = get_pattern_gaussian(chip.shape[-1], chip.shape[-2], chip.device)[None]
    chip = torch.cat((chip, alpha))
        
    w, h, t = chip.new_tensor((bbox[2] * (0.7 + 0.5 * np.random.rand()), bbox[3] * (0.7 + 0.5 * np.random.rand()), np.pi * np.random.rand()))
    if label in square_cls:
        t *= 0
    cosa = torch.abs(torch.cos(t))
    sina = torch.abs(torch.sin(t))
    sx, sy = int(torch.ceil(cosa * w + sina * h)), int(torch.ceil(sina * w + cosa * h))
    theta = chip.new_tensor(
        [[1 / w * torch.cos(t), 1 / w * torch.sin(t), 0],
        [1 / h * torch.sin(-t), 1 / h * torch.cos(t), 0]])
    theta[:, :2] @= chip.new_tensor([[sx, 0], [0, sy]])
    grid = torch.nn.functional.affine_grid(
        theta[None], (1, 1, sy, sx), align_corners=True)
    chip = torch.nn.functional.grid_sample(
        chip[None], grid, align_corners=True, mode='nearest')[0]
    bbox = np.float32([sx / 2, sy / 2, w.item(), h.item(), t.item()])
    return (chip, bbox, label)


def get_copy_paste_cache(images, bboxes, labels, square_cls, num_copies):
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()
    patterns = []
    for b, l in zip(bboxes, labels):
        try:
            p = get_single_pattern(images, b, l, square_cls)
            patterns.append(p)
            if len(patterns) > num_copies:
                break
        except:
            pass
    return patterns


def plot_one_rotated_box(img,
                         obb,
                         color=[0.0, 0.0, 128],
                         label=None,
                         line_thickness=None):
    width, height, theta = obb[2], obb[3], obb[4] / np.pi * 180
    if theta < 0:
        width, height, theta = height, width, theta + 90
    rect = [(obb[0], obb[1]), (width, height), theta]
    poly = np.intp(np.round(
        cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    cv2.drawContours(
        image=img, contours=[poly], contourIdx=-1, color=color, thickness=2)
    c1 = (int(obb[0]), int(obb[1]))
    if label:
        tl = 2
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        textcolor = [0, 0, 0] if max(color) > 192 else [255, 255, 255]
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, textcolor, thickness=tf, lineType=cv2.LINE_AA)

@ROTATED_DETECTORS.register_module()
class SemiPoint2RBoxV2(RotatedSingleStageDetector):
    """Implementation of `H2RBox-v2 <https://arxiv.org/abs/2304.04403>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 rotate_range = (0.25, 0.75),
                 scale_range = (0.5, 0.9),
                 ss_prob = [0.6, 0.15, 0.25],
                 copy_paste_start_iter = 60000,
                 num_copies = 10,
                 debug = False,
                 train_cfg = None,
                 test_cfg = None,
                 pretrained=None,
                 init_cfg = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.ss_prob = ss_prob
        self.copy_paste_start_iter = copy_paste_start_iter
        self.num_copies = num_copies
        self.debug = debug
        self.copy_paste_cache = None

        self.ted_model = TED()
        for param in self.ted_model.parameters():
            param.requires_grad = False
        self.ted_model.load_state_dict(torch.load('semi_mmrotate/models/third_parties/ted/ted.pth'))
        self.ted_model.eval()

    # def set_epoch(self, epoch):
    #     self.epoch = epoch
    #     self.bbox_head.epoch = epoch

    def rotate_crop(
            self,
            batch_inputs,
            rot = 0.,
            size = (768, 768),
            batch_gt_instances = None,
            padding = 'reflection'):
        """

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            rot (float): Angle of view rotation. Defaults to 0.
            size (tuple[int]): Crop size from image center.
                Defaults to (768, 768).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  
                padding (str): Padding method of image black edge.
                Defaults to 'reflection'.

        Returns:
            Processed batch_inputs (Tensor) and batch_gt_instances
            (list[:obj:`InstanceData`])
        """
        device = batch_inputs.device
        n, c, h, w = batch_inputs.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2 
        crop_w = (w - size_w) // 2 
        if rot != 0:
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = batch_inputs.new_tensor([[cosa, -sina], [sina, cosa]],
                                         dtype=torch.float)
            x_range = torch.linspace(-1, 1, w, device=device) 
            y_range = torch.linspace(-1, 1, h, device=device)
            y, x = torch.meshgrid(y_range, x_range)
            grid = torch.stack([x, y], -1).expand([n, -1, -1, -1]) 
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2) 
            # rotate
            batch_inputs = grid_sample(
                batch_inputs, grid, 'bilinear', padding, align_corners=True)
            if batch_gt_instances is not None:
                for i, gt_instances in enumerate(batch_gt_instances):
                    gt_bboxes = gt_instances
                    xy, wh, a = gt_bboxes[..., :2], gt_bboxes[  
                        ..., 2:4], gt_bboxes[..., [4]]
                    ctr = tf.new_tensor([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.T) + ctr
                    a = a + rot
                    rot_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                    batch_gt_instances[i] = rot_gt_bboxes
        batch_inputs = batch_inputs[..., crop_h:crop_h + size_h,
                                    crop_w:crop_w + size_w]
        if batch_gt_instances is None:
            return batch_inputs
        else:  # rot == 0
            for i, gt_instances in enumerate(batch_gt_instances):
                gt_bboxes = gt_instances
                xy, wh, a = gt_bboxes[..., :2], gt_bboxes[...,
                                                          2:4], gt_bboxes[...,
                                                                          [4]]
                xy = xy - xy.new_tensor([[crop_w, crop_h]])
                crop_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                batch_gt_instances[i] = crop_gt_bboxes

            return batch_inputs, batch_gt_instances
        
    def forward_train(self, 
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      get_data=False,
                      gt_bboxes_ignore=None):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            img_metas (list[dict]): The batch image metadata.
            gt_bboxes (list[Tensor]): Ground truth bounding boxes for each image.
            gt_labels (list[Tensor]): Class labels for each ground truth box.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes that are
                ignored during training. Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        H, W = img.shape[2:4]
        batch_gt_instances = []
        
        self.bbox_head.iter_count = self.iter_count
        
        # Convert gt_bboxes and gt_labels into structured instance format
        for i in range(len(gt_bboxes)):
            instance = {'bboxes': gt_bboxes[i], 'labels': gt_labels[i]}
            batch_gt_instances.append(instance)

        offset = 1
        for i, gt_instances in enumerate(batch_gt_instances):
            blen = len(gt_instances['bboxes'])
            bids = gt_instances['labels'].new_zeros(blen, 4)
            bids[:, 0] = i
            bids[:, 3] = torch.arange(0, blen, 1) + offset
            gt_instances['bids'] = bids
            offset += blen

        sel_p = torch.rand(1)
        if sel_p < self.ss_prob[0]:
            # Generate rotated images and gts
            rot = math.pi * (
                torch.rand(1).item() *
                (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0])
            for meta in img_metas:
                meta['ss'] = ('rot', rot)
            img_aug = transforms.functional.rotate(img, -rot / math.pi * 180)
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = img.new_tensor([[cosa, -sina], [sina, cosa]], dtype=torch.float)
            ctr = tf.new_tensor([[img.shape[-1] / 2, img.shape[-2] / 2]])
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            # img_aug, batch_gt_aug = self.rotate_crop(img, rot, [H, W], batch_gt_instances, 'reflection')
            for gt_instances in batch_gt_aug:
                gt_instances['bboxes'][:, :2] = (gt_instances['bboxes'][..., :2] - ctr).matmul(tf.T) + ctr
                gt_instances['bboxes'][:, 4] = gt_instances['bboxes'][:, 4] + rot
                gt_instances['bids'][:, 0] += len(batch_gt_instances)
                gt_instances['bids'][:, 2] = 1
        elif sel_p < self.ss_prob[0] + self.ss_prob[1]:
            # Generate flipped images and gts
            for meta in img_metas:
                meta['ss'] = ('flp', 0)
            img_aug = transforms.functional.vflip(img)
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            for gt_instances in batch_gt_aug:
                gt_instances['bboxes'][:, 1] = img.shape[-2] - gt_instances['bboxes'][:, 1]
                gt_instances['bboxes'][:, 4] = -gt_instances['bboxes'][:, 4]
                gt_instances['bids'][:, 0] += len(batch_gt_instances)
                gt_instances['bids'][:, 2] = 1
        else:
            # Generate scaled images and gts
            sca = (torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0])
            for meta in img_metas:
                meta['ss'] = ('sca', sca)
            img_aug = transforms.functional.resized_crop(img, 0, 0, int(H / sca), int(W / sca), [H, W])
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            for gt_instances in batch_gt_aug:
                gt_instances['bboxes'][:, :4] *= sca
                gt_instances['bids'][:, 0] += len(batch_gt_instances)
                gt_instances['bids'][:, 2] = 1
                
        img_all = torch.cat((img, img_aug))
        self.bbox_head.images = img_all
        # Edge
        if self.iter_count >= self.bbox_head.edge_loss_start_iter:
            with torch.no_grad():
                mean = img_all.new_tensor([123.675, 116.28, 103.53])[..., None, None]
                std = img_all.new_tensor([58.395, 57.12, 57.375])[..., None, None]
                batch_edges = self.ted_model(img_all * std + mean)
                self.bbox_head.edges = batch_edges[3].clamp(0)
                # cv2.imwrite('E.png', batch_edges[0].cpu().numpy() * 255)

        # if self.copy_paste_cache and len(batch_gt_aug) == len(self.copy_paste_cache):
        #     for i in range(len(batch_gt_aug)):
        #         gt_instances, patterns = batch_gt_aug[i], self.copy_paste_cache[i]
        #         bboxes_paste = []
        #         labels_paste = []
        #         for p, b, l in patterns:
        #             h, w = p.shape[1:3]
        #             ox = np.random.randint(0, img_aug.shape[-1] - w)
        #             oy = np.random.randint(0, img_aug.shape[-2] - h)
        #             img_aug[i, :, oy:oy + h, ox:ox + w] = \
        #                 img_aug[i, :, oy:oy + h, ox:ox + w] * (1 - p[(3,)]) + p[:3] * p[(3,)]
        #             bboxes_paste.append(b + np.float32((ox, oy, 0, 0, 0)))
        #             labels_paste.append(l)
        #         gt_instances['bboxes'] = torch.cat((gt_instances['bboxes'], gt_instances['bboxes'].new_tensor(np.float32(bboxes_paste))))
        #         gt_instances['labels'] = torch.cat((gt_instances['labels'], gt_instances['labels'].new_tensor(np.int32(labels_paste))))
        #         gt_instances['bids'] = torch.cat((gt_instances['bids'], gt_instances['bids'].new_tensor((i, 1, 0, 0)).expand(len(labels_paste), -1)))
        #         batch_gt_aug[i] = gt_instances

        batch_inputs_all = torch.cat((img, img_aug))
        batch_data_samples_all = []
        for gt_instances, img_metas in zip(batch_gt_instances + batch_gt_aug, img_metas + img_metas):
            data_sample = {'metainfo': img_metas, 'gt_instances': gt_instances}
            batch_data_samples_all.append(data_sample)
        
        feat = self.extract_feat(batch_inputs_all)
        cls_scores, bbox_preds, angle_preds = self.bbox_head.forward(feat)

        if get_data:
            bs, encode_size, H, W = angle_preds[0].shape
            merged_angle_preds = torch.cat([pred.flatten(2).permute(0, 2, 1) for pred in angle_preds], dim=0).reshape(-1, encode_size)
            decoded_angle_preds = self.bbox_head.angle_coder.decode(merged_angle_preds, keepdim=True)
            decoded_angle_preds = decoded_angle_preds.view(bs, 1, H, W)
            angle_preds = list(torch.chunk(decoded_angle_preds, len(angle_preds), dim=0))

            return (cls_scores, bbox_preds, angle_preds)
        
        # batch_gt_instances = [data_sample['gt_instances'] for data_sample in batch_data_samples_all]
        batch_gt_instances = [copy.deepcopy(data_sample['gt_instances']) for data_sample in batch_data_samples_all]
        batch_img_metas = [data_sample['metainfo'] for data_sample in batch_data_samples_all]
        
        results_list = self.bbox_head.get_bboxes(cls_scores, bbox_preds, angle_preds, batch_img_metas, batch_gt_instances=batch_gt_instances)
        converted_results_list = []
        for det_bboxes, det_labels in results_list:
            bboxes = det_bboxes[:, :5]
            scores = det_bboxes[:, 5]
            result_dict = {
                'bboxes': bboxes,
                'scores': scores,  
                'labels': det_labels 
            }
            converted_results_list.append(result_dict)
        
        # Update point annotations with predicted rbox
        for data_sample, results in zip(batch_gt_instances, converted_results_list):
            mask = data_sample['bids'][:, 1] == 0
            data_sample['bboxes'][mask] = results['bboxes'][mask]
            data_sample['labels'][mask] = results['labels'][mask]

        losses = self.bbox_head.loss(cls_scores, bbox_preds, angle_preds, batch_gt_instances, batch_img_metas)

        # if self.iter_count >= self.copy_paste_start_iter:
        #     self.copy_paste_cache = []
        #     for img_tensor, result in zip(img, converted_results_list):
        #         self.copy_paste_cache.append(get_copy_paste_cache(img_tensor, 
        #                                                           result['bboxes'], 
        #                                                           result['labels'], 
        #                                                           self.bbox_head.square_cls,
        #                                                           self.num_copies))

        # self.debug = True
        if self.debug:
            for i in range(len(batch_inputs_all)):
                img = batch_inputs_all[i]
                if self.bbox_head.vis[i]:
                    vor, wat = self.bbox_head.vis[i]
                    img[0, wat != wat.max()] += 2
                    img[:, vor != vor.max()] -= 1
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.ascontiguousarray(img[..., (2, 1, 0)] * 58 + 127)
                bb = batch_data_samples_all[i]['gt_instances']['bboxes']
                ll = batch_data_samples_all[i]['gt_instances']['labels']
                for b, l in zip(bb.cpu().numpy(), ll.cpu().numpy()):
                    b[2:4] = b[2:4].clip(3)
                    plot_one_rotated_box(img, b, (255, 0, 0))
                if i < len(converted_results_list):
                    bb = converted_results_list[i]['bboxes']
                    if hasattr(converted_results_list[i], 'informs'):
                        for b, l in zip(bb.cpu().numpy(), converted_results_list[i].infoms.cpu().numpy()):
                            plot_one_rotated_box(img, b, (0, 255, 0), label=f'{l}')
                    else:
                        for b in bb.cpu().numpy():
                            plot_one_rotated_box(img, b, (0, 255, 0))
                # img_id = batch_data_samples_all[i]['metainfo']['filename']
                img_id = i
                img = np.clip(img, 0, 255).astype(np.uint8)
                cv2.imwrite(f'./show/{img_id}.png', img)
        
        return losses



