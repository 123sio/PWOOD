import torch
import numpy as np
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector
import math
from torchvision import transforms
import copy


@ROTATED_DETECTORS.register_module()
class H2RV2MCLTeacher(RotatedSemiDetector):
    def __init__(self, model: dict, semi_loss, train_cfg=None, test_cfg=None,
                 prob_rot=0.95,
                 view_range=(0.25, 0.75)
                 ):
        super(H2RV2MCLTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            semi_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.prob_rot = prob_rot
        self.view_range = view_range
        if train_cfg is not None:
            self.freeze("teacher")
            # ugly manner to get start iteration, to fit resume mode
            self.iter_count = train_cfg.get("iter_count", 0)
            # Prepare semi-training config
            # step to start training student (not include EMA update)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)
            # prepare super & un-super weight
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.logit_specific_weights = train_cfg.get("logit_specific_weights")

    def extract_feat_student(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.student.backbone(img)
        x = self.student.neck(x)
        return x

    def extract_feat_teacher(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.teacher.backbone(img)
        x = self.teacher.neck(x)
        return x

    def forward_train(self, imgs, img_metas, **kwargs):
        super(H2RV2MCLTeacher, self).forward_train(imgs, img_metas, **kwargs)
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')

        # preprocess
        format_data = dict()

        sup_img = []
        sup_gt_labels = []
        sup_gt_bboxes = []
        sup_img_metas = []
        for i in range(len(imgs)):
            tag = img_metas[i]['tag']
            if tag in ['sup_strong', 'sup_weak']:
                sup_img.append(imgs[i])
                sup_gt_labels.append(gt_labels[i])
                sup_gt_bboxes.append(gt_bboxes[i])
                sup_img_metas.append(img_metas[i])
        sup_img = torch.stack(sup_img,dim = 0)
        #print(sup_img.size())
        #print(len(sup_gt_bboxes))
        #print(sup_gt_bboxes[0].size())
        #print(sup_gt_labels.size())

        #print('#####',sup_img.size())

        for idx in range(len(img_metas)):
            tag = img_metas[idx]['tag']
            if tag in ['sup_strong', 'sup_weak']:
                tag = 'sup'
            if tag == 'sup':
                if tag not in format_data.keys():
                    format_data[tag] = dict()
                    img = sup_img
                    img_meta = sup_img_metas
                    gt_bboxes =  sup_gt_bboxes
                    gt_labels = sup_gt_labels
                    # Concat original/rotated/flipped images and gts in the batch dim
                    # h2r--------------------
                    offset = 1
                    for i, bboxes in enumerate(gt_bboxes):
                        bids = torch.arange(
                            0, len(bboxes), 1, device=bboxes.device) + offset
                        gt_bboxes[i] = torch.cat((bboxes, bids[:, None]), dim=-1)
                        offset += len(bboxes)

                    # Concat original/rotated/flipped images and gts in the batch dim
                    if torch.rand(1) < self.prob_rot:
                        rot = math.pi * (
                                torch.rand(1).item() *
                                (self.view_range[1] - self.view_range[0]) + self.view_range[0])
                        img_ss = transforms.functional.rotate(img, -rot / math.pi * 180)

                        cosa, sina = math.cos(rot), math.sin(rot)
                        tf = img.new_tensor([[cosa, -sina], [sina, cosa]], dtype=torch.float)
                        ctr = tf.new_tensor([[img.shape[-1] / 2, img.shape[-2] / 2]])
                        gt_bboxes_ss = copy.deepcopy(gt_bboxes)
                        for bboxes in gt_bboxes_ss:
                            bboxes[:, :2] = (bboxes[..., :2] - ctr).matmul(tf.T) + ctr
                            bboxes[:, 4] = bboxes[:, 4] + rot
                            bboxes[:, 5] = bboxes[:, 5] + 0.5

                        img = torch.cat((img, img_ss), 0)
                        gt_bboxes = gt_bboxes + gt_bboxes_ss
                        gt_labels = gt_labels + gt_labels
                        for m in img_metas:
                            m['ss'] = ('rot', rot)
                    else:
                        img_ss = transforms.functional.vflip(img)
                        gt_bboxes_ss = copy.deepcopy(gt_bboxes)
                        for bboxes in gt_bboxes_ss:
                            bboxes[:, 1] = img.shape[-2] - bboxes[:, 1]
                            bboxes[:, 4] = -bboxes[:, 4]
                            bboxes[:, 5] = bboxes[:, 5] + 0.5

                        img = torch.cat((img, img_ss), 0)
                        gt_bboxes = gt_bboxes + gt_bboxes_ss
                        gt_labels = gt_labels + gt_labels
                        for m in img_metas:
                            m['ss'] = ('flp', 0)


                    format_data[tag]['img'] = [img]
                    format_data[tag]['img_metas'] = img_meta
                    format_data[tag]['gt_bboxes'] = gt_bboxes
                    format_data[tag]['gt_labels'] = gt_labels


            else:
                img = imgs[idx]
                gt_bbox = gt_bboxes[idx]
                gt_label = gt_labels[idx]
                img_meta = img_metas[idx]
                if tag not in format_data.keys():
                    format_data[tag] = dict()
                    format_data[tag]['img'] = [img]
                    format_data[tag]['img_metas'] = [img_meta]
                    format_data[tag]['gt_bboxes'] = [gt_bbox]
                    format_data[tag]['gt_labels'] = [gt_label]
                else:
                    if tag != 'sup':
                        format_data[tag]['img'].append(img)
                    format_data[tag]['img_metas'].append(img_meta)
                    format_data[tag]['gt_bboxes'].append(gt_bbox)
                    format_data[tag]['gt_labels'].append(gt_label)

        for key in format_data.keys():
             format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)
        format_data['sup']['img'] = torch.squeeze(format_data['sup']['img'],dim = 0)
        #for key in format_data.keys():
        #    print(f"{key}: {format_data[key]['img'].shape}")

        losses = dict()
        # supervised forward
        x = self.extract_feat_student(format_data['sup']['img'])
        
        #print('########',len(format_data['sup']['gt_labels']))
        #print(format_data['sup']['gt_labels'][0].size())
        sup_losses = self.student.bbox_head.forward_train(x, format_data['sup']['img_metas'],
                                                format_data['sup']['gt_bboxes'],
                                                format_data['sup']['gt_labels'])
        for key, val in sup_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[key] = val

        if self.iter_count > self.burn_in_steps:
            # Train Logic
            # unsupervised forward
            unsup_weight = self.unsup_weight
            if self.weight_suppress == 'exp':
                target = self.burn_in_steps + 2000
                if self.iter_count <= target:
                    scale = np.exp((self.iter_count - target) / 1000)
                    unsup_weight *= scale
            elif self.weight_suppress == 'step':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= 0.25
            elif self.weight_suppress == 'linear':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps

            with torch.no_grad():
                # get teacher data
                teacher_feat = self.extract_feat_teacher(format_data['unsup_weak']['img'])
                teacher_logits, featmap_sizes = self.teacher.bbox_head.forward_train(teacher_feat,
                                                                      format_data['unsup_weak']['img_metas'],
                                                                      get_data=True
                                                                      )

            # get student data
            student_feat = self.extract_feat_student(format_data['unsup_strong']['img'])
            student_logits, _ = self.student.bbox_head.forward_train(student_feat,
                                                                  format_data['unsup_strong']['img_metas'],
                                                                  get_data=True)
            unsup_losses = self.semi_loss(teacher_logits,
                                          student_logits,
                                          featmap_sizes,
                                          img_metas=format_data['unsup_weak'])

            for key, val in self.logit_specific_weights.items():
                if key in unsup_losses.keys():
                    unsup_losses[key] *= val
            for key, val in unsup_losses.items():
                if key[:4] == 'loss':
                    losses[f"{key}_unsup"] = unsup_weight * val
                else:
                    losses[key] = val
        self.iter_count += 1

        return losses