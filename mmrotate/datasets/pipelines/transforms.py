# Copyright (c) OpenMMLab. All rights reserved.
import copy

import cv2
import mmcv
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmdet.datasets.pipelines.transforms import (Mosaic, RandomCrop,
                                                 RandomFlip, Resize)
#from mmcv.transforms import to_tensor
from numpy import random

from mmrotate.core import norm_angle, obb2poly_np, poly2obb_np
from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class RResize(Resize):
    """Resize images & rotated bbox Inherit Resize pipeline class to handle
    rotated bboxes.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio).
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None):
        super(RResize, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=True)

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            orig_shape = bboxes.shape
            bboxes = bboxes.reshape((-1, 5))
            w_scale, h_scale, _, _ = results['scale_factor']
            bboxes[:, 0] *= w_scale
            bboxes[:, 1] *= h_scale
            bboxes[:, 2:4] *= np.sqrt(w_scale * h_scale)
            results[key] = bboxes.reshape(orig_shape)

@ROTATED_PIPELINES.register_module()
class AddNoise():
    """Convert RBoxes to Single Center Points."""

    def __call__(self, results):
          return self.transform(results)

    def __init__(self, p = 0.1):
         self.p = p

    def transform(self, results):
        """The transform function."""

        # h = torch.min(results['gt_bboxes'][:, 3:5], 1)[0]
        # results['gt_bboxes'][:, 0] += (torch.rand(len(results['gt_bboxes'])) * 2 - 1) * self.p * h
        # results['gt_bboxes'][:, 1] += (torch.rand(len(results['gt_bboxes'])) * 2 - 1) * self.p * h

        # 确保 gt_bboxes 是 torch.Tensor
        if isinstance(results['gt_bboxes'], np.ndarray):
            gt_bboxes = torch.tensor(results['gt_bboxes'], dtype=torch.float32)
        else:
            gt_bboxes = results['gt_bboxes']

        # 计算 h (取 3:5 维度的最小值)
        h, _ = torch.min(gt_bboxes[:, 2:4], dim=1)

        # 计算随机偏移量
        noise_x = (torch.rand(len(gt_bboxes)) * 2 - 1) * self.p * h
        noise_y = (torch.rand(len(gt_bboxes)) * 2 - 1) * self.p * h

        # 添加噪声
        gt_bboxes[:, 1] += noise_y

        # 如果原始数据是 NumPy，则转换回 NumPy
        results['gt_bboxes'] = gt_bboxes.numpy() if isinstance(results['gt_bboxes'], np.ndarray) else gt_bboxes
        
        return results
@ROTATED_PIPELINES.register_module()
class ConvertWeakSupervision():
    def __call__(self, results):
          return self.transform(results)

    def __init__(self, 
                point_proportion: float = 0.3,
                hbox_proportion: float = 0.3, 
                point_dummy: float = 1,
                hbox_dummy: float = 0,
                modify_labels: bool = False) -> None:
        self.point_proportion = point_proportion
        self.hbox_proportion = hbox_proportion
        self.point_dummy = point_dummy
        self.hbox_dummy = hbox_dummy
        self.modify_labels = modify_labels

    def transform(self, results: dict) -> dict:
        """The transform function."""

        max_idx_p = int(round(results['gt_bboxes'].tensor.shape[0] * self.point_proportion))
        results['gt_bboxes'].tensor[:max_idx_p, 2] = self.point_dummy
        results['gt_bboxes'].tensor[:max_idx_p, 3] = self.point_dummy
        results['gt_bboxes'].tensor[:max_idx_p, 4] = 0

        max_idx_h = max_idx_p + int(round(results['gt_bboxes'].tensor.shape[0] * self.hbox_proportion))
        results['gt_bboxes'][max_idx_p:max_idx_h] = \
            results['gt_bboxes'][max_idx_p:max_idx_h].convert_to('hbox').convert_to('rbox')
        results['gt_bboxes'].tensor[max_idx_p:max_idx_h, 4] = self.hbox_dummy

        if self.modify_labels:
            ws_types = torch.zeros(results['gt_bboxes'].tensor.shape[0], dtype=torch.long)
            ws_types[:max_idx_p] = 2
            ws_types[max_idx_p:max_idx_h] = 1
            #labels = to_tensor(results['gt_bboxes_labels'])
            results['gt_bboxes_labels'] = torch.stack((labels, ws_types), -1)

        return results

@ROTATED_PIPELINES.register_module()
class RRandomFlip(RandomFlip):
    """

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'.
        version (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self, flip_ratio=None, direction='horizontal', version='oc'):
        self.version = version
        super(RRandomFlip, self).__init__(flip_ratio, direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally or vertically.

        Args:
            bboxes(ndarray): shape (..., 5*k)
            img_shape(tuple): (height, width)

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 5 == 0
        orig_shape = bboxes.shape
        bboxes = bboxes.reshape((-1, 5))
        flipped = bboxes.copy()
        if direction == 'horizontal':
            flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        elif direction == 'vertical':
            flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
        elif direction == 'diagonal':
            flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
            flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
            return flipped.reshape(orig_shape)
        else:
            raise ValueError(f'Invalid flipping direction "{direction}"')
        if self.version == 'oc':
            rotated_flag = (bboxes[:, 4] != np.pi / 2)
            flipped[rotated_flag, 4] = np.pi / 2 - bboxes[rotated_flag, 4]
            flipped[rotated_flag, 2] = bboxes[rotated_flag, 3]
            flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
        else:
            flipped[:, 4] = norm_angle(np.pi - bboxes[:, 4], self.version)
        return flipped.reshape(orig_shape)


@ROTATED_PIPELINES.register_module()
class PolyRandomRotate(object):
    """Rotate img & bbox.
    Reference: https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA

    Args:
        rotate_ratio (float, optional): The rotating probability.
            Default: 0.5.
        mode (str, optional) : Indicates whether the angle is chosen in a
            random range (mode='range') or in a preset list of angles
            (mode='value'). Defaults to 'range'.
        angles_range(int|list[int], optional): The range of angles.
            If mode='range', angle_ranges is an int and the angle is chosen
            in (-angles_range, +angles_ranges).
            If mode='value', angles_range is a non-empty list of int and the
            angle is chosen in angles_range.
            Defaults to 180 as default mode is 'range'.
        auto_bound(bool, optional): whether to find the new width and height
            bounds.
        rect_classes (None|list, optional): Specifies classes that needs to
            be rotated by a multiple of 90 degrees.
        allow_negative (bool, optional): Whether to allow an image that does
            not contain any bbox area. Default False.
        version  (str, optional): Angle representations. Defaults to 'le90'.
    """

    def __init__(self,
                 rotate_ratio=0.5,
                 mode='range',
                 angles_range=180,
                 auto_bound=False,
                 rect_classes=None,
                 allow_negative=False,
                 version='le90'):
        self.rotate_ratio = rotate_ratio
        self.auto_bound = auto_bound
        assert mode in ['range', 'value'], \
            f"mode is supposed to be 'range' or 'value', but got {mode}."
        if mode == 'range':
            assert isinstance(angles_range, int), \
                "mode 'range' expects angle_range to be an int."
        else:
            assert mmcv.is_seq_of(angles_range, int) and len(angles_range), \
                "mode 'value' expects angle_range as a non-empty list of int."
        self.mode = mode
        self.angles_range = angles_range
        self.discrete_range = [90, 180, -90, -180]
        self.rect_classes = rect_classes
        self.allow_negative = allow_negative
        self.version = version

    @property
    def is_rotate(self):
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.rotate_ratio

    def apply_image(self, img, bound_h, bound_w, interp=cv2.INTER_LINEAR):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        return cv2.warpAffine(
            img, self.rm_image, (bound_w, bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y)
        points
        """
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def create_rotation_matrix(self,
                               center,
                               angle,
                               bound_h,
                               bound_w,
                               offset=0):
        """Create rotation matrix."""
        center += offset
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if self.auto_bound:
            rot_im_center = cv2.transform(center[None, None, :] + offset,
                                          rm)[0, 0, :]
            new_center = np.array([bound_w / 2, bound_h / 2
                                   ]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm

    def filter_border(self, bboxes, h, w):
        """Filter the box whose center point is outside or whose side length is
        less than 5."""
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bbox, h_bbox = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & \
                    (w_bbox > 5) & (h_bbox > 5)
        return keep_inds

    def __call__(self, results):
        """Call function of PolyRandomRotate."""
        if not self.is_rotate:
            results['rotate'] = False
            angle = 0
        else:
            results['rotate'] = True
            if self.mode == 'range':
                angle = self.angles_range * (2 * np.random.rand() - 1)
            else:
                i = np.random.randint(len(self.angles_range))
                angle = self.angles_range[i]

            class_labels = results['gt_labels']
            for classid in class_labels:
                if self.rect_classes:
                    if classid in self.rect_classes:
                        np.random.shuffle(self.discrete_range)
                        angle = self.discrete_range[0]
                        break

        h, w, c = results['img_shape']
        img = results['img']
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = \
            abs(np.cos(angle / 180 * np.pi)), abs(np.sin(angle / 180 * np.pi))
        if self.auto_bound:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos,
                 h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle,
                                                     bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)

        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)
        gt_bboxes = results.get('gt_bboxes', [])
        labels = results.get('gt_labels', [])

        if len(gt_bboxes):
            gt_bboxes = np.concatenate(
                [gt_bboxes, np.zeros((gt_bboxes.shape[0], 1))], axis=-1)
            polys = obb2poly_np(gt_bboxes, self.version)[:, :-1].reshape(-1, 2)
            polys = self.apply_coords(polys).reshape(-1, 8)
            gt_bboxes = []
            for pt in polys:
                pt = np.array(pt, dtype=np.float32)
                obb = poly2obb_np(pt, self.version) \
                    if poly2obb_np(pt, self.version) is not None\
                    else [0, 0, 0, 0, 0]
                gt_bboxes.append(obb)
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
            gt_bboxes = gt_bboxes[keep_inds, :]
            labels = labels[keep_inds]
        if len(gt_bboxes) == 0 and not self.allow_negative:
            return None
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = labels

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_ratio={self.rotate_ratio}, ' \
                    f'base_angles={self.base_angles}, ' \
                    f'angles_range={self.angles_range}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class RRandomCrop(RandomCrop):
    """Random crop the image & bboxes.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        iof_thr (float): The minimal iof between a object and window.
            Defaults to 0.7.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels must be aligned. That is, `gt_bboxes`
          corresponds to `gt_labels`, and `gt_bboxes_ignore` corresponds to
          `gt_labels_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 iof_thr=0.7,
                 version='oc'):
        self.version = version
        self.iof_thr = iof_thr
        super(RRandomCrop, self).__init__(crop_size, crop_type,
                                          allow_negative_crop)

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('bbox_fields', []):
            assert results[key].shape[-1] % 5 == 0

        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        height, width, _ = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, 0, 0, 0],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset

            windows = np.array([width / 2, height / 2, width, height, 0],
                               dtype=np.float32).reshape(-1, 5)

            valid_inds = box_iou_rotated(
                torch.tensor(bboxes), torch.tensor(windows),
                mode='iof').numpy().squeeze() > self.iof_thr

            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

        return results


@ROTATED_PIPELINES.register_module()
class RMosaic(Mosaic):
    """Rotate Mosaic augmentation. Inherit from
    `mmdet.datasets.pipelines.transforms.Mosaic`.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text
                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:
         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Defaults to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        version  (str, optional): Angle representations. Defaults to `oc`.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=10,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114,
                 prob=1.0,
                 version='oc'):
        super(RMosaic, self).__init__(
            img_scale=img_scale,
            center_ratio_range=center_ratio_range,
            min_bbox_size=min_bbox_size,
            bbox_clip_border=bbox_clip_border,
            skip_filter=skip_filter,
            pad_val=pad_val,
            prob=1.0)

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0] = \
                    scale_ratio_i * gt_bboxes_i[:, 0] + padw
                gt_bboxes_i[:, 1] = \
                    scale_ratio_i * gt_bboxes_i[:, 1] + padh
                gt_bboxes_i[:, 2:4] = \
                    scale_ratio_i * gt_bboxes_i[:, 2:4]

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            mosaic_bboxes, mosaic_labels = \
                self._filter_box_candidates(
                    mosaic_bboxes, mosaic_labels,
                    2 * self.img_scale[1], 2 * self.img_scale[0]
                )
        # If results after rmosaic does not contain any valid gt-bbox,
        # return None. And transform flows in MultiImageMixDataset will
        # repeat until existing valid gt-bbox.
        if len(mosaic_bboxes) == 0:
            return None

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _filter_box_candidates(self, bboxes, labels, w, h):
        """Filter out small bboxes and outside bboxes after Mosaic."""
        bbox_x, bbox_y, bbox_w, bbox_h = \
            bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        valid_inds = (bbox_x > 0) & (bbox_x < w) & \
                     (bbox_y > 0) & (bbox_y < h) & \
                     (bbox_w > self.min_bbox_size) & \
                     (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds]


@ROTATED_PIPELINES.register_module()
class FilterNoCenterObject:
    def __init__(self,
                 img_scale=(1100, 1100),
                 crop_size=(768, 768)):
        self.img_scale = img_scale
        self.crop_size = crop_size

    def __call__(self, results):
        bboxes = results['gt_bboxes']
        bboxes = bboxes.reshape((-1, 5))
        xy = bboxes[:, :2]
        flag = xy > (self.img_scale[0] - self.crop_size[0]) // 2
        flag = flag & (xy < ((self.img_scale[1] - self.crop_size[1]) // 2 + self.crop_size[1] - 1))
        flag = flag.all(axis=-1)
        if not flag.any():
            return None
        return results