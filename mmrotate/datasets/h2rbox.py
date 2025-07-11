from mmrotate.datasets.dota import DOTADataset
from mmrotate.datasets.dior import DIORDataset
from mmrotate.datasets.hrsc import HRSCDataset
from mmrotate.datasets.dior_dota import DIOR_DOTADataset
from .COD import COD_DOTADataset
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.core.bbox.transforms import obb2hbb
import torch
import numpy as np
import mmcv
import re
from functools import partial
from collections import defaultdict
from mmcv.ops import nms_rotated


@ROTATED_DATASETS.register_module()
class HRSCWSOODDataset(HRSCDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_subdir='JPEGImages',
                 ann_subdir='Annotations',
                 classwise=False,
                 version='oc',
                 rect_classes=None,
                 weak_supervised=True,
                 **kwargs):
        self.rect_classes = rect_classes
        self.weak_supervised = weak_supervised

        super(HRSCWSOODDataset, self).__init__(ann_file, pipeline,
                                               img_subdir, ann_subdir,
                                               classwise, version, **kwargs)

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx).copy()
        if self.weak_supervised:
            ann_info['bboxes'] = obb2hbb(
                torch.from_numpy(ann_info['bboxes']),
                version=self.version).numpy()
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)


@ROTATED_DATASETS.register_module()
class DIORWSOODDataset(DIORDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_subdir='JPEGImages-trainval',
                 ann_subdir='Annotations/Oriented Bounding Boxes/',
                 version='oc',
                 xmltype='hbb',
                 rect_classes=None,
                 weak_supervised=True,
                 **kwargs):
        self.rect_classes = rect_classes
        self.weak_supervised = weak_supervised

        super(DIORWSOODDataset, self).__init__(ann_file, pipeline,
                                               img_subdir, ann_subdir,
                                               version, xmltype, **kwargs)

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx).copy()
        if self.weak_supervised:
            ann_info['bboxes'] = obb2hbb(
                torch.from_numpy(ann_info['bboxes']),
                version=self.version).numpy()
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

@ROTATED_DATASETS.register_module()
class DIOR_DOTAWSOODDataset(DIOR_DOTADataset):

    CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt',
               'bridge', 'chimney', 'expressway-service-area',
               'expressway-toll-station', 'dam', 'golffield',
               'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
               'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
               'windmill')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 rect_classes=None,
                 weak_supervised=True,
                 **kwargs):
        self.rect_classes = rect_classes
        self.weak_supervised = weak_supervised

        super(DIOR_DOTAWSOODDataset, self).__init__(ann_file, pipeline,
                                               version, difficulty, **kwargs)
    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx).copy()
        if self.weak_supervised:
            ann_info['bboxes'] = obb2hbb(
                torch.from_numpy(ann_info['bboxes']),
                version=self.version).numpy()
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def obb2hbb_np_le90(self, obboxes):
        """Convert oriented bounding boxes to horizontal bounding boxes.

        Args:
            obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

        Returns:
            hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
        """
        # center, w, h, theta = obboxes[..., :2], obboxes[..., 2], obboxes[..., 3], obboxes[..., 4]
        center, w, h, theta = np.split(obboxes, [2, 3, 4], axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)
        x_bias = np.abs(w / 2 * Cos) + np.abs(h / 2 * Sin)
        y_bias = np.abs(w / 2 * Sin) + np.abs(h / 2 * Cos)
        bias = np.concatenate([x_bias, y_bias], axis=-1)
        hbboxes = np.concatenate([center - bias, center + bias], axis=-1)
        _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
        _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
        _w = hbboxes[..., 2] - hbboxes[..., 0]
        _h = hbboxes[..., 3] - hbboxes[..., 1]
        _theta = np.zeros(theta.shape[0])
        obboxes1 = np.stack([_x, _y, _w, _h, _theta], axis=-1)
        obboxes2 = np.stack([_x, _y, _h, _w, _theta - np.pi / 2], axis=-1)
        obboxes = np.where((_w >= _h)[..., None], obboxes1, obboxes2)
        return obboxes

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                if self.rect_classes:
                    if i in self.rect_classes:
                        ori_bboxes = self.obb2hbb_np_le90(ori_bboxes)
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

@ROTATED_DATASETS.register_module()
class COD_DOTAWSOODDataset(COD_DOTADataset):

    CLASSES = ('car','truck', 'traffic-sign', 'people', 
                'motor', 'bicycle', 'traffic-light', 'tricycle',
                'bridge', 'bus', 'boat', 'ship')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 rect_classes=None,
                 weak_supervised=True,
                 **kwargs):
        self.rect_classes = rect_classes
        self.weak_supervised = weak_supervised

        super(COD_DOTAWSOODDataset, self).__init__(ann_file, pipeline,
                                               version, difficulty, **kwargs)
    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx).copy()
        if self.weak_supervised:
            ann_info['bboxes'] = obb2hbb(
                torch.from_numpy(ann_info['bboxes']),
                version=self.version).numpy()
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def obb2hbb_np_le90(self, obboxes):
        """Convert oriented bounding boxes to horizontal bounding boxes.

        Args:
            obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

        Returns:
            hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
        """
        # center, w, h, theta = obboxes[..., :2], obboxes[..., 2], obboxes[..., 3], obboxes[..., 4]
        center, w, h, theta = np.split(obboxes, [2, 3, 4], axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)
        x_bias = np.abs(w / 2 * Cos) + np.abs(h / 2 * Sin)
        y_bias = np.abs(w / 2 * Sin) + np.abs(h / 2 * Cos)
        bias = np.concatenate([x_bias, y_bias], axis=-1)
        hbboxes = np.concatenate([center - bias, center + bias], axis=-1)
        _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
        _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
        _w = hbboxes[..., 2] - hbboxes[..., 0]
        _h = hbboxes[..., 3] - hbboxes[..., 1]
        _theta = np.zeros(theta.shape[0])
        obboxes1 = np.stack([_x, _y, _w, _h, _theta], axis=-1)
        obboxes2 = np.stack([_x, _y, _h, _w, _theta - np.pi / 2], axis=-1)
        obboxes = np.where((_w >= _h)[..., None], obboxes1, obboxes2)
        return obboxes

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                if self.rect_classes:
                    if i in self.rect_classes:
                        ori_bboxes = self.obb2hbb_np_le90(ori_bboxes)
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)
    


@ROTATED_DATASETS.register_module()
class DOTAWSOODDataset(DOTADataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 rect_classes=None,
                 weak_supervised=True,
                 **kwargs):
        self.rect_classes = rect_classes
        self.weak_supervised = weak_supervised

        super(DOTAWSOODDataset, self).__init__(ann_file, pipeline,
                                               version, difficulty, **kwargs)

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx).copy()
        if self.weak_supervised:
            ann_info['bboxes'] = obb2hbb(
                torch.from_numpy(ann_info['bboxes']),
                version=self.version).numpy()
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def obb2hbb_np_le90(self, obboxes):
        """Convert oriented bounding boxes to horizontal bounding boxes.

        Args:
            obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

        Returns:
            hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
        """
        # center, w, h, theta = obboxes[..., :2], obboxes[..., 2], obboxes[..., 3], obboxes[..., 4]
        center, w, h, theta = np.split(obboxes, [2, 3, 4], axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)
        x_bias = np.abs(w / 2 * Cos) + np.abs(h / 2 * Sin)
        y_bias = np.abs(w / 2 * Sin) + np.abs(h / 2 * Cos)
        bias = np.concatenate([x_bias, y_bias], axis=-1)
        hbboxes = np.concatenate([center - bias, center + bias], axis=-1)
        _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
        _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
        _w = hbboxes[..., 2] - hbboxes[..., 0]
        _h = hbboxes[..., 3] - hbboxes[..., 1]
        _theta = np.zeros(theta.shape[0])
        obboxes1 = np.stack([_x, _y, _w, _h, _theta], axis=-1)
        obboxes2 = np.stack([_x, _y, _h, _w, _theta - np.pi / 2], axis=-1)
        obboxes = np.where((_w >= _h)[..., None], obboxes1, obboxes2)
        return obboxes

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                if self.rect_classes:
                    if i in self.rect_classes:
                        ori_bboxes = self.obb2hbb_np_le90(ori_bboxes)
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)


@ROTATED_DATASETS.register_module()
class DOTAv15WSOODDataset(DOTAWSOODDataset):

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
               'container-crane')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255), (220, 20, 60)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 rect_classes=None,
                 weak_supervised=True,
                 **kwargs):

        super(DOTAv15WSOODDataset, self).__init__(ann_file, pipeline,
                                                  version, difficulty,
                                                  rect_classes, weak_supervised,
                                                  **kwargs)


@ROTATED_DATASETS.register_module()
class DOTAv2WSOODDataset(DOTAWSOODDataset):

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
               'container-crane', 'airport', 'helipad')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255), (220, 20, 60),
               (119, 11, 32), (0, 0, 142)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 rect_classes=None,
                 weak_supervised=True,
                 **kwargs):

        super(DOTAv2WSOODDataset, self).__init__(ann_file, pipeline,
                                                 version, difficulty,
                                                 rect_classes, weak_supervised,
                                                 **kwargs)


@ROTATED_DATASETS.register_module()
class SARWSOODDataset(DOTAWSOODDataset):
    """SAR ship dataset for detection (Support RSSDD and HRSID)."""

    CLASSES = ('ship',)
    PALETTE = [
        (0, 255, 0),
    ]


@ROTATED_DATASETS.register_module()
class STARWSOODDataset(DOTAWSOODDataset):

    CLASSES = ('ship','boat','crane','goods_yard','tank','storehouse','breakwater','dock',
               'airplane','boarding_bridge','runway','taxiway','terminal','apron','gas_station',
               'truck','car','truck_parking','car_parking','bridge','cooling_tower','chimney',
               'vapor','smoke','genset','coal_yard','lattice_tower', 'substation', 'wind_mill',
               'cement_concrete_pavement', 'toll_gate', 'flood_dam', 'gravity_dam', 'ship_lock',
               'ground_track_field','basketball_court','engineering_vehicle', 'foundation_pit', 
               'intersection', 'soccer_ball_field','tennis_court','tower_crane','unfinished_building',
               'arch_dam','roundabout','baseball_diamond','stadium','containment_vessel')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255),
               (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208), (166, 196, 102), (208, 195, 210), (255, 109, 65), 
               (0, 143, 149), (179, 0, 194)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 rect_classes=None,
                 weak_supervised=True,
                 **kwargs):

        super(STARWSOODDataset, self).__init__(ann_file, pipeline,
                                              version, difficulty,
                                              rect_classes, weak_supervised,
                                              **kwargs)


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results