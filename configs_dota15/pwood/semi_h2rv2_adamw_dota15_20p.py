angle_version = 'le90'

import torchvision.transforms as transforms
from copy import deepcopy

# model settings
detector = dict(
    type='SemiRotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='SemiRotatedFCOSHeadH2RV2MCL',
        num_classes=16,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True, 
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        square_cls=[1, 9, 11],
        resize_cls=[1],
        scale_angle=False, #v2
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        # sup_loss-------------
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),  
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # aug_loss----------------
        loss_ss_symmetry=dict(
            type='SmoothL1Loss', loss_weight=0.2, beta=0.1)),  

    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

model = dict(
    type="H2RV2MCLTeacher",
    model=detector,
    semi_loss=dict(type='SemiGMMLoss',
                    cls_channels=16,
                    policy = 'high'),
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=12800,  
        sup_weight=1.0,
        unsup_weight=1.0,  #
        # unsup_aug_weight = 1.0,     #
        weight_suppress="exp",
        logit_specific_weights=dict(),
        cls_channels=16
    ),
    test_cfg=dict(inference_on="teacher"),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
common_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'tag')
                    
         )
]
strong_pipeline = [
    dict(type='DTToPILImage'),
    dict(type='DTRandomApply', operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(type='DTRandomApply', operations=[
        dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])
    ]),
    # dict(type='DTRandCrop'),
    dict(type='DTToNumpy'),
    dict(type="ExtraAttrs", tag="unsup_strong"),
]
weak_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="unsup_weak"),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="LoadEmptyAnnotations", with_bbox=True),
    dict(type="STMultiBranch", unsup_strong=deepcopy(strong_pipeline), unsup_weak=deepcopy(weak_pipeline),
         common_pipeline=common_pipeline, is_seq=True),
]
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddNoise', p=0.1),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="sup_weak"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg', 'tag')
         )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'DOTAv15WSOODDataset'
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane')
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=5,
    train=dict(
        type="SemiDataset",
        sup=dict(
            type=dataset_type,
            pipeline=sup_pipeline,
            ann_file="data/train_20p_labeled/annfiles/",
            img_prefix="data/train_20p_labeled/images/",
            version=angle_version,
            classes=classes
        ),
        unsup=dict(
            type=dataset_type,
            pipeline=unsup_pipeline,
            ann_file="data/train_20p_unlabeled/empty_annfiles/",
            img_prefix="data/train_20p_unlabeled/images/",
            version=angle_version,
            classes=classes,
            filter_empty_gt=False
        ),
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        img_prefix="data/val/images/",
        ann_file='data/val/annfiles/',
        version=angle_version,
        classes=classes
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        img_prefix="data/vis_val/images/",
        ann_file='data/vis_val/annfiles/',
        version=angle_version,
        classes=classes

    ),
    sampler=dict(
        train=dict(
            type="MultiSourceSampler",
            sample_ratio=[2, 1],
            seed=42
        )
    ),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    # dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.9996, interval=1, start_steps=3200),
    #dict(type = 'PrintThres')  
]

# evaluation
evaluation = dict(type="SubModulesDistEvalHook", interval=3200, metric='mAP',
                  save_best='mAP')

# optimizer
#optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    #_delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy                                                         
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=120000)

# 120k iters is enough for DOTA
runner = dict(type="IterBasedRunner", max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=3200, max_keep_ckpts=1)

# Default: disable fp16 training
# fp16 = dict(loss_scale="dynamic")

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="rotated_DenseTeacher_10percent",
        #         name="default_bce4cls",
        #     ),
        #     by_epoch=False,
        # ),
    ],
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]  # mode, iters

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

custom_imports = dict(imports=['semi_mmrotate'],
                      allow_failed_imports=False)