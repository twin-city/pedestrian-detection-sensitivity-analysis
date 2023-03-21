# dataset settings
#from configs.paths_cfg import ECP_ROOT
#from configs.datasets_cfg import train_pipeline, test_pipeline


ECP_ROOT = "/media/raphael/Projects/datasets/EuroCityPerson/ECP"


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


dataset_type = 'EuroCityPersonDataset'
classes = ['person', 'rider']

ECP_train = dict(
        pipeline=train_pipeline,
        type=dataset_type,
        classes=classes,
        ann_file=f'{ECP_ROOT}/coco_train.json',
        img_prefix=f'{ECP_ROOT}/day/img/val/berlin_small/')

ECP_val = dict(
        pipeline=test_pipeline,
        type=dataset_type,
        classes=classes,
        ann_file=f'{ECP_ROOT}/coco_val.json',
        img_prefix=f'{ECP_ROOT}/day/img/val/berlin_small/')

ECP_test = dict(
        pipeline=test_pipeline,
        type=dataset_type,
        classes=classes,
        ann_file=f'{ECP_ROOT}/coco_test.json',
        img_prefix=f'{ECP_ROOT}/day/img/val/berlin_small/')

data = dict(
    train=ECP_train,
    val=ECP_val,
    test=ECP_test,
)


evaluation = dict(interval=1, metric='mAP')