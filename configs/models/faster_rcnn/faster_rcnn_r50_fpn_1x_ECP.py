#from configs.paths_cfg import MMDETECTION_ROOT
import os

MMDETECTION_ROOT = "../../../../mmdetection"

_base_ = [
    os.path.join(MMDETECTION_ROOT, 'configs/_base_/models/faster_rcnn_r50_fpn.py'),
    '../../../configs/datasets/ECP_coco.py',
#    '../../../configs/datasets/CARLA_detection.py',
    os.path.join(MMDETECTION_ROOT, 'configs/_base_/schedules/schedule_1x.py'),
    os.path.join(MMDETECTION_ROOT, 'configs/_base_/default_runtime.py'),
]

model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))


# model["roi_head"]["bbox_head"]["num_classes=3"] = 3
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'


