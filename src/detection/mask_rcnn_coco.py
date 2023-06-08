import os
import torch
from tqdm import tqdm
import json
from mmdet.apis import init_detector, inference_detector
import os.path as osp
from configs_path import ROOT_DIR, MMDET_DIR, CHECKPOINT_DIR

from .detector import Detector

class MaskRCNNCoco(Detector):
    def __init__(self, name, device="cuda", nms=False, task="pedestrian_detection"):
        super().__init__(name, device, nms, task)

        # todo handle paths
        self.config_path = f'{MMDET_DIR}/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
        self.checkpoint_path = f'{CHECKPOINT_DIR}/detection/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    @staticmethod
    def inference_processor(x):
        return x[0][0]