import os
import torch
from tqdm import tqdm
import json
from mmdet.apis import init_detector, inference_detector
import os.path as osp
from configs_path import ROOT_DIR, MMDET_DIR, CHECKPOINT_DIR
import numpy as np
from .detector import Detector

class MaskRCNNCoco(Detector):
    def __init__(self, name, device="cuda", nms=False, task="pedestrian_detection"):
        super().__init__(name, device, nms, task)

        # todo handle paths
        self.config_path = f'{MMDET_DIR}/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
        self.checkpoint_path = f'{CHECKPOINT_DIR}/detection/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    def inference_processor(self, x):
        pred = x[0][0]
        if self.task == "pedestrian_detection":
            return pred
        else:
            # ratios = [(bbox[2] - bbox[0])/(bbox[3]-bbox[1]) for bbox in x[0]]
            threshold = 0.75
            return np.array([bbox for bbox in pred if (bbox[2] - bbox[0])/(bbox[3]-bbox[1]) > threshold])