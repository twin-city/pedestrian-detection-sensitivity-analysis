import os
import torch
from tqdm import tqdm
import json
from mmdet.apis import init_detector, inference_detector
import os.path as osp
from configs_path import ROOT_DIR, MMDET_DIR, CHECKPOINT_DIR
from .detector import Detector

class FasterRCNCityscapesDetector(Detector):
    def __init__(self, name, device="cuda", nms=False, task="pedestrian_detection"):
        super().__init__(name, device, nms, task)

        # todo handle paths
        self.checkpoint_path = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
        self.config_path = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/faster_rcnn/faster-rcnn_cityscapes.py"

    @staticmethod
    def inference_processor(x):

        return x[0]