from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
from os import path as osp
import json
from mmcv import Config
from mmcv import Config
from mmdet.apis import set_random_seed
import mmcv
from mmdet.models import build_detector
import os.path as osp
from mmcv import Config
import numpy as np
from mmdet.apis import set_random_seed
import mmcv
import os
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp

config_file = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
config_file = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/faster_rcnn/faster_rcnn_r50_fpn_1x_ECP.py"
checkpoint_file = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"


cfg = Config.fromfile(config_file)
out_folder = f"exps/"
load_from = checkpoint_file

max_epochs = 3
evaluation_interval = 1
seed = 0

datasets = [build_dataset([cfg.data.train])]
cfg.model.roi_head.bbox_head.num_classes = len(datasets[0].CLASSES)
cfg.load_from = load_from
model = build_detector(cfg.model)

# %% Runner
cfg.runner.max_epochs = max_epochs
cfg.evaluation.interval = evaluation_interval
cfg.checkpoint_config.interval = max_epochs
cfg.seed = seed
set_random_seed(seed, deterministic=False)

# %% CUDA
cfg.data.workers_per_gpu = 0
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

# %% Logs, working dir to save files and logs.

cfg.log_config.interval = 1
cfg.work_dir = f'{out_folder}'
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]



# %% Dump config file
cfg.dump(osp.join(cfg.work_dir, "cfg.py"))
# %%
cfg.evaluation["save_best"] = "bbox"

# %% Launch
train_detector(model, datasets, cfg, distributed=False, validate=True)