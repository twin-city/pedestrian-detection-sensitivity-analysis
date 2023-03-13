import matplotlib.pyplot as plt

from configs.paths_cfg import *


#%% Show results on images / bboxes
from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
#checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
checkpoint_file = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results

img_path = "/home/raphael/work/code/dataset_conversion_utils/caltech-pedestrian-dataset-converter/data/images/set01_V000_1680.png"
result = inference_detector(model, img_path)

model.show_result(img_path, result, out_file='result.jpg')
#%%

#%% BBOXes people

import cv2
import matplotlib.pyplot as plt
bboxes_people = result[0]

img = plt.imread(img_path)


for bbox in bboxes_people:
    x1, y1, x2, y2, p = [int(v) for v in bbox]
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, int(255)), 1)


plt.imshow(img)
plt.show()
print("coucou")

#%% Now that I have tje bboxes what do I do ?

"""
https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734


- IoU on synscapes


On : a benchmark --> Miss rate
 
 This is preferred to precision recall curves for certain tasks, e.g. automotive applications, as typically there is an upper limit on the acceptable false positives perimage rate independent of pedestrian density.

"""




#%% Correlate result metrics to bbox properties / image property