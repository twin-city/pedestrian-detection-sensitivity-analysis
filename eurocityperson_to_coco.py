
"""

import time

import json
json_path = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/day/labels/val/berlin_small/berlin_00386.json"
with open(json_path) as jsonFile:
    annot = json.load(jsonFile)

json_path = "/home/raphael/work/datasets/CARLA/output/fixed_spawn_Town01_v1/coco_600.json"
with open(json_path) as jsonFile:
    annot_coco = json.load(jsonFile)


info = {
    "date": time.
}
"""


import os
import json
import cv2
import mmcv
from tqdm import tqdm
import numpy as np

def ecp_to_coco(ecp_root, output_file):



    coco_annotations = {
        'info': {
            'description': 'EuroCity Persons (ECP) dataset in COCO format',
            'url': 'https://eurocity-dataset.tudelft.nl/',
            'version': '1.0',
            'year': 2023,
            'contributor': 'ECP Dataset Creators',
            'date_created': '2023/03/21'
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'pedestrian',
                'supercategory': 'person'
            }
        ]
    }

    #with open(ecp_annotation_file) as f:
    #    ecp_annotations = json.load(f)
    import os.path as osp
    ecp_annotations = mmcv.scandir("/media/raphael/Projects/datasets/EuroCityPerson/ECP/night/labels/val/budapest")

    annotation_id = 0
    for i, annot_file in enumerate(tqdm(ecp_annotations)):
        image_name = annot_file.split('.json')[0]
        image_id = i
        # image_path = os.path.join(ecp_root, "night/img/val/budapest", image_file)
        #image = cv2.imread(image_path)
        #h, w, _ = image.shape

        import json
        #json_path = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/day/labels/val/berlin_small/berlin_00386.json"
        with open(osp.join(ecp_root, "night/labels/val/budapest", annot_file)) as jsonFile:
            annot_ECP = json.load(jsonFile)

        w, h = annot_ECP["imagewidth"], annot_ECP["imageheight"]

        coco_annotations['images'].append({
            'id': image_id,
            'width': w,
            'height': h,
            'file_name': image_name + ".png",
            'license': "EuroCityPerson",
            'date_captured': ''
        })

        #todo fow now forgerring about children (bycicles ...)
        for ann in annot_ECP['children']:
            x0, y0, x1, y1 = [ann["x0"], ann["y0"], ann["x1"], ann["y1"]]
            x, y, w, h = x0, y0, x1-x0, y1-y0
            coco_annotations['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 0, #todo
                'segmentation': [],
                'area': w * h,
                'bbox': [x1, y1, w, h],
                'iscrowd': 0
            })
            annotation_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_annotations, f)

    return coco_annotations

if __name__ == '__main__':
    ecp_root = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/"
    #ecp_annotation_file = '/path/to/your/ecp/annotation/file.json'
    output_file = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/coco.json"

    coco_annotations = ecp_to_coco(ecp_root, output_file)

