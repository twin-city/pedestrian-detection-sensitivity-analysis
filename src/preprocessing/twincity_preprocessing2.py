import os
import platform
import numpy as np
import re
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import glob
import os.path as osp
from torchvision.ops import masks_to_boxes
import torch
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
from https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html
"""

root = "/home/raphael/work/datasets/twincity-Unreal/v4/"
cases = glob.glob(osp.join(root, "*"))


#%% Functions

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime

def code_rgba_str_2_code_rgba_float(x):
    # define the regular expression pattern to match the float values
    pattern = r'R=([\d\.]+),G=([\d\.]+),B=([\d\.]+),A=([\d\.]+)'
    # use re.findall() to extract the float values as strings
    float_strs = re.findall(pattern, x)[0]
    # convert the float strings to floats and store them in a tuple
    return tuple(map(float, float_strs))


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


case = cases[0]

metadata_path = glob.glob(osp.join(case, "*.json"))[0]
images_path_list = glob.glob(osp.join(case, "*.png"))

import json
with open(metadata_path) as file:
    metadata = json.load(file)

ordering = np.argsort([creation_date(x) for x in images_path_list])


#%%


img_annot_list = np.array(images_path_list)[ordering[::2]]
img_rgb_list = np.array(images_path_list)[ordering[1::2]]

img_rgb = mpimg.imread(img_rgb_list[0])
img_annot = mpimg.imread(img_annot_list[0])

#%% for a tuple (img_rgb, img_annot) get all the bounding boxes

import pandas as pd

excl_colors = [
    '(R=0.160640,G=0.000000,B=1.000000,A=1.000000)',
    '(R=0.143117,G=1.000000,B=0.000000,A=1.000000)',
]

img = img_annot

def get_twincity_boxes(img):

    num_pedestrian_in_scene = len(metadata["peds"])

    mask_list = []
    for i in range(num_pedestrian_in_scene):
        if metadata["peds"][str(i)] not in excl_colors:
            code = code_rgba_str_2_code_rgba_float(metadata["peds"][str(i)])
            mask = torch.tensor(((img-code)**2).sum(axis=2)<1e-4)
            if mask.sum() > 0:
                mask_list.append(mask)

    masks = torch.stack(mask_list)
    boxes = masks_to_boxes(masks)

    # Calculate the heights and areas of the boxes
    heights = boxes[:, 3] - boxes[:, 1]
    areas = (boxes[:, 2] - boxes[:, 0]) * heights

    # Create a DataFrame with the heights and areas
    data = {'Height': heights, 'Area': areas}
    df = pd.DataFrame(data)

    return boxes, df



for img in img_annot_list:
    boxes, df = get_twincity_boxes(img)

#todo rename

#%% Return

img_path_list = img_rgb_list
root = '/home/raphael/work/datasets/twincity-Unreal/v4/'


#metadatas = df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata
#root, targets, metadatas, frame_id_list, img_path_list

#%%
img = img_annot
img_rgb_torch = torch.tensor(img_rgb*255, dtype=torch.uint8)[:,:,:3]
img_rgb_torch = torch.swapaxes(img_rgb_torch, 0, 1)
img_rgb_torch = torch.swapaxes(img_rgb_torch, 0, 2)
plt.imshow(img)
plt.show()


from torchvision.utils import draw_bounding_boxes
drawn_boxes = draw_bounding_boxes(torch.tensor(img_rgb_torch), boxes,
                                  colors="red", width=4)
show(drawn_boxes)
plt.show()

#todo remove the sky one




