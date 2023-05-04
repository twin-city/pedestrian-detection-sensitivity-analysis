
"""
from https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html
"""
import glob
import os.path as osp

root = "/home/raphael/work/datasets/twincity-Unreal/v4/"
cases = glob.glob(osp.join(root, "*"))

#%%
""" What would help

Save semantic seg in .png, and with the same name

"""


case = cases[0]

metadata_path = glob.glob(osp.join(case, "*.json"))[0]
images_path_list = glob.glob(osp.join(case, "*.png"))

import json
with open(metadata_path) as file:
    metadata = json.load(file)

#%%

import os
import platform

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


#creation_date(x) for x in images_path_list]
import numpy as np
ordering = np.argsort([creation_date(x) for x in images_path_list])

#%%



for i in ordering:
    # read image using matplotlib
    img = mpimg.imread(images_path_list[i])

    # plot image
    plt.imshow(img)
    plt.show()

#%%
import torch
img = mpimg.imread(images_path_list[ordering[-2]])#[800:,450:480]
img_rgb = mpimg.imread(images_path_list[ordering[-1]])#[800:,450:480]


img = mpimg.imread(images_path_list[ordering[-2]])#[:200,850:950]
img_rgb = mpimg.imread(images_path_list[ordering[-1]])#[800:,]

img_rgb_torch = torch.tensor(img_rgb*255, dtype=torch.uint8)[:,:,:3]
img_rgb_torch = torch.swapaxes(img_rgb_torch, 0, 1)
img_rgb_torch = torch.swapaxes(img_rgb_torch, 0, 2)

# plot image
plt.imshow(img)
plt.show()


#%% Select a pedestrian

import re

input_str = '(R=1.000000,G=0.000000,B=0.030066,A=1.000000)'

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


for i in range(10):
    code = code_rgba_str_2_code_rgba_float(metadata["peds"][str(i)])
    mask = torch.tensor(((img-code)**2).sum(axis=2)<1e-4)
    masks = torch.stack([mask])



    from torchvision.utils import draw_segmentation_masks
    import torchvision.transforms.functional as F


    drawn_masks = []
    for mask in masks:
        drawn_masks.append(draw_segmentation_masks(img_rgb_torch, mask, alpha=0.8, colors="blue"))
    show(drawn_masks)
    plt.title(f"Pedestrian with id {i}")
    plt.show()

#%%

# Compute boxes
from torchvision.ops import masks_to_boxes


from torchvision.utils import draw_bounding_boxes
drawn_boxes = draw_bounding_boxes(img_rgb_torch, boxes, colors="red")
show(drawn_boxes)
plt.show()


#%%

import torch
img_annot_list = np.array(images_path_list)[ordering[::2]]
img_rgb_list = np.array(images_path_list)[ordering[1::2]]

img_rgb = mpimg.imread(img_rgb_list[0])
img_annot = mpimg.imread(img_annot_list[0])

#%% for a tuple (img_rgb, img_annot) get all the bounding boxes

num_pedestrian_in_scene = len(metadata["peds"])


mask_list = []

for i in range(num_pedestrian_in_scene):
    if code !=
    code = code_rgba_str_2_code_rgba_float(metadata["peds"][str(i)])
    mask = torch.tensor(((img-code)**2).sum(axis=2)<1e-4)
    if mask.sum()>0:
        mask_list.append(mask)

masks = torch.stack(mask_list)

boxes = masks_to_boxes(masks)


#%%
img = img_annot

img_rgb_torch = torch.tensor(img_rgb*255, dtype=torch.uint8)[:,:,:3]
img_rgb_torch = torch.swapaxes(img_rgb_torch, 0, 1)
img_rgb_torch = torch.swapaxes(img_rgb_torch, 0, 2)

plt.imshow(img)
plt.show()


from torchvision.utils import draw_bounding_boxes
drawn_boxes = draw_bounding_boxes(torch.tensor(img_rgb_torch), boxes, colors="red")
show(drawn_boxes)
plt.show()

#todo remove the sky one




