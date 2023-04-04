from utils import *
import numpy as np
import os.path as osp


#%%

# Parameters data
video_ids = ["004", "170","130", "033", "103", "107", "145"]
max_sample = 100

#todo bug 140, 174


#%% params

# model loading

model_name = "yolo3_coco"
model_name = "faster-rcnn_coco"
model_name = "faster-rcnn_cityscapes"
device = "cuda"
checkpoint_root = "/home/raphael/work/checkpoints/detection"
configs_root = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs"


if model_name == "faster-rcnn_cityscapes":
    checkpoint_path = "faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
    config_path = "models/faster_rcnn/faster_rcnn_cityscapes.py"
elif model_name == "yolo3_coco":
    config_path = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/yolo/yolov3_d53_320_273e_coco.py"
    checkpoint_path = "/home/raphael/work/checkpoints/detection/yolov3_d53_320_273e_coco-421362b6.pth"
elif model_name == "faster-rcnn_coco":
    config_path = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
    checkpoint_path = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
else:
    raise ValueError(f"Model name {model_name} not known")
config_file = osp.join(configs_root, config_path)
checkpoint_file = osp.join(checkpoint_root, checkpoint_path)

targets, metadatas, frame_id_list, img_path_list = get_MoTSynth_annotations_and_imagepaths(video_ids=video_ids, max_samples=max_sample)
preds = get_preds_from_files(config_file, checkpoint_file, frame_id_list, img_path_list, device=device)

#%%

df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = metadatas
df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])
df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata["keypoints"].apply(lambda x: 1-(x-1).mean())
delay = 3

"""
sequence -> weather
sequence -> is_night
gtbbox -> occlusion_rate
"""

#%% Analyze results on an image

threshold = 0.6

# choose image
i = 1
frame_id = frame_id_list[i]
img_path = img_path_list[i]

occlusions_ids = np.where(df_gtbbox_metadata.loc[frame_id+delay, "occlusion_rate"] > 0.8)[0].tolist()

# plot
plot_results_img(img_path, frame_id, preds, targets, occlusions_ids)

# Compute metrics from image
pred_bbox, target_bbox = preds[frame_id], targets[frame_id]


#%% Compute depending on a condition


#todo a gt filtering for frames/sequence also ?

# GT filtering #todo minimal value for now
gtbbox_filtering = {"occlusion_rate": (0.9, "max"),
                    "area": (20, "min")}

# Cofactor to explore #todo discrete or continuous
cofactor = "weather"

df_mr_fppi = compute_ffpi_against_fp2(preds, targets, df_gtbbox_metadata, gtbbox_filtering, model_name)

#frame_ids_day = df_frame_metadata[df_frame_metadata[cofactor] != "THUNDER"].index.to_list()
#avrg_fp_list_2, avrg_missrate_list_2 = compute_ffpi_against_fp2(preds, targets, df_gtbbox_metadata, gtbbox_filtering, frame_ids_day)


def get_mr_fppi_curve(df_mr_fppi, frame_ids):
    metrics = df_mr_fppi.loc[frame_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
    mr = metrics["MR"]
    fppi = metrics["FPPI"]
    return mr, fppi

cofactor = "is_moving"
value = 1
thunder_frame_ids = df_frame_metadata[df_frame_metadata[cofactor] == value].index.to_list()
nothunder_frame_ids = df_frame_metadata[df_frame_metadata[cofactor] != value].index.to_list()



import matplotlib.pyplot as plt
fig, ax = plt.subplots()

mr, fppi = get_mr_fppi_curve(df_mr_fppi, thunder_frame_ids)
ax.plot(mr, fppi, c="green", label=f"has {cofactor}")
ax.scatter(mr, fppi, c="green")


mr, fppi = get_mr_fppi_curve(df_mr_fppi, nothunder_frame_ids)
ax.plot(mr, fppi, c="red", label=f"no has {cofactor}")
ax.scatter(mr, fppi, c="red")

"""
ax.plot(avrg_fp_list_2, avrg_missrate_list_2, c="purple", label="Night test set")
ax.scatter(avrg_fp_list_2, avrg_missrate_list_2, c="purple")
"""

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(0.1, 1)
ax.set_xlim(0.1, 20)

plt.legend()
plt.show()
