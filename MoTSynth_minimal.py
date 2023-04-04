from utils import *
import numpy as np
import os.path as osp


#%%

# Parameters data
video_ids = ["004", "170","130", "033", "103", "107", "145"]
max_sample = 50

#todo bug 140, 174


#%% params

# model
model_name = "cityscapes"
device = "cuda"

if model_name == "cityscapes":
    checkpoint_root = "/home/raphael/work/checkpoints/detection"
    configs_root = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs"
    faster_rcnn_cityscapes_pth = "faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
    faster_rcnn_cityscapes_cfg = "models/faster_rcnn/faster_rcnn_cityscapes.py"
    config_file = osp.join(configs_root, faster_rcnn_cityscapes_cfg)
    checkpoint_file = osp.join(checkpoint_root, faster_rcnn_cityscapes_pth)
else:
    raise ValueError(f"Model name {model_name} not known")

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

# GT filtering #todo minimal value for now
gtbbox_filtering = {"occlusion_rate": (0.9, "max")}

# Cofactor to explore #todo discrete or continuous
cofactor = "is_night"

frame_ids_night = df_frame_metadata[df_frame_metadata["is_night"] == 1].index.to_list()
avrg_fp_list_1, avrg_missrate_list_1 = compute_ffpi_against_fp2(preds, targets, df_gtbbox_metadata, gtbbox_filtering, frame_ids_night)

frame_ids_day = df_frame_metadata[df_frame_metadata["is_night"] == 0].index.to_list()
avrg_fp_list_2, avrg_missrate_list_2 = compute_ffpi_against_fp2(preds, targets, df_gtbbox_metadata, gtbbox_filtering, frame_ids_day)



#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.plot(avrg_fp_list_1, avrg_missrate_list_1, c="green", label="Day test set")
ax.scatter(avrg_fp_list_1, avrg_missrate_list_1, c="green")

ax.plot(avrg_fp_list_2, avrg_missrate_list_2, c="purple", label="Night test set")
ax.scatter(avrg_fp_list_2, avrg_missrate_list_2, c="purple")

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(0.1, 1)
ax.set_xlim(0.1, 20)

plt.legend()
plt.show()
