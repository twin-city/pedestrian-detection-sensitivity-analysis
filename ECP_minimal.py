import os

import matplotlib.pyplot as plt
import pandas as pd
import setuptools.errors
from utils import filter_gt_bboxes, plot_results_img, compute_ffpi_against_fp2
import numpy as np
import os.path as osp
import json
import torch

#%% params
dataset_name = "EuroCityPerson"
model_name = "faster-rcnn_cityscapes"
max_sample = 15 # Uniform sampled in dataset

# Dataset #todo add statistical comparison between datasets
#from src.preprocessing.motsynth_processing import MotsynthProcessing
#motsynth_processor = MotsynthProcessing(max_samples=max_sample, video_ids=None)

#%%

def get_ECP_annotations_and_imagepaths(time, set, city, max_samples=100000):

    # Here ECP specific
    root = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/labels/{set}/{city}"
    total_frame_ids = [x.split(".json")[0] for x in os.listdir(root) if ".json" in x]

    # Init dicts for bboxes annotations and metadata
    targets = {}
    targets_metadata = {}
    for i, frame_id in enumerate(total_frame_ids):

        # set max samples #todo
        if i>max_samples:
            break

        # Load ECP annotations
        json_path = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/labels/{set}/{city}/{frame_id}.json"
        with open(json_path) as jsonFile:
            annot_ECP = json.load(jsonFile)

            target = [
              dict(
                boxes=torch.tensor(
                    [(c["x0"], c["y0"], c["x1"], c["y1"]) for c in annot_ECP["children"] if c["identity"] in ["pedestrian", "rider"]] #todo might not be the thing todo
                ),
              )
            ]

            target[0]["labels"] = torch.tensor([0]*len(target[0]["boxes"]))

            # Keep only if at least 1 pedestrian
            if len(target[0]["boxes"]) > 0:
                targets[frame_id] = target

                tags = [c["tags"] for c in annot_ECP["children"] if c["identity"] in ["pedestrian", "rider"]]
                areas = [(c["x1"]-c["x0"]) *  (c["y1"]-c["y0"]) for c in annot_ECP["children"] if c["identity"] in ["pedestrian", "rider"]]
                iscrowd = [1*("group" in c["identity"]) for c in annot_ECP["children"] if c["identity"] in ["pedestrian", "rider"]]

                targets_metadata[frame_id] = (annot_ECP["tags"], tags, areas, iscrowd)

    frame_id_list = list(targets.keys())
    img_path_list = []
    for frame_id in frame_id_list:
        # print(frame_id)
        img_path = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/img/val/{city}/{frame_id}.png"
        img_path_list.append(img_path)

    # todo plot it ?

    return targets, targets_metadata, frame_id_list, img_path_list


#%%
#todo see the categories

targets = {}
frame_id_list = []
img_path_list = []
df_frame_metadata = pd.DataFrame()
df_gtbbox_metadata = pd.DataFrame()

for luminosity in ["day", "night"]:
    for chosen_set in ["val"]:
        for city in os.listdir(f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{luminosity}/img/{chosen_set}"):
            if city not in ["berlin_small"]:
                print(luminosity, city)

                targets_folder, targets_metadata_folder, frame_id_list_folder, img_path_list_folder = get_ECP_annotations_and_imagepaths(luminosity, chosen_set, city, max_samples=max_sample)


                # Df frame                 # print(np.unique([val[0] for key,val in targets_metadata.items()]))
                categories = ["motionBlur", "rainy", "wiper", "lenseFlare", "constructionSite"]
                df_frames_metadata_folder = pd.DataFrame({key: [cat in val[0] for cat in categories] for key, val in targets_metadata_folder.items()}).T
                df_frames_metadata_folder.columns = categories
                df_frames_metadata_folder["is_night"] = luminosity == "night"
                df_frames_metadata_folder["city"] = city
                df_frames_metadata_folder["path"] = img_path_list_folder
                df_frames_metadata_folder["frame_id"] = frame_id_list_folder
                df_frames_metadata_folder["adverse_weather"] = df_frames_metadata_folder["rainy"]


                # Df bbox_gt
                # {key: {key: v for v in val[1]} for key,val in targets_metadata.items()}
                categories_gtbbox = [f"occluded>{i}0" for i in range(1, 10)] + ["depiction"]
                df_gt_bbox_folder = pd.DataFrame()
                for key, val in targets_metadata_folder.items():
                    df_gt_bbox_frame = pd.DataFrame(val[1:]).T
                    df_gt_bbox_frame["frame_id"] = key
                    for cat in categories_gtbbox:
                        try:
                            df_gt_bbox_frame[cat] = cat in val[0]
                        except:
                            print("coucou")
                    df_gt_bbox_frame.drop(columns=[0], inplace=True)
                    df_gt_bbox_frame.rename(columns={1: "area", 2: "iscrowd"}, inplace=True)
                    df_gt_bbox_folder = pd.concat([df_gt_bbox_folder, df_gt_bbox_frame])

                # Add the folder
                targets.update(targets_folder)
                frame_id_list += frame_id_list_folder
                img_path_list += img_path_list_folder
                df_frame_metadata = pd.concat([df_frame_metadata, df_frames_metadata_folder])
                df_gtbbox_metadata = pd.concat([df_gtbbox_metadata, df_gt_bbox_folder])

df_gtbbox_metadata = df_gtbbox_metadata.set_index("frame_id")

#%%
from src.detection.detector import Detector
detector = Detector(model_name, device="cuda")
preds = detector.get_preds_from_files(dataset_name, frame_id_list, img_path_list)


#%%Plot example

i = 20
frame_id = frame_id_list[i]
img_path = img_path_list[i]
plot_results_img(img_path, frame_id, preds, targets, [])
print(img_path_list)
print(frame_id_list)

#%% Compute the metrics
gtbbox_filtering = {}
df_mr_fppi = compute_ffpi_against_fp2(dataset_name, model_name, preds, targets, df_gtbbox_metadata, gtbbox_filtering)


#%% Concat results and metadata
df_analysis = pd.merge(df_mr_fppi, df_frame_metadata, on="frame_id")
df_analysis_frame = df_analysis.groupby("frame_id").apply(lambda x: x.mean())

#%% study correlations
import matplotlib.pyplot as plt
frame_cofactors = ["rainy", "is_night"]
metrics = ["MR", "FPPI"]
from scipy.stats import pearsonr
corr_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[0])
p_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[1])

print(p_matrix)
import seaborn as sns
sns.heatmap(corr_matrix, annot=True)
plt.show()

sns.heatmap(p_matrix, annot=True)
plt.show()

#%% Day 'n Night
import matplotlib.pyplot as plt

night_ids = df_analysis_frame[df_analysis_frame["is_night"]==1].index.to_list()
day_ids = df_analysis_frame[df_analysis_frame["is_night"]==0].index.to_list()

metrics_day = df_mr_fppi.loc[day_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
metrics_night = df_mr_fppi.loc[night_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))

fig, ax = plt.subplots(1,1)
ax.plot(metrics_day["MR"], metrics_day["FPPI"], label="day")
ax.plot(metrics_night["MR"], metrics_night["FPPI"], label="night")

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(0.1, 1)
ax.set_xlim(0.1, 20)
plt.legend()
plt.show()
