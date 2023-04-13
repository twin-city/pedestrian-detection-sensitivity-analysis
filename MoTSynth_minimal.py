import setuptools.errors

from utils import *
import numpy as np
import os.path as osp



#%% params

model_name = "faster-rcnn_cityscapes"
max_sample = 20

# Dataset #todo add statistical comparison between datasets
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(max_samples=max_sample, video_ids=None)
targets, metadatas, frame_id_list, img_path_list = motsynth_processor.get_MoTSynth_annotations_and_imagepaths()
df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = metadatas
delay = motsynth_processor.delay

# Detections
from src.detection.detector import Detector
detector = Detector(model_name)
preds = detector.get_preds_from_files(frame_id_list, img_path_list)



#########################################   Peform Tests   ############################################################

#%% Analyze results on an image

threshold = 0.6

# choose image
i = 1
frame_id = frame_id_list[i]
img_path = img_path_list[i]

occlusions_ids = np.where(df_gtbbox_metadata.loc[int(frame_id)+delay, "occlusion_rate"] > 0.8)[0].tolist()

# plot
plot_results_img(img_path, frame_id, preds, targets, occlusions_ids)

# Compute metrics from image
pred_bbox, target_bbox = preds[str(frame_id)], targets[str(frame_id)]


#%% Compute depending on a condition


#todo a gt filtering for frames/sequence also ? Also save it to save time

# GT filtering #todo minimal value for now
gtbbox_filtering = {"occlusion_rate": (0.9, "max"),
                    "area": (20, "min")}

df_mr_fppi = compute_ffpi_against_fp2(preds, targets, df_gtbbox_metadata, gtbbox_filtering, model_name)



#########################################   Peform Analysis   #########################################################



# Cofactor to explore #todo discrete or continuous
cofactor = "weather"

#%%

def get_mr_fppi_curve(df_mr_fppi, frame_ids):
    metrics = df_mr_fppi.loc[frame_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
    mr = metrics["MR"]
    fppi = metrics["FPPI"]
    return mr, fppi


adverse_weather = ['THUNDER', 'SMOG', 'FOGGY', 'BLIZZARD', 'RAIN', 'CLOUDS', 'OVERCAST'] # 'CLEAR' 'EXTRASUNNY',

df_frame_metadata["adverse_weather"] = df_frame_metadata["weather"].apply(lambda x: x in adverse_weather)

#todo combine multiple : weather and night ...

cofactor = "is_night"
value = 1

def listint2liststr(l):
    return [str(i) for i in l]

cof_frame_ids = listint2liststr(df_frame_metadata[df_frame_metadata[cofactor] == value].index.to_list())
nocof_frame_ids = listint2liststr(df_frame_metadata[df_frame_metadata[cofactor] != value].index.to_list())



import matplotlib.pyplot as plt
fig, ax = plt.subplots()

mr, fppi = get_mr_fppi_curve(df_mr_fppi, cof_frame_ids)
ax.plot(mr, fppi, c="green", label=f"has {cofactor}")
ax.scatter(mr, fppi, c="green")


mr, fppi = get_mr_fppi_curve(df_mr_fppi, nocof_frame_ids)
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


#%%
from utils import visual_check_motsynth_annotations
visual_check_motsynth_annotations(video_num="145", img_file_name="1455.jpg", shift=3)


#%% Analysis of p-value of cofactors (and plot it) for both MR and FPPI

import statsmodels.api as sm
# df_frame_metadata[["blizzard", "smog", "thunder"]] = pd.get_dummies(df_frame_metadata["weather"])[['BLIZZARD', 'SMOG', 'THUNDER']]


"""

#%% All at once
cofactors = ["is_night", 'pitch', 'roll', 'x', 'y', 'z','is_moving', "blizzard", "smog", "thunder"]
X = df_frame_metadata[cofactors]
Y = df_mr_fppi[df_mr_fppi["threshold"]==0.5]["MR"].loc[listint2liststr(X.index)]
X = sm.add_constant(X)
fit = sm.OLS(Y, X).fit()
for i, cofactor in enumerate(cofactors):
    print(fit.pvalues[i], cofactor)


#%% Separated
cofactors = ["is_night", 'pitch', 'roll', 'x', 'y', 'z','is_moving', "blizzard", "smog", "thunder"]

for cofactor in cofactors:
    X = df_frame_metadata[cofactor]
    Y = df_mr_fppi[df_mr_fppi["threshold"]==0.5]["MR"].loc[listint2liststr(X.index)]
    X = sm.add_constant(X)
    fit = sm.OLS(Y, X).fit()
    for i, cofactor in enumerate([cofactor]):
        print(fit.pvalues[i], cofactor)
"""



"""
Corriger BOnferroni + DataViz

Matrice de correlation des cofacteurs

Peut-être plutot les performances au niveau de la séquence ??? 
Par exemple pour weather oui... Car sinon les tests sont non indépendants

"""

"""

#%%
import pandas as pd
import seaborn as sns


# compute the correlation matrix using the corr method
correlation_matrix = df_frame_metadata[cofactors].corr()

# plot the correlation matrix using the heatmap function from seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
"""

