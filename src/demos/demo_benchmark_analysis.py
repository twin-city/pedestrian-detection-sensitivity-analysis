import numpy as np

#%% params of input dataset

# Parameters for results generation
from src.demos.configs import ODD_limit, ODD_criterias, param_heatmap_metrics, metrics, \
    gtbbox_filtering_all, dict_filter_frames, gtbbox_filtering_cats

# Which models to study
model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]
dataset_names = ["motsynth", "twincity", "EuroCityPerson"]
max_samples = [600, 50, 30]

# Dataset


dataset_name = "coco_Fudan"
max_samples = 200
root = "/home/raphael/work/datasets/PennFudanPed"
coco_json_path = "/home/raphael/work/datasets/PennFudanPed/coco.json"


#dataset_name, max_sample = "motsynth", 600
#

#%%
from src.dataset.dataset_factory import DatasetFactory
#for dataset_name, max_sample in zip(dataset_names, max_samples):


# dataset_name, max_sample = "motsynth", 600
#dataset_name, max_sample = "EuroCityPerson", 30
# dataset_name, max_sample = "twincity", 50

dataset = DatasetFactory.get_dataset(dataset_name, max_samples, root=root, coco_json_path=coco_json_path)
dataset_tuple = dataset.get_dataset_as_tuple()
root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset_tuple


#%% See Dataset Characteristics ==============================

# Correlations
sequence_cofactors = ["is_night"]
dataset.plot_dataset_sequence_correlation(sequence_cofactors)

# Height & Occlusion
dataset.plot_dataset_statistics()



#%% Compute metrics and plot them ==============================

#%% Compute metrics
from src.utils import compute_models_metrics_from_gtbbox_criteria
gtbbox_filtering = gtbbox_filtering_all
df_analysis = compute_models_metrics_from_gtbbox_criteria(dataset, gtbbox_filtering, model_names)

#%% Model performance :  Plots MR vs FPPI on frame filtering
from src.plot_utils import plot_fppi_mr_vs_frame_cofactor
plot_fppi_mr_vs_frame_cofactor(df_analysis, dict_filter_frames, ODD_criterias, results_dir="")

#%% Model performance : Plot MR vs FPPI on gtbbox filtering

#gtbbox_filtering = gtbbox_filtering_cats
#df_analysis_cats = compute_models_metrics_from_gtbbox_criteria(dataset, gtbbox_filtering, model_names)
from src.plot_utils import plot_fppi_mr_vs_gtbbox_cofactor
#plot_fppi_mr_vs_gtbbox_cofactor(df_analysis_cats, ODD_criterias=None)


#%% do the plot value difference (simplified, each metric)
from src.plot_utils import plot_heatmap_metrics
thresholds = [0.5, 0.9, 0.99]
df_analysis_heatmap = df_analysis[np.isin(df_analysis["threshold"], thresholds)]
plot_heatmap_metrics(df_analysis_heatmap, model_names, metrics, ODD_limit,
                     param_heatmap_metrics=param_heatmap_metrics, results_dir=dataset.results_dir)


#%% Study the Ground Truth Bounding Boxes : does their detection (matched) is correlated with their metadata ? (bheight, occlusion ?)

#from src.plot_utils import plot_gtbbox_matched_correlations
#threshold = 0.9
#features_bbox = ['height', "occlusion_rate", "aspect_ratio_is_typical"]
#plot_gtbbox_matched_correlations(model_names, dataset, features_bbox, threshold, gtbbox_filtering)


#%% Study a particular image with one of the model ==============================

# Plot an image in particular
from src.plot_utils import plot_image_with_detections
frame_idx = 1
model_name = model_names[0]
plot_thresholds = [0., 0.9, 0.9999]
gtbbox_filtering = gtbbox_filtering_all["Overall"]
plot_image_with_detections(dataset_tuple, dataset_name, model_name, plot_thresholds, gtbbox_filtering, i=frame_idx)