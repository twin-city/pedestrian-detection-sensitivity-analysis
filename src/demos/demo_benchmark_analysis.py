import numpy as np
from src.dataset.dataset_factory import DatasetFactory

# Parameters for results generation
from src.demos.configs import ODD_limit, ODD_criterias, param_heatmap_metrics, metrics, \
    gtbbox_filtering_all, dict_filter_frames, gtbbox_filtering_cats

# Plot functions to show results
from src.plot_utils import plot_fppi_mr_vs_gtbbox_cofactor
from src.utils import compute_models_metrics_from_gtbbox_criteria
from src.plot_utils import plot_fppi_mr_vs_frame_cofactor
from src.plot_utils import plot_heatmap_metrics
from src.plot_utils import plot_image_with_detections

#todo find a way to harmonize the coco_json_path
def run_demo_pedestrian_detection(root, dataset_name, max_samples, model_names, coco_json_path=None,
                                  dataset_analysis=False, frame_analysis=False, gtbbox_analysis=False,
                                  plot_image=False):

    # Default parameters
    thresholds = [0.5, 0.9, 0.99]

    #%% Load Dataset ==================================================================================================
    dataset = DatasetFactory.get_dataset(dataset_name, max_samples, root=root, coco_json_path=coco_json_path)
    dataset_tuple = dataset.get_dataset_as_tuple()

    #%% See Dataset Characteristics ====================================================================================

    if dataset_analysis:
        # Correlations
        sequence_cofactors = ["is_night"]
        dataset.plot_dataset_sequence_correlation(sequence_cofactors)
        # Height & Occlusion
        dataset.plot_dataset_statistics()


    #%% Compute metrics Overall and check them according to frame characteristics ======================================

    if frame_analysis:
        #%% Model performance :  Plots MR vs FPPI on frame filtering
        gtbbox_filtering = gtbbox_filtering_all
        df_analysis = compute_models_metrics_from_gtbbox_criteria(dataset, gtbbox_filtering, model_names)
        plot_fppi_mr_vs_frame_cofactor(df_analysis, dict_filter_frames, ODD_criterias, results_dir="")

        #%% Model performance :  Plots Metric difference on frame filtering
        df_analysis_heatmap = df_analysis[np.isin(df_analysis["threshold"], thresholds)]
        plot_heatmap_metrics(df_analysis_heatmap, model_names, metrics, ODD_limit,
                             param_heatmap_metrics=param_heatmap_metrics, results_dir=dataset.results_dir)

    #%% Compute metrics Overall and check them according to ground truth bounding box characteristics ==================

    if gtbbox_analysis:
        #%% Model performance : Plot MR vs FPPI on gtbbox filtering
        gtbbox_filtering = gtbbox_filtering_cats
        df_analysis_cats = compute_models_metrics_from_gtbbox_criteria(dataset, gtbbox_filtering, model_names)
        plot_fppi_mr_vs_gtbbox_cofactor(df_analysis_cats, ODD_criterias=None)

    #todo make work for every dataset
    #%% Model performance : Plot Metric difference on gtbbox filtering
    #%% Study the Ground Truth Bounding Boxes : does their detection (matched) is correlated with their metadata ?
    # (bheight, occlusion ?)
    #from src.plot_utils import plot_gtbbox_matched_correlations
    #threshold = 0.9
    #features_bbox = ['height', "occlusion_rate", "aspect_ratio_is_typical"]
    #plot_gtbbox_matched_correlations(model_names, dataset, features_bbox, threshold, gtbbox_filtering)

    #todo make work for every model, change dataset tuple as input
    #%% Study a particular image with one of the model ==============================

    if plot_image:
        # Plot an image in particular
        frame_idx = 0
        model_name = model_names[0]
        gtbbox_filtering = gtbbox_filtering_all["Overall"]
        plot_image_with_detections(dataset_tuple, dataset_name, model_name, thresholds, gtbbox_filtering, i=frame_idx)

if __name__ == "__main__":

    # Parameters
    dataset_name = "coco_Fudan"
    max_samples = 200
    root = "/home/raphael/work/datasets/PennFudanPed"
    coco_json_path = "/home/raphael/work/datasets/PennFudanPed/coco.json"
    model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]

    # Launches the demo code
    run_demo_pedestrian_detection(root, dataset_name, max_samples, model_names, coco_json_path=coco_json_path,
                                  gtbbox_analysis=True)


#%% params of input dataset
# Which models to study
#
#dataset_names = ["motsynth", "twincity", "EuroCityPerson"]
#max_samples = [600, 50, 30]
# Dataset






