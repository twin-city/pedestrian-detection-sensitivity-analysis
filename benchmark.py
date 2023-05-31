import os

from src.demos.demo_benchmark_analysis import run_demo_pedestrian_detection
import os.path as osp

if __name__ == "__main__":
    """
    Launch on all datasets and models yet implemented.
    """

    model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]
    DATASET_DIR = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/"
    OUTPUT_DIR = "benchmarkv5"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    force_recompute = False
    do_dataset_analysis = False
    do_frame_analysis = True
    do_gtbbox_analysis = True
    do_plot_image = True
    do_show = False

    benchmark_params = [
        {"dataset_name": "ecp_small", "max_samples": 30},
        {"dataset_name": "motsynth_small", "max_samples": 30},
        #{"dataset_name": "PennFudanPed", "max_samples": 200},
        {"dataset_name": "Twincity-Unreal-v5", "max_samples": 30},
    ]

    # Compute the descriptive markdown table
    from src.dataset.dataset_factory import DatasetFactory
    for param in benchmark_params:
        dataset_name, max_samples = param.values()
        print(dataset_name, max_samples)
        root = osp.join(DATASET_DIR, dataset_name)
        dataset = DatasetFactory.get_dataset(dataset_name, max_samples, root, force_recompute=force_recompute)
        output_path = osp.join(OUTPUT_DIR)
        dataset.create_markdown_description_table(output_path)

    # Run the demo
    for param in benchmark_params:
        dataset_name, max_samples = param.values()
        root = osp.join(DATASET_DIR, dataset_name)
        run_demo_pedestrian_detection(root, dataset_name, max_samples, model_names,
                                  dataset_analysis=do_dataset_analysis, frame_analysis=do_frame_analysis,
                                  gtbbox_analysis=do_gtbbox_analysis,
                                  plot_image=do_plot_image, output_dir=OUTPUT_DIR, show=do_show)