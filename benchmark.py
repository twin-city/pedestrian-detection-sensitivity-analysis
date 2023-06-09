import json
import os
from src.demos.demo_benchmark_analysis import run_demo_detection
import os.path as osp
import logging
from configs_path import DATASET_DIR

def dump_params(OUTPUT_DIR, param_dict):
    i = 0
    os.makedirs(osp.join(OUTPUT_DIR, "params"), exist_ok=True)
    while os.path.exists(osp.join(OUTPUT_DIR, "params",f"benchmark_params_{i}.json")):
        i += 1
    with open(osp.join(OUTPUT_DIR, "params",f"benchmark_params_{i}.json"), "w") as f:
        json.dump(param_dict, f)


if __name__ == "__main__":
    """
    Launch on all datasets and models yet implemented.
    """

    ## DIRS
    OUTPUT_DIR = "results/benchv14_final?"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ## LOGGER
    # Create and configure the logger
    logger = logging.getLogger('benchmark_logger')
    logger.setLevel(logging.DEBUG)
    # Create a file handler and set the logging level
    file_handler = logging.FileHandler(osp.join(OUTPUT_DIR, 'benchmark.log'))
    file_handler.setLevel(logging.DEBUG)
    # Create a formatter and set it for the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)


    model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]

    force_recompute = False
    do_dataset_analysis = False
    do_frame_analysis = True
    do_gtbbox_analysis = True
    do_plot_image = False
    do_show = False

    run_params = {
        "force_recompute": force_recompute,
        "do_dataset_analysis": do_dataset_analysis,
        "do_frame_analysis": do_frame_analysis,
        "do_gtbbox_analysis" : do_gtbbox_analysis,
        "do_plot_image": do_plot_image,
        "do_show": do_show,
    }

    datasets_params = [
        {"dataset_name": "Twincity-Unreal-v9", "max_samples": 20},
        {"dataset_name": "ecp_small", "max_samples": 30},
        {"dataset_name": "motsynth_small", "max_samples": 30},
        #{"dataset_name": "PennFudanPed", "max_samples": 200},
    ]

    parameters = {
        "datasets_params": datasets_params,
        "run_params": run_params,
    }

    dump_params(OUTPUT_DIR, parameters)
    logger.info(parameters)

    logger.info('Computing descriptive markdown table for all datasets.')
    # Compute the descriptive markdown table
    from src.dataset.dataset_factory import DatasetFactory
    for param in datasets_params:
        dataset_name, max_samples = param.values()
        print(dataset_name, max_samples)
        root = osp.join(DATASET_DIR, dataset_name)
        dataset = DatasetFactory.get_dataset(dataset_name, max_samples, root, force_recompute=force_recompute)
        output_path = osp.join(OUTPUT_DIR)
        dataset.create_markdown_description_table(output_path)


    logger.info('Running the demo.')
    # Run the demo
    for param in datasets_params:
        dataset_name, max_samples = param.values()
        root = osp.join(DATASET_DIR, dataset_name)
        logger.info(f'Running the demo for {dataset_name}.')
        run_demo_detection(root, dataset_name, max_samples, model_names,
                                  dataset_analysis=do_dataset_analysis, frame_analysis=do_frame_analysis,
                                  gtbbox_analysis=do_gtbbox_analysis,
                                  plot_image=do_plot_image, output_dir=OUTPUT_DIR, show=do_show)

    logger.info('Closing.')
    # Close the file handler
    file_handler.close()