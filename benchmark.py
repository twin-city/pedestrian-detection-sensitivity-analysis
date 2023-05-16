from src.demos.demo_benchmark_analysis import run_demo_pedestrian_detection

if __name__ == "__main__":
    """
    Launch on all datasets and models yet implemented.
    """

    # Parameters coco-Fudan
    dataset_name = "coco_Fudan"
    max_samples = 200
    model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]
    root = "/home/raphael/work/datasets/PennFudanPed"
    coco_json_path = "/home/raphael/work/datasets/PennFudanPed/coco.json"
    run_demo_pedestrian_detection(root, dataset_name, max_samples, model_names, coco_json_path=coco_json_path,
                                  dataset_analysis=False, frame_analysis=True, gtbbox_analysis=True,
                                  plot_image=False, output_dir="results/benchmark", show=False)


    # Parameters MoTSynth
    dataset_name = "motsynth"
    root = "/home/raphael/work/datasets/MOTSynth"
    max_samples = 600
    coco_json_path = None
    run_demo_pedestrian_detection(root, dataset_name, max_samples, model_names, coco_json_path=coco_json_path,
                                  dataset_analysis=False, frame_analysis=True, gtbbox_analysis=True,
                                  plot_image=False, output_dir="results/benchmark", show=False)

    # Parameters ECP
    dataset_name = "EuroCityPerson"
    root = "/media/raphael/Projects/datasets/EuroCityPerson"
    max_samples = 30
    model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]
    coco_json_path = None
    run_demo_pedestrian_detection(root, dataset_name, max_samples, model_names, coco_json_path=coco_json_path,
                                  dataset_analysis=False, frame_analysis=True, gtbbox_analysis=True,
                                  plot_image=False, output_dir="results/benchmark", show=False)


    # Parameters Twincity
    dataset_name = "twincity"
    root = "/home/raphael/work/datasets/twincity-Unreal/v5"
    max_samples = 50
    model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]
    coco_json_path = None
    run_demo_pedestrian_detection(root, dataset_name, max_samples, model_names, coco_json_path=coco_json_path,
                                  dataset_analysis=False, frame_analysis=True, gtbbox_analysis=True,
                                  plot_image=False, output_dir="results/benchmark", show=False)

