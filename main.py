from src.demos.demo_benchmark_analysis import run_demo_detection
import argparse

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Launch Pedestrian Detection Sensitivity Analysis on a given dataset')

    # Add dataset arguments
    parser.add_argument('--dataset', '-d', type=str, help='Dataset name')
    parser.add_argument('--root', '-r', type=str, help='Dataset root path')
    parser.add_argument('--max_samples',  type=str, default=100,
                        help='Maximum number offrames to consider (sampled uniformly in the dataset)')
    parser.add_argument('--coco_json_path', '-c', type=str,
                        help='In case of coco dataset, path of the .json annotations.', default=None)

    # Add model arguments
    parser.add_argument('--models', '-m', type=list, help='List of models to consider', default=["faster-rcnn_cityscapes", "mask-rcnn_coco"])

    # What analysis to perform
    parser.add_argument('--frame_analysis', '-frame', help='Wether to perform frame analysis', action='store_true')
    parser.add_argument('--gtbbox_analysis', '-gtbbox', help='Weather to perform gtbbox analysis', action='store_true')
    parser.add_argument('--plot_image', '-p', help='Weather to show detection results', action='store_true')

    parser.add_argument('--show', help='Weather to show results in plot windows', action='store_true')
    parser.add_argument('--output', '-o', type=str, help='Output dir path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose mode')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments
    root = args.root
    output = args.output
    show = args.show
    dataset_name = args.dataset
    max_samples = args.max_samples
    model_names = args.models
    coco_json_path = args.coco_json_path
    frame_analysis = args.frame_analysis
    gtbbox_analysis = args.gtbbox_analysis
    plot_image = args.plot_image
    verbose = args.verbose

    # Display the arguments
    if verbose:
        print('Verbose mode enabled')
    if coco_json_path:
        print(f'Input coco file path: {coco_json_path}')

    run_demo_detection(root, dataset_name, max_samples, model_names, coco_json_path=coco_json_path,
                                  gtbbox_analysis=gtbbox_analysis, frame_analysis=frame_analysis, output_dir=output, show=show,
                                  plot_image=plot_image)


if __name__ == '__main__':
    main()


