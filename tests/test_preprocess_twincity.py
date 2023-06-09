import unittest
import os
import os.path as osp
from src.preprocessing.twincity_preprocessing_utils import find_duplicate_indices
import cv2
import numpy as np
import matplotlib.image as mpimg
import json
import torch
from src.preprocessing.twincity_preprocessing_utils import code_rgba_str_2_code_rgba_float
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks

from src.preprocessing.twincity_preprocessing_utils import postprocess_mask

TEST_PLOT_SHOW = True



def get_scenario_day_rain_plongee():
    #todo set in tests
    metadata_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_treeleft/day_rainy_50_plongee/labels/Metadata2.json"
    rgb_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_treeleft/day_rainy_50_plongee/png/Snapshot-2023-06-01_150631-50595184.png"
    png_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_treeleft/day_rainy_50_plongee/png/Snapshot-2023-06-01_150630-67107764.png"
    return metadata_path, rgb_path, png_path

def get_scenario_day_sun_frontal():
    #todo set in tests
    metadata_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_bigbox/day_sun_50_frontal/labels/Metadata1.json"
    rgb_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_bigbox/day_sun_50_frontal/png/Snapshot-2023-06-05_120133-6876815.png"
    png_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_bigbox/day_sun_50_frontal/png/Snapshot-2023-06-05_120132-71650432.png"
    return metadata_path, rgb_path, png_path

def get_scenario_day_sun_plongee():
    metadata_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_missingperson/day_sun_50_plongee/labels/Metadata2.json"
    rgb_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_missingperson/day_sun_50_plongee/png/Snapshot-2023-06-01_094123-34011800.png"
    png_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_missingperson/day_sun_50_plongee/png/Snapshot-2023-06-01_094123-8948311.png"
    return metadata_path, rgb_path, png_path

def get_scenario_night_sun_plongeeloin():
    metadata_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_plongeeloin/night_sun_50_plongeeloin/labels/Metadata4.json"
    rgb_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_plongeeloin/night_sun_50_plongeeloin/png/Snapshot-2023-06-01_151316-62238492.png"
    png_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/Twincity-Unreal-v8bis_test_plongeeloin/night_sun_50_plongeeloin/png/Snapshot-2023-06-01_151316-6470282.png"
    return metadata_path, rgb_path, png_path



class testPreprocessTwincity(unittest.TestCase):



    def test_plongeeloin(self):

        # Forth case : plongee loin

        metadata_path, rgb_path, png_path = get_scenario_night_sun_plongeeloin()

        img_semantic_seg = mpimg.imread(png_path)
        img_rgb = mpimg.imread(rgb_path)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Third Case : Similar to ground blue()

        # Presently What it does
        from src.preprocessing.twincity_preprocessing_utils import get_twincity_boxes
        bboxes, df = get_twincity_boxes(img_semantic_seg, metadata)

        idx_ignored = list(df[df["ignore_region"]==1].index)
        idx_kept = list(df[df["ignore_region"]==0].index)

        from src.plot_utils import add_bboxes_to_img_ax
        fig, ax = plt.subplots()
        ax.imshow(img_rgb)
        add_bboxes_to_img_ax(ax, bboxes[idx_ignored],  c=(1, 1, 0))
        add_bboxes_to_img_ax(ax, bboxes[idx_kept],  c=(0, 1, 0))
        plt.show()


        for i in range(1, 6):

            code = code_rgba_str_2_code_rgba_float(metadata["peds"][f"{i}"])
            #code = (0.0, 1.0, 0.749918, 1.0)
            mask = torch.tensor(((img_semantic_seg - code) ** 2).sum(axis=2) < 1e-4)
            new_mask, mask_info = postprocess_mask(mask)

            if TEST_PLOT_SHOW:
                fig, ax = plt.subplots(figsize=(20,10))
                ax.imshow(img_rgb, alpha=0.6)
                ax.imshow(mask, alpha=0.7)
                plt.show()
                plt.close()


                fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                plt.imshow(new_mask.numpy())
                plt.show()
                plt.close()


    def test_remove_tree_on_left(self):

        # Third case : Tree on left on Day/Rain/Plongee

        metadata_path, rgb_path, png_path = get_scenario_day_rain_plongee()

        img_semantic_seg = mpimg.imread(png_path)
        img_rgb = mpimg.imread(rgb_path)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Third Case : Similar to ground blue()

        # code = code_rgba_str_2_code_rgba_float(metadata["peds"]["10"])
        code = (0.0, 1.0, 0.749918, 1.0)
        mask = torch.tensor(((img_semantic_seg - code) ** 2).sum(axis=2) < 1e-4)
        new_mask, _ = postprocess_mask(mask)

        if TEST_PLOT_SHOW:
            fig, ax = plt.subplots(figsize=(20,10))
            ax.imshow(img_rgb, alpha=0.6)
            ax.imshow(mask, alpha=0.7)
            plt.show()
            plt.close()


            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            plt.imshow(new_mask.numpy())
            plt.show()
            plt.close()



    def test_remove_fence_and_ground_not_clear(self):
        """
        Remove small blobs : in order to catch when same color with trees, ground ...
        :return:
        """

        # Day / Sun / Frontal
        metadata_path, rgb_path, png_path = get_scenario_day_sun_frontal()
        img_semantic_seg = mpimg.imread(png_path)
        img_rgb = mpimg.imread(rgb_path)

        # First case : similar to plants
        # Second Case : Similar to ground blue()
        codes  = (0.0, 1.0, 0.749918, 1.0),(0.070666, 0.366667, 0.906666, 1.0)

        for code in codes:
            mask = torch.tensor(((img_semantic_seg - code) ** 2).sum(axis=2) < 1e-5)
            # Postprocess
            new_mask, _ = postprocess_mask(mask)

            if TEST_PLOT_SHOW:
                # Plot
                fig, ax = plt.subplots(figsize=(20,10))
                ax.imshow(img_rgb, alpha=0.6)
                ax.imshow(mask, alpha=0.7)
                plt.show()

                fig, ax = plt.subplots(1,1, figsize=(20,10))
                plt.imshow(new_mask)
                plt.show()

        """
        # Plot each contour on the blank image
        contour_image = np.zeros_like(img_semantic_seg)
        for contour in contours:
            cv2.drawContours(contour_image, [contour], -1, (255), thickness=2)

        # Plot the image with the contours
        plt.imshow(contour_image.astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()

        plt.hist(contour_areas, bins=100)
        plt.show()
        """

        """ Mean
        img_semantic_seg[585:600, 1020:1065].mean(axis=(0,1))
        --> array([0.07058813, 0.36470354, 0.9058836 , 1.        ], dtype=float32)

        img_semantic_seg[585:600, 1020:1065].std(axis=(0,1))
        --> array([1.0430813e-07, 2.3543835e-06, 1.2516975e-06, 0.0000000e+00],
      dtype=float32)


      img_semantic_seg[563:570, 969:975].mean(axis=(0,1))
      --> array([0.07058827, 0.36470598, 0.90588224, 1.        ], dtype=float32)

      std
      --> array([2.9802322e-08, 8.9406967e-08, 1.1920929e-07, 0.0000000e+00],
      dtype=float32)
      
          plt.imshow(img_semantic_seg[563:570, 969:975])
        plt.show()
        plt.close()
      
        """



    def test_bigbox_twincity(self):
        """
        I don't remember excatly here ...
        :return:
        """
        # Day / Sun / Frontal
        metadata_path, rgb_path, png_path = get_scenario_day_sun_frontal()
        img_semantic_seg = mpimg.imread(png_path)
        img_rgb = mpimg.imread(rgb_path)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Presently What it does
        from src.preprocessing.twincity_preprocessing_utils import get_twincity_boxes
        bboxes, df = get_twincity_boxes(img_semantic_seg, metadata)

        idx_ignored = list(df[df["ignore_region"]==1].index)
        idx_kept = list(df[df["ignore_region"]==0].index)

        from src.plot_utils import add_bboxes_to_img_ax
        fig, ax = plt.subplots()
        ax.imshow(img_rgb)
        add_bboxes_to_img_ax(ax, bboxes[idx_ignored],  c=(1, 1, 0))
        add_bboxes_to_img_ax(ax, bboxes[idx_kept],  c=(0, 1, 0))
        plt.show()


        # %% Draw masks from torch --> Bug Exploration
        """
        import os
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import torchvision.transforms.functional as F
        from torchvision.io import read_image
        img = read_image(rgb_path)

        drawn_masks = []
        masks = []
        for ped_idx in range(len(metadata["peds"])):
            ped_id = list(metadata["peds"])[ped_idx]
            if ped_id == "10":
                print("coucou")
            code = code_rgba_str_2_code_rgba_float(metadata["peds"][ped_id])
            mask = torch.tensor(((img_semantic_seg - code) ** 2).sum(axis=2) < 1e-4)
            masks.append(mask)
            drawn_masks.append(draw_segmentation_masks(img[:3] , mask.unsqueeze(0), alpha=0.8, colors="blue"))

        for i, img in enumerate(drawn_masks):
            img = img.detach()
            img = F.to_pil_image(img)
            fig, ax = plt.subplots()
            ax.imshow(np.asarray(img))
            plt.title(f'Ped with idx {i} and id {list(metadata["peds"])[i]}')
            mask = masks[i]
            ax.text(int(mask.sum(axis=0).argmax()),
                    int(mask.sum(axis=1).argmax()), i, size=50)
            plt.show()
            plt.close()
        """


    def test_bugexplore_missingperson_twincity(self):

        metadata_path, rgb_path, png_path = get_scenario_day_sun_plongee()

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Assert the Transparency is 1
        alphas = [{key: code_rgba_str_2_code_rgba_float(val)[3]} for key, val in metadata["peds"].items()]
        alpha_not_one = [x for x in alphas if list(x.values())[0] != 1]
        # assert len(alpha_not_one) == 0 #todo trigger a warning instead ?

        """ Conclusion : This one had an alpha != 1 !!!!!!
        #%% Check the one not detected
        plt.imshow(img_semantic_seg[815:-20, 700:720])
        plt.show()
        code = img_semantic_seg[815:-20, 700:720].mean(axis=(0,1)) # array([0.8745116 , 0.54117453, 0.21960492, 1.        ], dtype=float32)
        code_alpha_one = np.array((code[:3] + (1.0,)))

        import numpy as np
        dists = [((np.array(code_rgba_str_2_code_rgba_float(x))-code_alpha_one)**2).sum() for x in metadata["peds"].values()]
        print(list(metadata["peds"].keys())[np.argmin(dists)]) # 10
        """



### Simple Tests

    def test_doublons_pedestrian_color(self):

        DATASET_DIR = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/"
        TWINCITY_DIR = osp.join(DATASET_DIR, "Twincity-Unreal-v8")

        for scenario_folder in os.listdir(TWINCITY_DIR):
            print(scenario_folder)
            files = os.listdir(osp.join(TWINCITY_DIR, scenario_folder, "labels"))
            metadata_files = [x for x in files if "Metadata" in x]
            print(len(metadata_files))

            # There should be one metadata file
            self.assertEqual(len(metadata_files), 1)

            # Get doublons
            with open(osp.join(TWINCITY_DIR, scenario_folder, "labels",metadata_files[0])) as f:
                metadata = json.load(f)
            duplicates = find_duplicate_indices(metadata["peds"])
            if len(duplicates) > 0:
                print("Warning, doublons in metadata")
            self.assertEqual(len(duplicates), 0)



    def test_preprocess_twincity(self):
        print("coucou")

        DATASET_DIR = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/"

        force_recompute = True

        benchmark_params = [
            {"dataset_name": "Twincity-Unreal-v8bis_test_bigbox", "max_samples": 1},
        ]

        # Compute the descriptive markdown table
        from src.dataset.dataset_factory import DatasetFactory
        for param in benchmark_params:
            dataset_name, max_samples = param.values()
            print(dataset_name, max_samples)
            root = osp.join(DATASET_DIR, dataset_name)
            dataset = DatasetFactory.get_dataset(dataset_name, max_samples, root, force_recompute=force_recompute)
