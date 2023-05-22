from src.utils import target_2_json, target_2_torch
import glob
import matplotlib.image as mpimg
import json
import numpy as np
import torch
import os.path as osp
import pandas as pd
import os
from .processing import DatasetProcessing
from .twincity_preprocessing2 import get_twincity_boxes, find_duplicate_indices, target_2_torch, creation_date
from configs_path import ROOT_DIR

class TwincityProcessing(DatasetProcessing):
    """
    Class that handles the preprocessing of (extracted) ECP Dataset in order to get a standardized dataset format.
    """

    def __init__(self, root, max_samples_per_sequence=10):
        self.dataset_name = "twincity"
        super().__init__(root, max_samples_per_sequence)
        os.makedirs(self.saves_dir, exist_ok=True)

    def get_sequence_dict(self):
        folders = glob.glob(osp.join(self.root, "*"))
        sequence_dict = {x.split("/")[-1]: (osp.join(x, "png"), osp.join(x, "labels")) for x in folders}
        return sequence_dict

    def preprocess_sequence(self, sequence_id, img_sequence_dir, annot_sequence_dir):

        # Check files
        metadata_path = glob.glob(osp.join(annot_sequence_dir, "Metadata*"))[0]
        images_path_list = glob.glob(osp.join(img_sequence_dir, "*.png"))

        with open(metadata_path) as file:
            metadata = json.load(file)

        ordering = np.argsort([creation_date(x) for x in images_path_list])
        img_annot_path_list = np.array(images_path_list)[ordering[::2]]
        img_rgb_path_list = np.array(images_path_list)[ordering[1::2]]

        # Here perform annotation complement for Twincity
        os.makedirs(annot_sequence_dir, exist_ok=True)
        for i, img_annot_path in enumerate(img_rgb_path_list):

            if i > self.max_samples_per_sequence:
                break

            image_id = img_rgb_path_list[i].split("/")[-1].split(".png")[0]
            path_annot_img = osp.join(annot_sequence_dir, f"{image_id}.json")

            if not os.path.exists(path_annot_img):

                print(f"Perform annotation transformation for image {image_id}")

                img_semantic_seg = mpimg.imread(img_annot_path_list[i])
                bboxes, df = get_twincity_boxes(img_semantic_seg, metadata)

                img_annots = []
                for j, _ in enumerate(df.iterrows()):
                    annot = {"image_id": image_id, "id": image_id + "_" + str(j), "sequence_id": sequence_id}
                    annot.update(df.iloc[j].to_dict())
                    annot.update({"x0": float(bboxes[j].numpy()[0]), "x1": float(bboxes[j].numpy()[2]), "y0":
                    float(bboxes[j].numpy()[1]), "y1": float(bboxes[j].numpy()[3])})
                    img_annots.append(annot)

                with open(path_annot_img, 'w') as f:
                    json.dump(img_annots, f)

        infos = metadata
        infos["sequence_id"] = annot_sequence_dir.split("/")[-2]
        new_images = []
        new_annots = []

        # For all images in the sequence
        for i, img_annot_path in enumerate(img_annot_path_list):

            if i>self.max_samples_per_sequence:
                break

            image_id = img_rgb_path_list[i].split("/")[-1].split(".png")[0]
            path_annot_img = osp.join(annot_sequence_dir, f"{image_id}.json")

            # Frame
            img = {"id": image_id, "pitch": metadata["cameraRotation"]["pitch"],
                   "hour": metadata["hour"], "is_night": metadata["hour"] > 21 or metadata[
                "hour"] < 6, "num_vehicles": metadata["vehiclesNb"], "weather": metadata["weather"],
                #   "file_name": img_rgb_path_list[i].split("v5/")[1], #todo make this more robust
                   "file_name": osp.join(img_sequence_dir.split("/")[-2],img_sequence_dir.split("/")[-1], img_rgb_path_list[i].split("/")[-1]), #todo make this more robust
                   "sequence_id": sequence_id}
            new_images.append(img)

            # Process the annotation
            #img_annot = mpimg.imread(img_annot_path)
            #bboxes, df = get_twincity_boxes(img_annot, metadata)
            #for j, row in enumerate(df.iterrows()):
            #    annot = {"image_id": image_id, "id": image_id+"_"+str(j), "sequence_id": sequence_id}
            #    annot.update(df.iloc[j].to_dict())
            #    annot.update({"x0": bboxes[j].numpy()[0], "x1": bboxes[j].numpy()[2], "y0": bboxes[j].numpy()[1], "y1": bboxes[j].numpy()[3]})
            with open(path_annot_img) as jsonFile:
                annot = json.load(jsonFile)
            new_annots += (annot)

        return infos, new_images, new_annots



