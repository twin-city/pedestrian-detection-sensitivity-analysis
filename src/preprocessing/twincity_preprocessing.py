import glob
import matplotlib.image as mpimg
import json
import numpy as np
import os.path as osp
from .processing import DatasetProcessing
from src.preprocessing.twincity_preprocessing_utils import get_twincity_boxes, creation_date
import os
import shutil

class TwincityProcessing(DatasetProcessing):
    """
    Class that handles the preprocessing of (extracted) ECP Dataset in order to get a standardized dataset format.
    """

    def __init__(self, root, max_samples_per_sequence=10, task="pedestrian_detection"):

        num_v = root.split("/")[-1].split("-v")[-1] #todo not working with lyingped
        self.dataset_name = f"twincity_v{num_v}"
        super().__init__(root, max_samples_per_sequence, task)
        os.makedirs(self.saves_dir, exist_ok=True)

    def get_sequence_dict(self):
        folders = glob.glob(osp.join(self.root, "*"))
        sequence_dict = {x.split("/")[-1]: (osp.join(x, "png"), osp.join(x, "labels")) for x in folders}
        return sequence_dict

    def preprocess_sequence(self, sequence_id, img_sequence_dir, annot_sequence_dir, force_recompute=False):

        # Check if need to modify structure
        sequence_dir = "/".join(img_sequence_dir.split("/")[:-1])
        png_folder = img_sequence_dir
        labels_folder = annot_sequence_dir

        # Step 1: Create the 'png' and 'labels' folders if they don't exist, and move the files
        if not os.path.exists(png_folder) and not os.path.exists(labels_folder):
            os.makedirs(png_folder)
            os.makedirs(labels_folder)

            # Step 2: Get a list of all .png and .json files in the sequence_dir
            files = os.listdir(sequence_dir)
            png_files = [file for file in files if file.endswith('.png')]
            json_files = [file for file in files if file.endswith('.json')]

            # Step 3: Move the .png files to the 'png' folder
            for file in png_files:
                src = os.path.join(sequence_dir, file)
                dst = os.path.join(png_folder, file)
                shutil.move(src, dst)

            # Move the .json files to the 'labels' folder
            for file in json_files:
                src = os.path.join(sequence_dir, file)
                dst = os.path.join(labels_folder, file)
                shutil.move(src, dst)

        # Check files
        metadata_path = glob.glob(osp.join(annot_sequence_dir, "Metadata*"))[0]
        images_path_list = glob.glob(osp.join(img_sequence_dir, "*.png"))

        with open(metadata_path) as file:
            metadata = json.load(file)

        is_night = metadata["hour"] > 21 or metadata["hour"] < 6
        weather = metadata["weather"]
        pitch = metadata["cameraRotation"]["pitch"]

        print(is_night, weather, pitch)
        if is_night == True and weather == "Clear Sky" and np.abs((pitch - (-30.0))) < 1e-5:
            print("coucou")


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

            if not os.path.exists(path_annot_img) or force_recompute:

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

        #todo sequence to rename

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


    def preprocess_specific(self, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata):


        df_frame_metadata['weather_original'] = df_frame_metadata['weather']


        # Weather renaming
        weather_renaming = {
            # Dry Weather
            #"EXTRASUNNY": "extrasunny",
            #"CLEAR": "clear",
            "Partially cloudy": "clouds",
            #"OVERCAST": "overcast",
            # Rainy Weather
            #"RAIN": "rainy",
            #"THUNDER": "thunder",
            # Reduced Visibility
            "Clear Sky": "clear",
            "Rain": "rain",
            "Snow": "snow",
        }

        # Assuming you have a DataFrame named 'df' with a column named 'weather'
        df_frame_metadata['weather'] = df_frame_metadata['weather_original'].replace(weather_renaming)

        # Weather categories according to homegenized weather naming
        df_frame_metadata = self.add_weather_cats(df_frame_metadata)


        return df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata
