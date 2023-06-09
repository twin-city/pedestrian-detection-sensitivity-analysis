import json
import numpy as np
import os.path as osp
import os
from src.plot_utils import xywh2xyxy
from .processing import DatasetProcessing
from .coco_processing import subset_dict

class MotsynthProcessing(DatasetProcessing):
    """
    Here sequence are annotated (vs img annotation for ECP)
    """

    def __init__(self, root, max_samples_per_sequence=10, task="pedestrian_detection"):

        self.dataset_name = "motsynth"
        super().__init__(root, max_samples_per_sequence, task)

        self.delay = 3
        self.frames_dir = f"{root}/frames"
        self.annot_dir = f"{root}/coco annot"
        os.makedirs(self.saves_dir, exist_ok=True)


        #self.sequence_ids = sequence_ids
        self.sequence_ids = self.get_usable_sequence_ids()
        self.num_sequences = len(self.sequence_ids)

        #if max_samples is not None:
        #    self.max_num_sample_per_sequence = int(max_samples / self.num_sequences)
        #else:
        # self.max_samples_per_sequence = 1000  # todo max to 1000 arbirterary
        # self.max_samples_per_sequence = self.max_samples // len(self.get_usable_sequence_ids())

        # Additional info
        self.RESOLUTION = (1920, 1080)
        self.NUM_KEYPOINTS = 22

        assert task in ["pedestrian_detection"]

    # To load from multiple sequences

    def get_usable_sequence_ids(self):
        """
        Get usable sequence ids.
        :return:
        """
        #if self.sequence_ids is None:
        exclude_ids_frames = {"060", "081", "026", "132", "136", "102", "099", "174", "140"}
        sequence_ids_frames = set(list(np.sort(os.listdir(self.frames_dir))))
        sequence_ids_json = set([i.replace(".json", "") for i in
                              os.listdir(self.annot_dir)]) - exclude_ids_frames
        sequence_ids = list(np.sort(list(set.intersection(sequence_ids_frames, sequence_ids_json))))

            #if self.max_samples is not None:
            #    if self.max_samples < len(sequence_ids):
            #        sequence_ids = np.random.choice(sequence_ids, self.max_samples, replace=False)
        #else:
        #    sequence_ids = self.sequence_ids

        return sequence_ids

    def get_sequence_dict(self):

        sequence_dict = {}
        for sequence_id in self.sequence_ids:
            img_sequence_dir = osp.join(self.frames_dir, sequence_id)
            annot_sequence_dir = osp.join(self.annot_dir)
            sequence_dict[sequence_id] = (img_sequence_dir, annot_sequence_dir)
        return sequence_dict


    #todo here should be shared with coco style datasets. Can be factorized to a large extent
    def preprocess_sequence(self, sequence_id, img_sequence_dir, annot_sequence_dir, force_recompute=False):

        # Open annotation file
        json_path = f"{self.annot_dir}/{sequence_id}.json"
        with open(json_path) as jsonFile:
            annot_motsynth = json.load(jsonFile)

        for img in annot_motsynth["images"]:
            img["id"] += self.delay

        # If there is subsampling
        #if self.max_samples_per_sequence < len(annot_motsynth["images"]):
        #    images_list = annot_motsynth["images"][::len(annot_motsynth["images"]) // self.max_samples_per_sequence]
        #    annot_list = [x for x in annot_motsynth["annotations"] if x["image_id"] in [i["id"] for i in images_list]]
        #else:
        images_list = annot_motsynth["images"]
        annot_list = annot_motsynth["annotations"]


        frame_keys_to_keep = ['file_name', 'id', 'frame_n', 'cam_world_pos', 'cam_world_rot', 'height', 'width'] #'ignore_mask' #todo renaming ?
        annot_keys_to_keep = ['id', 'image_id', 'category_id', 'area', 'bbox', 'iscrowd',
                              'num_keypoints', 'ped_id', 'model_id', 'attributes', 'is_blurred', 'keypoints'] # , 'keypoints_3d' 'segmentation',


        new_images, new_annots = [], []
        for image in images_list:
            subset_image = subset_dict(image, frame_keys_to_keep)
            subset_image["sequence_id"] = sequence_id
            new_images.append(subset_image)

        for annot in annot_list:
            subset_annot = subset_dict(annot, annot_keys_to_keep)
            subset_annot["sequence_id"] = sequence_id
            subset_annot["bbox"] = xywh2xyxy(subset_annot["bbox"]) # because bbox format can be different
            new_annots.append(subset_annot)

        infos = annot_motsynth["info"]
        infos["sequence_id"] = sequence_id

        #todo filter when too many annots because of delay
        image_id_annot = set(np.unique([img["image_id"] for img in new_annots]))
        image_id_image = set([img["id"] for img in new_images])
        intersection_id = set.intersection(image_id_annot, image_id_image)
        new_annots = [annot for annot in new_annots if annot["image_id"] in intersection_id]
        new_images = [img for img in new_images if img["id"] in intersection_id]

        return infos, new_images, new_annots


    def preprocess_specific(self, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata):
        # Add sequence information
        sequence_columns = ['is_night', "weather"]
        for sequence_id in df_sequence_metadata.index:
            df_frame_metadata.loc[df_frame_metadata["sequence_id"] == sequence_id, sequence_columns] = df_sequence_metadata.loc[sequence_id, sequence_columns].values


        weather_renaming = {
            # Dry Weather
            "EXTRASUNNY": "sunny",
            "CLEAR": "clear",
            "CLOUDS": "clouds",
            "OVERCAST": "overcast",
            # Rainy Weather
            "RAIN": "rain",
            "THUNDER": "thunder",
            # Reduced Visibility
            "SMOG": "smog",
            "FOGGY": "foggy",
            "BLIZZARD": "snow",
        }

        df_frame_metadata["weather_original"] = df_frame_metadata["weather"]
        df_frame_metadata["weather"] = df_frame_metadata["weather"].replace(weather_renaming)

        """
        We want : 
        - Dry : CLEAR, OVERCAST, EXTRASUNNY, CLOUDS
        - Rain : RAIN, THUNDER
        - Reduced Visibility : SMOG, FOGGY, BLIZZARD
        """

        # Weather categories according to homegenized weather naming
        df_frame_metadata = self.add_weather_cats(df_frame_metadata)

        # Occlusions
        keypoints_label_names = [f"o_{i}" for i in range(self.NUM_KEYPOINTS)]
        df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata[keypoints_label_names].apply(lambda x: (2 - x)).mean(axis=1)

        return df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

