import json
import numpy as np
import torch
import os.path as osp
import pandas as pd
import os
from src.utils import target_2_json, target_2_torch
from src.plot_utils import xywh2xyxy
from .processing import DatasetProcessing
from .coco_processing import subset_dict


def get_scalar_dict(dict, dict_frame_to_scalar):
    scalar_dict = {}
    for key, values in dict.items():
        if key in dict_frame_to_scalar.keys():
            for i, value in enumerate(values):
                scalar_dict[f"{dict_frame_to_scalar[key][i]}"] = value
        else:
            scalar_dict[key] = values
    return scalar_dict

class MotsynthProcessing(DatasetProcessing):

    def __init__(self, root, max_samples=200, sequence_ids=None):

        self.dataset_name = "motsynth"
        super().__init__(root, max_samples)

        self.delay = 3
        self.frames_dir = f"{root}/frames"
        self.annot_dir = f"{root}/coco annot"
        os.makedirs(self.saves_dir, exist_ok=True)


        self.sequence_ids = sequence_ids
        self.sequence_ids = self.get_usable_sequence_ids()
        self.num_sequences = len(self.sequence_ids)

        #if max_samples is not None:
        #    self.max_num_sample_per_sequence = int(max_samples / self.num_sequences)
        #else:
        self.max_num_sample_per_sequence = 1000  # todo max to 1000 arbirterary
        # self.max_samples_per_sequence = self.max_samples // len(self.get_usable_sequence_ids())

        # Additional info
        self.RESOLUTION = (1920, 1080)
        self.NUM_KEYPOINTS = 22



    def get_usable_sequence_ids(self):
        """
        Get usable sequence ids.
        :return:
        """
        if self.sequence_ids is None:
            exclude_ids_frames = set(["060", "081", "026", "132", "136", "102", "099", "174", "140"])
            sequence_ids_frames = set(np.sort(os.listdir(self.frames_dir)).tolist())
            sequence_ids_json = set([i.replace(".json", "") for i in
                                  os.listdir(self.annot_dir)]) - exclude_ids_frames
            sequence_ids = list(np.sort(list(set.intersection(sequence_ids_frames, sequence_ids_json))))

            if self.max_samples is not None:
                if self.max_samples < len(sequence_ids):
                    sequence_ids = np.random.choice(sequence_ids, self.max_samples, replace=False)
        else:
            sequence_ids = self.sequence_ids
        return sequence_ids


    def get_dataset(self, force_recompute=False):
        targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.load_or_preprocess(force_recompute)

        # Common post-processing
        df_gtbbox_metadata = self.format_gtbbox_metadata(df_gtbbox_metadata)
        df_frame_metadata = self.format_frame_metadata(df_frame_metadata, df_gtbbox_metadata)

        return self.root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata


    def load_or_preprocess(self, force_recompute=False):

        # Set the paths
        path_df_gtbbox_metadata = osp.join(self.saves_dir, f"df_gtbbox_{self.max_samples}.csv")
        path_df_frame_metadata = osp.join(self.saves_dir, f"df_frame_{self.max_samples}.csv")
        path_df_sequence_metadata = osp.join(self.saves_dir, f"df_sequence_{self.max_samples}.csv")
        path_target = osp.join(self.saves_dir, f"targets_{self.max_samples}.json")

        # Check if all files exist
        exist_all_paths = True
        file_paths = [path_df_gtbbox_metadata, path_df_frame_metadata, path_df_sequence_metadata, path_target]
        for file_path in file_paths:
            if not os.path.exists(file_path):
                exist_all_paths = False

        # If all files exist, load them
        if exist_all_paths and not force_recompute:
            print("Loading previously computed dataset")
            # Load it
            df_gtbbox_metadata = pd.read_csv(path_df_gtbbox_metadata, index_col=["image_id", "id"])
            df_frame_metadata = pd.read_csv(path_df_frame_metadata, index_col=["frame_id"])
            df_sequence_metadata = pd.read_csv(path_df_sequence_metadata, index_col=["sequence_id"])
            with open(path_target) as jsonFile:
                targets = target_2_torch(json.load(jsonFile))

        # Else, compute them
        else:
            targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.preprocess_motsynth()

            # Save dataframes
            df_gtbbox_metadata.to_csv(path_df_gtbbox_metadata)
            df_frame_metadata.to_csv(path_df_frame_metadata)
            df_sequence_metadata.to_csv(path_df_sequence_metadata)
            with open(path_target, 'w') as f:
                json.dump(target_2_json(targets), f)

        return targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata





    #todo here should be shared with coco style datasets
    def preprocess_motsynth_sequence(self, sequence_id="004"):

        # Open annotation file
        json_path = f"{self.annot_dir}/{sequence_id}.json"
        with open(json_path) as jsonFile:
            annot_motsynth = json.load(jsonFile)

        for img in annot_motsynth["images"]:
            img["id"] += self.delay


        # If there is subsampling
        if self.max_num_sample_per_sequence < len(annot_motsynth["images"]):
            images_list = annot_motsynth["images"][::len(annot_motsynth["images"]) // self.max_num_sample_per_sequence]
            annot_list = [x for x in annot_motsynth["annotations"] if x["image_id"] in [i["id"] for i in images_list]]
        else:
            images_list = annot_motsynth["images"]
            annot_list = annot_motsynth["annotations"]


        frame_keys_to_keep = ['file_name', 'id', 'frame_n', 'cam_world_pos', 'cam_world_rot', 'height', 'width'] #'ignore_mask' #todo renaming ?
        annot_keys_to_keep = ['id', 'image_id', 'category_id', 'area', 'bbox', 'iscrowd',
                              'num_keypoints', 'ped_id', 'model_id', 'attributes', 'is_blurred', 'keypoints'] # , 'keypoints_3d' 'segmentation',

        # image_types = {key: type(val) for key, val in images_list[0].items()} #todo later

        new_images, new_annots = [], []

        for image in images_list:
            subset_image = subset_dict(image, frame_keys_to_keep)
            subset_image["sequence_id"] = sequence_id
            new_images.append(subset_image)

        for annot in annot_list:
            subset_annot = subset_dict(annot, annot_keys_to_keep)
            subset_annot["sequence_id"] = sequence_id
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

        """
        
                df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = [pd.DataFrame()]*3
        frame_metadata = {}
        targets = {}
        img_path_list = []
        targets_metadata = {}
            # todo more info in image
            for i, name in enumerate(["yaw", "pitch", "roll"]):
                image[name] = image["cam_world_rot"][i]
            for i, name in enumerate(["x", "y", "z"]):
                image[name] = image["cam_world_pos"][i]
            for info_name in ["is_night", "seq_name", "weather", "is_moving", "cam_fov", 'fx', 'fy', 'cx', 'cy']:
                image[info_name] = annot_motsynth["info"][info_name]

            frame_id = image["id"]
            bboxes = [xywh2xyxy(x["bbox"]) for x in annot_motsynth["annotations"] if x["image_id"] == frame_id+delay]

            # BBOXES metadata
            annots = [x for x in annot_motsynth["annotations"] if x["image_id"] == frame_id+delay]
            height = [x["bbox"][3] for x in annots]
            width = [x["bbox"][2] for x in annots]
            aspect_ratio = [w/h for h,w in zip(height, width)]
            keypoints_label = [(np.array(annot["keypoints"])).reshape((self.NUM_KEYPOINTS, 3))[:,2] for annot in annots]
            keypoints_posx = [(np.array(annot["keypoints"])).reshape((self.NUM_KEYPOINTS, 3))[:,0] for annot in annots]
            keypoints_posy = [(np.array(annot["keypoints"])).reshape((self.NUM_KEYPOINTS, 3))[:, 1] for annot in annots]
            area = [annot["area"] for annot in annots]
            is_crowd = [annot["iscrowd"] for annot in annots]
            is_blurred = [annot["is_blurred"] for annot in annots]
            attributes = [annot["attributes"] for annot in annots]
            ped_id = [annot["ped_id"] for annot in annots]
            id = [annot["id"] for annot in annots]
            image_id = [annot["image_id"] for annot in annots]

            target_metadata = {
                "image_id": image_id,
                "id": id,
                "keypoints_label": keypoints_label,
                "keypoints_posx": keypoints_posx,
                "keypoints_posy": keypoints_posy,
                "area": area,
                "height": height,
                "width": width,
                "aspect_ratio": aspect_ratio,
                "is_crowd": is_crowd,
                "is_blurred": is_blurred,
                "attributes": attributes,
                "ped_id": ped_id,
            }

            targets_metadata[frame_id] = target_metadata

            # Target and labels
            target = [
                dict(
                    boxes=torch.tensor(
                        bboxes)
                    )]
            target[0]["labels"] = torch.tensor([0] * len(target[0]["boxes"]))

            # Keep only if at least 1 pedestrian
            if len(target[0]["boxes"]) > 0:
                targets[str(frame_id)] = target
                frame_metadata[frame_id] = annot_motsynth["info"]
                img_path_list.append(osp.join(self.root, image["file_name"]))
                # frame_metadata[frame_id] = (annot_ECP["tags"], [ann["tags"] for ann in annot_ECP["children"]])

                # Dataframes
                df_gtbbox_metadata_current = pd.DataFrame(target_metadata)
                attributes_names = [f"attributes_{i}" for i in range(11)]
                df_gtbbox_metadata_current[attributes_names] = attributes

                df_gtbbox_metadata = pd.concat([df_gtbbox_metadata, df_gtbbox_metadata_current], axis=0)

                keypoints_label_names = [f"keypoints_label_{i}" for i in range(self.NUM_KEYPOINTS)]
                keypoints_posx_names = [f"keypoints_posx_{i}" for i in range(self.NUM_KEYPOINTS)]
                keypoints_posy_names = [f"keypoints_posy_{i}" for i in range(self.NUM_KEYPOINTS)]


                df_gtbbox_metadata[keypoints_label_names] = df_gtbbox_metadata["keypoints_label"].apply(lambda x: pd.Series(x))
                df_gtbbox_metadata[keypoints_posx_names] = df_gtbbox_metadata["keypoints_posx"].apply(lambda x: pd.Series(x))
                df_gtbbox_metadata[keypoints_posy_names] = df_gtbbox_metadata["keypoints_posy"].apply(lambda x: pd.Series(x))


                frame_metadata_features = ['file_name', 'id', 'frame_n'] + \
                                          ["is_night", "seq_name", "weather"] + \
                                          ["is_moving", "cam_fov", 'fx', 'fy', 'cx', 'cy'] + \
                                          ["x", "y", "z", "yaw", "pitch", "roll"]
                df_frame_metadata = pd.concat([df_frame_metadata, pd.DataFrame({key:val for key,val in image.items() if key in frame_metadata_features}, index=[frame_id])], axis=0)
        frame_id_list = list(targets.keys())

        # Metadata at the sequence level
        df_sequence_metadata = pd.DataFrame(annot_motsynth["info"], index=[sequence_id])
        df_gtbbox_metadata["seq_name"] = df_frame_metadata["seq_name"].iloc[0]

        # Set the indexes
        # Operations on indexes
        df_gtbbox_metadata = df_gtbbox_metadata.reset_index()
        df_gtbbox_metadata["frame_id"] = df_gtbbox_metadata["image_id"] - self.delay
        df_gtbbox_metadata[["frame_id", "id"]] = df_gtbbox_metadata[["frame_id", "id"]].astype(str)

        df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])

        df_frame_metadata.index = df_frame_metadata.index.astype(str)
        df_frame_metadata.index.name = "frame_id"

        df_sequence_metadata.index.name = "sequence_id"
        df_sequence_metadata.index = df_sequence_metadata.index.astype(str)
        """







    def preprocess_motsynth(self):
        """
        Get a full coco style Dataset
        :return:
        """

        #targets_sequence_list, df_gtbbox_metadata_sequence_list, \
        #    df_frame_metadata_sequence_list, df_sequence_metadata_sequence_list = [], [], [], []

        infos, new_images, new_annots = [], [], []

        for sequence in self.sequence_ids:
            print(f"Checking sequence {sequence}")

            # try:
            #targets_sequence, df_gtbbox_metadata_sequence, df_frame_metadata_sequence, df_sequence_metadata_sequence = \
            #    self.preprocess_motsynth_sequence(sequence_id=sequence)

            infos_sequence, new_images_sequence, new_annots_sequence = self.preprocess_motsynth_sequence(sequence_id=sequence)
            new_images.append(new_images_sequence)
            new_annots.append(new_annots_sequence)
            infos.append(infos_sequence)

            # targets_sequence_list.append(targets_sequence)
            # df_gtbbox_metadata_sequence_list.append(df_gtbbox_metadata_sequence)
            # df_frame_metadata_sequence_list.append(df_frame_metadata_sequence)
            # df_sequence_metadata_sequence_list.append(df_sequence_metadata_sequence)

        def flatten_list(l):
            return [item for sublist in l for item in sublist]

        new_images = [item for sublist in new_images for item in sublist]
        new_annots = [item for sublist in new_annots for item in sublist]

        #%% Now how to transform to dataframe ? To a scalar that can be in a dataframe


        dict_frame_to_scalar = {
            "cam_world_rot": ("yaw", "pitch", "roll"),
            "cam_world_pos": ("x", "y", "z"),
        }

        dict_annot_to_scalar = {
            "attributes": [f"attribute_{i}" for i in range(11)],
            "bbox": ("x0", "y0", "x1", "y1"),
            "keypoints": flatten_list([[f"x_{i}", f"y_{i}", f"o_{i}"] for i in range(22)]),
        }


        df_frame_metadata = pd.DataFrame([get_scalar_dict(img, dict_frame_to_scalar) for img in new_images])
        df_gtbbox_metadata = pd.DataFrame([get_scalar_dict(annot, dict_annot_to_scalar) for annot in new_annots])
        df_sequence_metadata = pd.DataFrame(infos)


        #%% Renaming
        df_sequence_metadata = df_sequence_metadata.set_index("sequence_id")
        df_frame_metadata["id"] = df_frame_metadata["id"].astype(str)
        df_gtbbox_metadata[['image_id', 'id']] = df_gtbbox_metadata[['image_id', 'id']].astype(str)
        df_frame_metadata = df_frame_metadata.rename(columns={"id": "frame_id"}).set_index("frame_id")
        df_gtbbox_metadata = df_gtbbox_metadata.rename(columns={"image_id": "frame_id"}).set_index(["frame_id", "id"])

        #todo temporary
        #%% Add Sequence information to Frame
        sequence_columns = ['is_night', "weather"]
        for sequence_id in df_sequence_metadata.index:
            df_frame_metadata.loc[df_frame_metadata["sequence_id"]==sequence_id, sequence_columns] = df_sequence_metadata.loc[sequence_id, sequence_columns].values


        #%% Tagets
        targets = df_gtbbox_metadata.groupby("frame_id").apply(lambda x: x[["x0", "y0", "x1", "y1"]].values).to_dict()

        targets_torch = {}
        for key, val in targets.items():
            # Target and labels
            target = [
                dict(
                    boxes=torch.tensor(
                        val)
                )]
            target[0]["labels"] = torch.tensor([0] * len(target[0]["boxes"]))
            targets_torch[key] = target


        #targets = {k: v for d in targets_sequence_list for k, v in d.items()}
        #df_gtbbox_metadata = pd.concat(df_gtbbox_metadata_sequence_list, axis=0)
        #df_frame_metadata = pd.concat(df_frame_metadata_sequence_list, axis=0)
        #df_sequence_metadata = pd.concat(df_sequence_metadata_sequence_list, axis=0)


        # todo specific to MOTSynth
        adverse_weather = ['THUNDER', 'SMOG', 'FOGGY', 'BLIZZARD', 'RAIN', 'CLOUDS', 'OVERCAST']  # 'CLEAR' 'EXTRASUNNY',
        df_frame_metadata["adverse_weather"] = 1 * df_frame_metadata["weather"].apply(lambda x: x in adverse_weather)
        extreme_weather = ['THUNDER', 'SMOG', 'FOGGY', 'BLIZZARD', 'RAIN']  # 'CLEAR' 'EXTRASUNNY',
        df_frame_metadata["extreme_weather"] = 1 * df_frame_metadata["weather"].apply(lambda x: x in extreme_weather)

        # Additional processing
        # Todo https://github.com/cocodataset/cocoapi/issues/130
        keypoints_label_names = [f"o_{i}" for i in range(self.NUM_KEYPOINTS)]
        df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata[keypoints_label_names].apply(lambda x: (2 - x)).mean(axis=1)

        # todo seems there is a bug on pitch/roll/yaw. We assume a mistake of MoTSynth authors, and the referenced "yaw" is in fact "pitch"
        df_frame_metadata["temp"] = df_frame_metadata["pitch"]
        df_frame_metadata["pitch"] = df_frame_metadata["yaw"]
        df_frame_metadata["yaw"] = df_frame_metadata["temp"]

        return targets_torch, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata