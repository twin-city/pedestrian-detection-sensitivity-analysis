import json
import numpy as np
import torch
import os.path as osp
import pandas as pd
import os
from src.utils import target_2_json, target_2_torch
from src.plot_utils import xywh2xyxy
from .processing import DatasetProcessing
from configs_path import ROOT_DIR


# todo specific to motsynth
# todo bug 140, 174 and whatt appens if less samples than sequences ?????
# todo the bug of delay of 3

class MotsynthProcessing(DatasetProcessing):

    def __init__(self, root, max_samples=200, sequence_ids=None):

        self.dataset_name = "motsynth"
        super().__init__(root, max_samples)

        self.delay = 3
        self.frames_dir = f"{root}/frames"
        self.annot_dir = f"{root}/coco annot"
        os.makedirs(self.saves_dir, exist_ok=True)
        self.sequence_ids = sequence_ids



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
        targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.load_or_preprocess_motsynth(force_recompute)

        # Common post-processing
        df_gtbbox_metadata = self.format_gtbbox_metadata(df_gtbbox_metadata)
        df_frame_metadata = self.format_frame_metadata(df_frame_metadata, df_gtbbox_metadata)

        return self.root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

    def get_MoTSynth_annotations_and_imagepaths_sequence(self, sequence_id="004", max_samples=100, random_sampling=True, delay=3):

        np.random.seed(0)
        df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = [pd.DataFrame()]*3
        json_path = f"{self.annot_dir}/{sequence_id}.json"
        with open(json_path) as jsonFile:
            annot_motsynth = json.load(jsonFile)

        frame_metadata = {}
        targets = {}
        img_path_list = []
        targets_metadata = {}

        # Set images to process (subset for ptotyping)

        if not "small" in self.root:
            if random_sampling:
                random_set = np.random.choice(len(annot_motsynth["images"][delay:]), max_samples, replace=False)
                image_set = [x for i, x in enumerate(annot_motsynth["images"][delay:]) if i in random_set]
            else:
                image_set = annot_motsynth["images"][delay:delay+max_samples]
        else:
            image_set = annot_motsynth["images"]

        for image in image_set:

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
        metadatas = df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

        return targets, metadatas, frame_id_list, img_path_list


    def load_or_preprocess_motsynth(self, force_recompute=False):

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


        """
        try:
            print("Try")
            df_gtbbox_metadata = pd.read_csv(path_df_gtbbox_metadata).set_index(["image_id", "id"])
            df_frame_metadata = pd.read_csv(path_df_frame_metadata).set_index("Unnamed: 0")
            df_sequence_metadata = pd.read_csv(path_df_sequence_metadata).set_index("Unnamed: 0")
            with open(path_target) as jsonFile:
                targets = target_2_torch(json.load(jsonFile))
            print("End Try")

        except:
            print("Did not find precomputed metadatas.")
            targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.preprocess_motsynth()



        if force_recompute:
            targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.preprocess_motsynth()
        """

        """
        df_frame_metadata.index.rename("frame_id", inplace=True)
        df_frame_metadata = df_frame_metadata.reset_index()
        df_frame_metadata["frame_id"] = df_frame_metadata["frame_id"].astype(str)
        df_frame_metadata = df_frame_metadata.set_index("frame_id")

        df_gtbbox_metadata = df_gtbbox_metadata.reset_index()
        df_gtbbox_metadata["id"] = df_gtbbox_metadata["id"].astype(str)
        df_gtbbox_metadata["image_id"] = df_gtbbox_metadata["image_id"].astype(str)
        df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])
        #df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])

        df_gtbbox_metadata.index = df_gtbbox_metadata.index.rename({"image_id": "frame_id"})
        df_gtbbox_metadata["num_pedestrian"] = df_gtbbox_metadata.groupby("frame_id").apply(len)
        """

        return targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata



    def preprocess_motsynth(self):

        sequence_ids = self.get_usable_sequence_ids()
        max_samples = self.max_samples
        df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = [pd.DataFrame()] * 3

        if sequence_ids is None:
            folders = os.listdir(self.frames_dir)
        else:
            folders = sequence_ids

        num_folders = len(list(folders))

        if max_samples is not None:
            max_num_sample_per_sequence = int(max_samples / num_folders)
        else:
            max_num_sample_per_sequence = 1000  # todo max to 1000 arbirterary

        targets, targets_metadata, frames_metadata, frame_id_list, img_path_list = {}, {}, {}, [], []

        for folder in folders:
            print(f"Checking folder {folder}")

            # try:
            targets_folder, targets_metadatas, frame_id_list_folder, img_path_list_folder = \
                self.get_MoTSynth_annotations_and_imagepaths_sequence(sequence_id=folder,
                                                                      max_samples=max_num_sample_per_sequence)
            df_gtbbox_metadata_folder, df_frame_metadata_folder, df_sequence_metadata_folder = targets_metadatas
            targets.update(targets_folder)
            df_gtbbox_metadata = pd.concat([df_gtbbox_metadata, pd.DataFrame(df_gtbbox_metadata_folder)], axis=0)
            df_frame_metadata = pd.concat([df_frame_metadata, pd.DataFrame(df_frame_metadata_folder)], axis=0)
            df_sequence_metadata = pd.concat([df_sequence_metadata, pd.DataFrame(df_sequence_metadata_folder)], axis=0)
            frame_id_list += [str(i) for i in frame_id_list_folder]
            img_path_list += img_path_list_folder
            # except:
            #    print(f"Could not load data from sequence {folder}")

        # compute occlusion rates
        # df_gtbbox_metadata = df_gtbbox_metadata.reset_index()
        # df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])
        # -->  0 : visible, 1 : occluded/truncated
        df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata["keypoints_label"].apply(lambda x: (2 - x).mean())

        # Compute specific cofactors
        adverse_weather = ['THUNDER', 'SMOG', 'FOGGY', 'BLIZZARD', 'RAIN', 'CLOUDS',
                           'OVERCAST']  # 'CLEAR' 'EXTRASUNNY',
        df_frame_metadata["adverse_weather"] = 1 * df_frame_metadata["weather"].apply(lambda x: x in adverse_weather)

        extreme_weather = ['THUNDER', 'SMOG', 'FOGGY', 'BLIZZARD', 'RAIN']  # 'CLEAR' 'EXTRASUNNY',
        df_frame_metadata["extreme_weather"] = 1 * df_frame_metadata["weather"].apply(lambda x: x in extreme_weather)

        """
        # Drop and rename too
        df_frame_metadata.drop(["cam_fov", "frame_n"], inplace=True)
        new_cam_extr_names = {key: f'cam-extr-{key}' for key in ["x", "y", "z", "yaw", "pitch", "roll"]}
        #df_frame_metadata = df_frame_metadata.rename(columns=new_cam_extr_names)
        new_cam_intr_names = {key: f'cam-intr-{key}' for key in ['fx', 'fy', 'cx', 'cy']}
        #df_frame_metadata = df_frame_metadata.rename(columns=new_cam_intr_names)
        """

        # Additional processing
        keypoints_label_names = [f"keypoints_label_{i}" for i in range(self.NUM_KEYPOINTS)]
        # Todo https://github.com/cocodataset/cocoapi/issues/130
        df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata[keypoints_label_names].apply(lambda x: (2 - x)).mean(
            axis=1)
        # todo seems there is a bug on pitch/roll/yaw. We assume a mistake of MoTSynth authors, and the referenced "yaw" is in fact "pitch"
        df_frame_metadata["temp"] = df_frame_metadata["pitch"]
        df_frame_metadata["pitch"] = df_frame_metadata["yaw"]
        df_frame_metadata["yaw"] = df_frame_metadata["temp"]


        # Operations on indexes
        df_gtbbox_metadata = df_gtbbox_metadata.reset_index()
        df_gtbbox_metadata["frame_id"] = df_gtbbox_metadata["image_id"] - self.delay
        df_gtbbox_metadata[["frame_id", "id"]] = df_gtbbox_metadata[["frame_id", "id"]].astype(str)

        df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])

        df_frame_metadata.index = df_frame_metadata.index.astype(str)
        df_frame_metadata.index.name = "frame_id"

        df_sequence_metadata.index.name = "sequence_id"
        df_sequence_metadata.index = df_sequence_metadata.index.astype(str)

        return targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata
        # todo : american, synthetic, ... (labels)


"""

def visual_check_motsynth_annotations(sequence_num="004", img_file_name="0200.jpg", shift=3):


    json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{sequence_num}.json"
    with open(json_path) as jsonFile:
        annot_motsynth = json.load(jsonFile)


    img_id = [(x["id"]) for x in annot_motsynth["images"] if img_file_name in x["file_name"]][0]
    bboxes = [xywh2xyxy(x["bbox"]) for x in annot_motsynth["annotations"] if x["image_id"] == img_id+shift]

    img_path = f"/home/raphael/work/datasets/MOTSynth/frames/{sequence_num}/rgb/{img_file_name}"
    img = plt.imread(img_path)
    img = add_bboxes_to_img(img, bboxes, c=(0, 255, 0), s=6)

    keypoints = [(np.array(x["keypoints"])).reshape((self.NUM_KEYPOINTS, 3)) for x in annot_motsynth["annotations"] if
                 x["image_id"] == img_id + shift]

    for keypoint in keypoints:
        plt.scatter(keypoint[:, 0], keypoint[:, 1], c=keypoint[:, 2])


    plt.imshow(img)
    plt.show()



def get_motsynth_day_night_sequence_ids(max_iter=50, force=False):

    # Save
    if os.path.exists("/home/raphael/work/datasets/MOTSynth/coco_infos.json") or not force:
        with open("/home/raphael/work/datasets/MOTSynth/coco_infos.json") as jsonFile:
            sequence_info = json.load(jsonFile)
    else:
        sequence_info = {}


    for i, sequence_file in enumerate(mmcv.scandir("/home/raphael/work/datasets/MOTSynth/coco annot/")):

        print(sequence_file)

        if sequence_file.replace(".json", "") not in sequence_info.keys():
            try:
                json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{sequence_file}"

                with open(json_path) as jsonFile:
                    annot_motsynth = json.load(jsonFile)


                is_night = annot_motsynth["info"]["is_night"]
                print(sequence_file, is_night)

                sequence_info[sequence_file.replace(".json", "")] = annot_motsynth["info"]
            except:
                print(f"Did not work for {sequence_file}")

        if i > max_iter:
            break

    with open("/home/raphael/work/datasets/MOTSynth/coco_infos.json", 'w') as f:
        json.dump(sequence_info, f)
    night = []
    day = []

    day_index = [key for key, value in sequence_info.items() if
           not value["is_night"] and os.path.exists(f"/home/raphael/work/datasets/MOTSynth/frames/{key}")]
    night_index = [key for key, value in sequence_info.items() if
           value["is_night"] and os.path.exists(f"/home/raphael/work/datasets/MOTSynth/frames/{key}")]

    print("night", night_index)
    print("day", day_index)

    return day, night

"""