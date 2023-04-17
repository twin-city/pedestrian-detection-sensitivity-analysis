import json
import numpy as np
import torch
import os.path as osp
import pandas as pd
import os
from utils import xywh2xyxy


def target_2_json(targets):
    return {key: [{
        "boxes": val[0]["boxes"].numpy().tolist(),
        "labels": val[0]["labels"].numpy().tolist(),
    }
    ] for key, val in targets.items()}


def target_2_torch(targets):
    return {key: [{
        "boxes": torch.tensor(val[0]["boxes"]),
        "labels": torch.tensor(val[0]["labels"]),
    }
    ] for key, val in targets.items()}


class MotsynthProcessing:
    """
    Class that handles the preprocessing of (extracted) MotSynth Dataset in order to get a standardized dataset format.
    """

    def __init__(self, max_samples=200, video_ids=None):

        np.random.seed(0)

        self.max_samples = max_samples



        #todo change this directory
        self.frames_dir = "/home/raphael/work/datasets/MOTSynth/frames"
        self.annot_dir = "/home/raphael/work/datasets/MOTSynth/coco annot"
        self.delay = 3
        self.saves_dir = "data/preprocessing/motsynth"
        os.makedirs(self.saves_dir, exist_ok=True)

        # todo bug 140, 174 and whatt appens if less samples than sequences ?????
        if video_ids is None:
            exclude_ids_frames = set(["060", "081", "026", "132", "136", "102", "099", "174", "140"])
            video_ids_frames = set(np.sort(os.listdir("/home/raphael/work/datasets/MOTSynth/frames")).tolist())
            video_ids_json = set([i.replace(".json", "") for i in
                                  os.listdir(self.annot_dir)]) - exclude_ids_frames
            self.video_ids = list(np.sort(list(set.intersection(video_ids_frames, video_ids_json))))
            if self.max_samples < len(self.video_ids):
                self.video_ids = np.random.choice(self.video_ids, max_samples, replace=False)
        else:
            self.video_ids = video_ids


    def get_dataset(self):
        targets, metadatas, frame_id_list, img_path_list = self.get_MoTSynth_annotations_and_imagepaths(
            self,
            video_ids=self.video_ids,
            max_samples=self.max_sample)

        return targets, metadatas, frame_id_list, img_path_list


    def get_MoTSynth_annotations_and_imagepaths_video(self, video_id="004", max_samples=100000, random_sampling=True, delay=3):


        np.random.seed(0)
        df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = [pd.DataFrame()]*3
        json_path = f"{self.annot_dir}/{video_id}.json"
        with open(json_path) as jsonFile:
            annot_motsynth = json.load(jsonFile)

        frame_metadata = {}
        targets = {}
        img_path_list = []
        targets_metadata = {}

        # Set images to process (subset for ptotyping)
        if random_sampling:
            random_set = np.random.choice(len(annot_motsynth["images"][delay:]), max_samples, replace=False)
            image_set = [x for i, x in enumerate(annot_motsynth["images"][delay:]) if i in random_set]
        else:
            image_set = annot_motsynth["images"][delay:delay+max_samples]

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
            keypoints = [(np.array(annot["keypoints"])).reshape((22, 3))[:,2] for annot in annots]
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
                "keypoints": keypoints,
                "area": area,
                "is_crowd": is_crowd,
                "is_blurred": is_blurred, #todo check examples in dataset
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
                img_path_list.append(osp.join("/home/raphael/work/datasets/MOTSynth", image["file_name"]))
                # frame_metadata[frame_id] = (annot_ECP["tags"], [ann["tags"] for ann in annot_ECP["children"]])

                # Dataframes
                df_gtbbox_metadata = pd.concat([df_gtbbox_metadata, pd.DataFrame(target_metadata)], axis=0)



                frame_metadata_features = ['file_name', 'id', 'frame_n'] + \
                                          ["is_night", "seq_name", "weather"] + \
                                          ["is_moving", "cam_fov", 'fx', 'fy', 'cx', 'cy'] + \
                                          ["x", "y", "z", "yaw", "pitch", "roll"]
                df_frame_metadata = pd.concat([df_frame_metadata, pd.DataFrame({key:val for key,val in image.items() if key in frame_metadata_features}, index=[frame_id])], axis=0)
        frame_id_list = list(targets.keys())

        # Metadata at the video level
        df_sequence_metadata = pd.DataFrame(annot_motsynth["info"], index=[video_id])

        metadatas = df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

        return targets, metadatas, frame_id_list, img_path_list




    def get_MoTSynth_annotations_and_imagepaths(self):

        path_df_gtbbox_metadata = osp.join(self.saves_dir, f"df_gtbbox_{self.max_samples}.csv")
        path_df_frame_metadata = osp.join(self.saves_dir, f"df_frame_{self.max_samples}.csv")
        path_df_sequence_metadata = osp.join(self.saves_dir, f"df_sequence_{self.max_samples}.csv")
        path_target = osp.join(self.saves_dir, f"targets_{self.max_samples}.json")

        try:
            print("Try")
            df_gtbbox_metadata = pd.read_csv(path_df_gtbbox_metadata).set_index(["image_id", "id"])
            df_frame_metadata = pd.read_csv(path_df_frame_metadata).set_index("Unnamed: 0")
            df_sequence_metadata = pd.read_csv(path_df_sequence_metadata).set_index("Unnamed: 0")


            with open(path_target) as jsonFile:
                targets = target_2_torch(json.load(jsonFile))
            print("End Try")

        except:
            print("Except")
            video_ids = self.video_ids
            max_samples = self.max_samples
            df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = [pd.DataFrame()]*3

            if video_ids is None:
                folders = os.listdir(self.frames_dir)
            else:
                folders = video_ids

            num_folders = len(list(folders))
            max_num_sample_per_video = int(max_samples/num_folders)

            targets, targets_metadata, frames_metadata, frame_id_list, img_path_list = {}, {}, {}, [], []

            for folder in folders:
                print(f"Checking folder {folder}")

                try:
                    targets_folder, targets_metadatas, frame_id_list_folder, img_path_list_folder =\
                        self.get_MoTSynth_annotations_and_imagepaths_video(video_id=folder, max_samples=max_num_sample_per_video)
                    df_gtbbox_metadata_folder, df_frame_metadata_folder, df_sequence_metadata_folder = targets_metadatas
                    targets.update(targets_folder)
                    df_gtbbox_metadata = pd.concat([df_gtbbox_metadata, pd.DataFrame(df_gtbbox_metadata_folder)], axis=0)
                    df_frame_metadata = pd.concat([df_frame_metadata, pd.DataFrame(df_frame_metadata_folder)], axis=0)
                    df_sequence_metadata = pd.concat([df_sequence_metadata, pd.DataFrame(df_sequence_metadata_folder)], axis=0)
                    frame_id_list += [str(i) for i in frame_id_list_folder]
                    img_path_list += img_path_list_folder
                except:
                    print(f"Could not load data from sequence {folder}")

            # compute occlusion rates
            df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])
            df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata["keypoints"].apply(lambda x: 1 - (x - 1).mean())

            # Compute specific cofactors
            adverse_weather = ['THUNDER', 'SMOG', 'FOGGY', 'BLIZZARD', 'RAIN', 'CLOUDS',
                               'OVERCAST']  # 'CLEAR' 'EXTRASUNNY',
            df_frame_metadata["adverse_weather"] = df_frame_metadata["weather"].apply(lambda x: x in adverse_weather)

            extreme_weather = ['THUNDER', 'SMOG', 'FOGGY', 'BLIZZARD', 'RAIN']  # 'CLEAR' 'EXTRASUNNY',
            df_frame_metadata["extreme_weather"] = df_frame_metadata["weather"].apply(lambda x: x in extreme_weather)


            # Drop and rename too
    #        df_frame_metadata.drop(["cam_fov", "frame_n"], inplace=True)

            new_cam_extr_names = {key: f'cam-extr-{key}' for key in ["x", "y", "z", "yaw", "pitch", "roll"]}
            #df_frame_metadata = df_frame_metadata.rename(columns=new_cam_extr_names)

            new_cam_intr_names = {key: f'cam-intr-{key}' for key in ['fx', 'fy', 'cx', 'cy']}
            #df_frame_metadata = df_frame_metadata.rename(columns=new_cam_intr_names)

            metadatas = df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

            df_gtbbox_metadata = df_gtbbox_metadata.reset_index()
            df_gtbbox_metadata["image_id"] = df_gtbbox_metadata["image_id"] - self.delay
            df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])

            #todo : american, synthetic, ... (labels)

            # Save dataframes
            df_gtbbox_metadata.to_csv(path_df_gtbbox_metadata)
            df_frame_metadata.to_csv(path_df_frame_metadata)
            df_sequence_metadata.to_csv(path_df_sequence_metadata)
            with open(path_target, 'w') as f:
                json.dump(target_2_json(targets), f)

        return targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata
