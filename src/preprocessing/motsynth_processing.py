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


class MotsynthProcessing(DatasetProcessing):


    def __init__(self, root, max_samples=200, video_ids=None):

        self.dataset_name = "motsynth"
        super().__init__(root, max_samples)

        self.delay = 3
        self.frames_dir = f"{root}/frames"
        self.annot_dir = f"{root}/coco annot"
        os.makedirs(self.saves_dir, exist_ok=True)
        self.video_ids = video_ids

        # todo specific to motsynth
        # todo bug 140, 174 and whatt appens if less samples than sequences ?????


    def get_video_ids(self):
        if self.video_ids is None:
            exclude_ids_frames = set(["060", "081", "026", "132", "136", "102", "099", "174", "140"])
            video_ids_frames = set(np.sort(os.listdir(self.frames_dir)).tolist())
            video_ids_json = set([i.replace(".json", "") for i in
                                  os.listdir(self.annot_dir)]) - exclude_ids_frames
            video_ids = list(np.sort(list(set.intersection(video_ids_frames, video_ids_json))))
            if self.max_samples < len(video_ids):
                video_ids = np.random.choice(video_ids, self.max_samples, replace=False)
        else:
            video_ids = self.video_ids

        return video_ids


    def get_dataset(self):
        targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata= self.get_MoTSynth_annotations_and_imagepaths()

        df_gtbbox_metadata["aspect_ratio"] = 1 / df_gtbbox_metadata["aspect_ratio"]
        mu = 0.4185
        std = 0.12016
        df_gtbbox_metadata["aspect_ratio_is_typical"] = np.logical_and(df_gtbbox_metadata["aspect_ratio"] < mu + std,
                                                                       df_gtbbox_metadata["aspect_ratio"] > mu - std)

        df_frame_metadata["num_pedestrian"] = df_gtbbox_metadata.groupby("frame_id").apply(len).loc[df_frame_metadata.index]

        return self.root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

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

            height = [x["bbox"][3] for x in annots]
            width = [x["bbox"][2] for x in annots]
            aspect_ratio = [h/w for h,w in zip(height, width)]

            keypoints_label = [(np.array(annot["keypoints"])).reshape((22, 3))[:,2] for annot in annots]
            keypoints_posx = [(np.array(annot["keypoints"])).reshape((22, 3))[:,0] for annot in annots]
            keypoints_posy = [(np.array(annot["keypoints"])).reshape((22, 3))[:, 1] for annot in annots]
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

                keypoints_label_names = [f"keypoints_label_{i}" for i in range(22)]
                keypoints_posx_names = [f"keypoints_posx_{i}" for i in range(22)]
                keypoints_posy_names = [f"keypoints_posy_{i}" for i in range(22)]


                df_gtbbox_metadata[keypoints_label_names] = df_gtbbox_metadata["keypoints_label"].apply(lambda x: pd.Series(x))
                df_gtbbox_metadata[keypoints_posx_names] = df_gtbbox_metadata["keypoints_posx"].apply(lambda x: pd.Series(x))
                df_gtbbox_metadata[keypoints_posy_names] = df_gtbbox_metadata["keypoints_posy"].apply(lambda x: pd.Series(x))


                frame_metadata_features = ['file_name', 'id', 'frame_n'] + \
                                          ["is_night", "seq_name", "weather"] + \
                                          ["is_moving", "cam_fov", 'fx', 'fy', 'cx', 'cy'] + \
                                          ["x", "y", "z", "yaw", "pitch", "roll"]
                df_frame_metadata = pd.concat([df_frame_metadata, pd.DataFrame({key:val for key,val in image.items() if key in frame_metadata_features}, index=[frame_id])], axis=0)
        frame_id_list = list(targets.keys())

        # Metadata at the video level
        df_sequence_metadata = pd.DataFrame(annot_motsynth["info"], index=[video_id])

        df_gtbbox_metadata["seq_name"] = df_frame_metadata["seq_name"].iloc[0]

        metadatas = df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

        # todo this info ? as a dict of additional dataset parameters I would say
        resolution = (1920, 1080)


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
            print("Did not find precomputed metadatas.")
            video_ids = self.get_video_ids()
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

                #try:
                targets_folder, targets_metadatas, frame_id_list_folder, img_path_list_folder =\
                    self.get_MoTSynth_annotations_and_imagepaths_video(video_id=folder, max_samples=max_num_sample_per_video)
                df_gtbbox_metadata_folder, df_frame_metadata_folder, df_sequence_metadata_folder = targets_metadatas
                targets.update(targets_folder)
                df_gtbbox_metadata = pd.concat([df_gtbbox_metadata, pd.DataFrame(df_gtbbox_metadata_folder)], axis=0)
                df_frame_metadata = pd.concat([df_frame_metadata, pd.DataFrame(df_frame_metadata_folder)], axis=0)
                df_sequence_metadata = pd.concat([df_sequence_metadata, pd.DataFrame(df_sequence_metadata_folder)], axis=0)
                frame_id_list += [str(i) for i in frame_id_list_folder]
                img_path_list += img_path_list_folder
                #except:
                #    print(f"Could not load data from sequence {folder}")

            # compute occlusion rates
            #df_gtbbox_metadata = df_gtbbox_metadata.reset_index()
            #df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])
            # -->  0 : visible, 1 : occluded/truncated
            df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata["keypoints_label"].apply(lambda x: (2-x).mean())

            # Compute specific cofactors
            adverse_weather = ['THUNDER', 'SMOG', 'FOGGY', 'BLIZZARD', 'RAIN', 'CLOUDS',
                               'OVERCAST']  # 'CLEAR' 'EXTRASUNNY',
            df_frame_metadata["adverse_weather"] = 1*df_frame_metadata["weather"].apply(lambda x: x in adverse_weather)

            extreme_weather = ['THUNDER', 'SMOG', 'FOGGY', 'BLIZZARD', 'RAIN']  # 'CLEAR' 'EXTRASUNNY',
            df_frame_metadata["extreme_weather"] = 1*df_frame_metadata["weather"].apply(lambda x: x in extreme_weather)

            """
            # Drop and rename too
            df_frame_metadata.drop(["cam_fov", "frame_n"], inplace=True)
            new_cam_extr_names = {key: f'cam-extr-{key}' for key in ["x", "y", "z", "yaw", "pitch", "roll"]}
            #df_frame_metadata = df_frame_metadata.rename(columns=new_cam_extr_names)
            new_cam_intr_names = {key: f'cam-intr-{key}' for key in ['fx', 'fy', 'cx', 'cy']}
            #df_frame_metadata = df_frame_metadata.rename(columns=new_cam_intr_names)
            """

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

        df_frame_metadata.index.rename("frame_id", inplace=True)
        df_frame_metadata = df_frame_metadata.reset_index()
        df_frame_metadata["frame_id"] = df_frame_metadata["frame_id"].astype(str)
        df_frame_metadata = df_frame_metadata.set_index("frame_id")

        df_gtbbox_metadata = df_gtbbox_metadata.reset_index()
        df_gtbbox_metadata["id"] = df_gtbbox_metadata["id"].astype(str)
        df_gtbbox_metadata["image_id"] = df_gtbbox_metadata["image_id"].astype(str)
        df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id", "id"])

        df_gtbbox_metadata.index = df_gtbbox_metadata.index.rename({"image_id": "frame_id"})
        df_gtbbox_metadata["num_pedestrian"] = df_gtbbox_metadata.groupby("frame_id").apply(len)
        keypoints_label_names = [f"keypoints_label_{i}" for i in range(22)]

        # Todo https://github.com/cocodataset/cocoapi/issues/130
        df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata[keypoints_label_names].apply(lambda x: (2 - x)).mean(
            axis=1)

        # todo seems there is a bug on pitch/roll/yaw. We assume a mistake of MoTSynth authors, and the referenced "yaw" is in fact "pitch"
        df_frame_metadata["temp"] = df_frame_metadata["pitch"]
        df_frame_metadata["pitch"] = df_frame_metadata["yaw"]
        df_frame_metadata["yaw"] = df_frame_metadata["temp"]


        df_gtbbox_metadata["ignore-region"] = 0

        return targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata



"""

def visual_check_motsynth_annotations(video_num="004", img_file_name="0200.jpg", shift=3):


    json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{video_num}.json"
    with open(json_path) as jsonFile:
        annot_motsynth = json.load(jsonFile)


    img_id = [(x["id"]) for x in annot_motsynth["images"] if img_file_name in x["file_name"]][0]
    bboxes = [xywh2xyxy(x["bbox"]) for x in annot_motsynth["annotations"] if x["image_id"] == img_id+shift]

    img_path = f"/home/raphael/work/datasets/MOTSynth/frames/{video_num}/rgb/{img_file_name}"
    img = plt.imread(img_path)
    img = add_bboxes_to_img(img, bboxes, c=(0, 255, 0), s=6)

    keypoints = [(np.array(x["keypoints"])).reshape((22, 3)) for x in annot_motsynth["annotations"] if
                 x["image_id"] == img_id + shift]

    for keypoint in keypoints:
        plt.scatter(keypoint[:, 0], keypoint[:, 1], c=keypoint[:, 2])


    plt.imshow(img)
    plt.show()



def get_motsynth_day_night_video_ids(max_iter=50, force=False):

    # Save
    if os.path.exists("/home/raphael/work/datasets/MOTSynth/coco_infos.json") or not force:
        with open("/home/raphael/work/datasets/MOTSynth/coco_infos.json") as jsonFile:
            video_info = json.load(jsonFile)
    else:
        video_info = {}


    for i, video_file in enumerate(mmcv.scandir("/home/raphael/work/datasets/MOTSynth/coco annot/")):

        print(video_file)

        if video_file.replace(".json", "") not in video_info.keys():
            try:
                json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{video_file}"

                with open(json_path) as jsonFile:
                    annot_motsynth = json.load(jsonFile)


                is_night = annot_motsynth["info"]["is_night"]
                print(video_file, is_night)

                video_info[video_file.replace(".json", "")] = annot_motsynth["info"]
            except:
                print(f"Did not work for {video_file}")

        if i > max_iter:
            break

    with open("/home/raphael/work/datasets/MOTSynth/coco_infos.json", 'w') as f:
        json.dump(video_info, f)
    night = []
    day = []

    day_index = [key for key, value in video_info.items() if
           not value["is_night"] and os.path.exists(f"/home/raphael/work/datasets/MOTSynth/frames/{key}")]
    night_index = [key for key, value in video_info.items() if
           value["is_night"] and os.path.exists(f"/home/raphael/work/datasets/MOTSynth/frames/{key}")]

    print("night", night_index)
    print("day", day_index)

    return day, night

"""