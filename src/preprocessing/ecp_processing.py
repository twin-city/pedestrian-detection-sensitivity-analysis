import json
import os.path as osp
import os
from .processing import DatasetProcessing
from .preprocessing_utils import *


class ECPProcessing(DatasetProcessing):
    """
    Class that handles the preprocessing of (extracted) ECP Dataset in order to get a standardized dataset format.
    """

    def __init__(self, root, max_samples_per_sequence=10, task="pedestrian_detection"):
        self.dataset_name = "ecp"
        super().__init__(root, max_samples_per_sequence, task)
        os.makedirs(self.saves_dir, exist_ok=True)

        assert task in ["pedestrian_detection"]

    def preprocess_sequence(self, sequence_id, img_sequence_dir, annot_sequence_dir, force_recompute=False):

        # todo some asserts
        # Here ECP specific
        total_frame_ids = [x.split(".png")[0] for x in os.listdir(img_sequence_dir) if ".png" in x]
        total_annot_ids = [x.split(".json")[0] for x in os.listdir(annot_sequence_dir) if ".json" in x]
        assert(len(total_frame_ids) == len(total_annot_ids))

        time = "day" if "day" in img_sequence_dir else "night"

        infos = {}
        infos["sequence_id"] = sequence_id
        new_annots = []
        new_images = []

        for i, frame_id in enumerate(total_frame_ids):

            # Load ECP annotations
            img_path = f"{img_sequence_dir}/{frame_id}.png"
            json_path = f"{annot_sequence_dir}/{frame_id}.json"

            with open(json_path) as jsonFile:

                # Load and keep subset of annots
                annot_ECP = json.load(jsonFile)
                annots = [c for c in annot_ECP["children"] if c["identity"] in ["pedestrian", "rider"]]

                # Keep only if there are annots
                if len(annots) > 0:

                    # New imgs
                    img = {"file_name": img_path.split(self.root)[1][1:], "id": frame_id, "is_night": time == "night"}
                    img["sequence_id"] = sequence_id
                    categories = ["motionBlur", "rainy", "wiper", "lenseFlare", "constructionSite"]
                    img_tags = {cat: cat in annot_ECP["tags"] for cat in categories}
                    img_tags.update({"weather": "rainy" if "rainy" in annot_ECP["tags"] else "dry"})
                    img_tags.update({"weather_original": "rainy" if "rainy" in annot_ECP["tags"] else "dry"})
                    img.update(img_tags)
                    new_images.append(img)

                    # New annots
                    for j, annot in enumerate(annots):
                        new_annots.append({"id": f"{frame_id}_{j}", "image_id": frame_id,
                                 "x0": annots[j]["x0"], "y0": annots[j]["y0"], "x1": annots[j]["x1"], "y1": annots[j]["y1"],
                                 'identity': annots[j]["identity"], 'tags': annots[j]["tags"],
                                 "sequence_id": sequence_id,
                                 "occlusion_rate": syntax_occl_ECP(annots[j]["tags"]),
                                 "truncation_rate": syntax_truncated_ECP(annots[j]["tags"]),
                                        })
                    #todo depiction ?

        return infos, new_images, new_annots


    def get_sequence_dict(self):

        sequence_dict = {}
        for luminosity in ["day", "night"]:
            for chosen_set in ["val"]:
                img_folder_dir = f"{self.root}/{luminosity}/img/{chosen_set}"
                annot_folder_dir = f"{self.root}/{luminosity}/labels/{chosen_set}"

                #todo an assert here ?
                for city in os.listdir(img_folder_dir):
                    sequence_id = f"{luminosity}_{chosen_set}_{city}"
                    img_sequence_dir = osp.join(img_folder_dir, city)
                    annot_sequence_dir = osp.join(annot_folder_dir, city)
                    sequence_dict[sequence_id] = (img_sequence_dir, annot_sequence_dir)

        return sequence_dict


    def preprocess_specific(self, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata):


        # Weather
        df_frame_metadata["weather_original"] = df_frame_metadata["weather"]
        df_frame_metadata["weather"] = df_frame_metadata["weather_original"].replace({"dry": "clear", "rainy": "rain"})

        # Weather categories according to homegenized weather naming
        df_frame_metadata = self.add_weather_cats(df_frame_metadata)

        # Occlusion rate
        df_gtbbox_metadata["occlusion_rate_original"] = df_gtbbox_metadata["occlusion_rate"]
        df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata.apply(lambda x: x[["occlusion_rate", "truncation_rate"]].max(), axis=1)
        return df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata


    def explore_tags(self):
        #%% Check all tags
        """
        print("Data Check ECP ======================")
        frame_tags_list = []
        frame_children_tags_list = []
        frame_children_identities = []
        for i, frame_id in enumerate(total_frame_ids):
            json_path = f"{self.root}/{time}/labels/{set}/{city}/{frame_id}.json"
            with open(json_path) as jsonFile:
                annot_ECP = json.load(jsonFile)
                if len(annot_ECP["children"]) >0:
                    frame_tags_list += annot_ECP["tags"]
                    frame_children_tags_list += np.concatenate([x["tags"] for x in annot_ECP["children"]]).tolist()
                    frame_children_identities += [x["identity"] for x in annot_ECP["children"]]

        print(pd.Series(frame_tags_list).value_counts())
        print(pd.Series(frame_children_tags_list).value_counts())
        print(pd.Series(frame_children_identities).value_counts())
        print("END Data Check ECP ======================")
        """
        pass