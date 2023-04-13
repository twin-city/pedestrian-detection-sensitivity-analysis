from utils import visual_check_motsynth_annotations
visual_check_motsynth_annotations(video_num="145", img_file_name="1455.jpg", shift=3)


#%% What is the json ?????

import json
video_num = "004"

json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{video_num}.json"
with open(json_path) as jsonFile:
    annot_motsynth = json.load(jsonFile)


annot_tiny = annot_motsynth.copy()

#%%

annot_tiny["images"] = annot_tiny["images"][:2]
image_ids = [img_annot["id"] for img_annot in annot_tiny["images"]]
annot_tiny["annotations"] = [annot for annot in annot_tiny["annotations"] if annot["image_id"] in image_ids]

for annot in annot_tiny["annotations"]:
    del annot["segmentation"]

#%%

output_file = "annot_tiny2.json"

with open(output_file, 'w') as f:
    json.dump(annot_tiny, f)