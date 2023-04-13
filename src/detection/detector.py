import os
import torch
from tqdm import tqdm
import json
from mmdet.apis import init_detector, inference_detector

class Detector:
    def __init__(self, name, device="cuda", nms=False):
        self.model_name = name
        self.device = device
        self.config_path, self.checkpoint_path = self.get_config_and_checkpoints_path()
        self.nms = nms

    def get_config_and_checkpoints_path(self):
        if self.model_name == "faster-rcnn_cityscapes": #todo change here
            checkpoint_path = "faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
            config_path = "../../models/faster_rcnn/faster-rcnn_cityscapes.py"
        elif self.model_name == "yolo3_coco":
            config_path = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/yolo/yolov3_d53_320_273e_coco.py"
            checkpoint_path = "/home/raphael/work/checkpoints/detection/yolov3_d53_320_273e_coco-421362b6.pth"
        elif self.model_name == "faster-rcnn_coco":
            config_path = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
            checkpoint_path = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
        else:
            raise ValueError(f"Model name {self.model_name} not known")

        return config_path, checkpoint_path

    #model_name = "yolo3_coco"
    #model_name = "faster-rcnn_coco"
    #checkpoint_root = "/home/raphael/work/checkpoints/detection"
    #configs_root = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs"

    # todo here bug with 1711 too many
    def get_preds_from_files(self, frame_id_list, file_list):

        config_file = self.config_path
        checkpoint_file = self.checkpoint_path

        # todo problem with int / str id

        json_root = f"data/preds/{config_file.split('/')[-1].replace('.py', '')}"
        os.makedirs(json_root, exist_ok=True)
        json_file = f"{json_root}/preds.json"

        # If exist load it
        if os.path.isfile(json_file):
            with open(json_file) as f:
                preds = json.load(f)

                # To pytorch for the ones it needs
                for key in preds.keys():
                    preds[key][0]["boxes"] = torch.tensor(preds[key][0]["boxes"])
                    preds[key][0]["scores"] = torch.tensor(preds[key][0]["scores"])
                    preds[key][0]["labels"] = torch.tensor(preds[key][0]["labels"])
        else:
            preds = {}

        # How many preds done, how many to do more ?
        missing_frames = []
        set_missing_frames = list(set(frame_id_list) - set(preds.keys()))
        missing_files = []
        for (file, img_id) in zip(file_list, frame_id_list):
            if img_id in set_missing_frames:
                missing_files.append(file)
                missing_frames.append(img_id)
        print(f"{len(frame_id_list)} img done already, predicting for {len(missing_frames)} more.")

        if len(missing_frames) > 0:
            model = init_detector(config_file, checkpoint_file, device=self.device)

        for i in tqdm(range(len(missing_files))):

            frame_id = missing_frames[i]
            img_path = missing_files[i]

            try:
                # test a single image and show the results
                result = inference_detector(model, img_path)
                bboxes_people = result[0]

                if self.nms:
                    bboxes_people, _ = self.nms(
                        bboxes_people[:, :4],
                        bboxes_people[:, 4],
                        0.25,
                        score_threshold=0.25)

                pred = [
                    dict(
                        boxes=torch.tensor(bboxes_people[:, :4]),
                        scores=torch.tensor(bboxes_people[:, 4]),
                        labels=torch.tensor([0] * len(bboxes_people)),
                        img_path=img_path
                    )
                ]
                preds[str(frame_id)] = pred
            except:
                print(f"Could not infer {frame_id} {img_path}")

        # to be able to save it
        preds_json = {}
        for key, val in preds.items():
            preds_json[key] = [{
                "boxes": val[0]["boxes"].numpy().tolist(),
                "scores": val[0]["scores"].numpy().tolist(),
                "labels": val[0]["labels"].numpy().tolist(),
                "img_path": val[0]["img_path"],
            }]

        # Save predictions that have been done
        with open(json_file, 'w') as f:
            json.dump(preds_json, f)

        # Only keys we want
        preds_out = {key: preds[key] for key in frame_id_list}

        return preds_out