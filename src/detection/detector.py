import os
import torch
from tqdm import tqdm
import json
from mmdet.apis import init_detector, inference_detector
import os.path as osp
from configs_path import ROOT_DIR, MMDET_DIR, CHECKPOINT_DIR

class Detector:
    def __init__(self, name, device="cuda", nms=False, task="pedestrian_detection"):
        self.model_name = name
        self.device = device
        #self.config_path, self.checkpoint_path, self.inference_processor = self.get_config_and_checkpoints_path()
        self.nms = nms
        self.task = task

    """
    def get_config_and_checkpoints_path(self):


        if self.model_name == "faster-rcnn_cityscapes": #todo change here
            checkpoint_path = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
            config_path = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/faster_rcnn/faster-rcnn_cityscapes.py"
            def get_person_bbox(x):
                return x[0]
        elif self.model_name == "mask-rcnn_coco":
            # Specify the path to MMDetection model config and checkpoint file
            config_path = f'{MMDET_DIR}/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
            checkpoint_path = f'{CHECKPOINT_DIR}/detection/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
            def get_person_bbox(x):
                return x[0][0]
        else:
            raise ValueError(f"Model name {self.model_name} not known")
        
        elif self.model_name == "yolo3_coco":
            config_path = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/yolo/yolov3_d53_320_273e_coco.py"
            checkpoint_path = "/home/raphael/work/checkpoints/detection/yolov3_d53_320_273e_coco-421362b6.pth"
        elif self.model_name == "faster-rcnn_coco":
            #config_path = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
            config_path = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs/models/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
            checkpoint_path = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
        
        return config_path, checkpoint_path, get_person_bbox
        """



    def get_preds_from_files(self, dataset_name, root, df_frame_metadata):

        # Frame id list
        file_list = [osp.join(root, x) for x in df_frame_metadata["file_name"]]
        frame_id_list = list(df_frame_metadata.index)

        config_file = self.config_path
        checkpoint_file = self.checkpoint_path

        # todo problem with int / str id

        json_root = f"{ROOT_DIR}/cache/inference/{dataset_name}/{config_file.split('/')[-1].replace('.py', '')}"
        os.makedirs(json_root, exist_ok=True)
        json_file = f"{json_root}/preds_{dataset_name}_{config_file.split('/')[-1].replace('.py', '')[:10]}.json"

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
        print(f"{len(frame_id_list)-len(missing_frames)} img done already, predicting for {len(missing_frames)} more.")

        if len(missing_frames) > 0:
            model = init_detector(config_file, checkpoint_file, device=self.device)
        else:
            model = init_detector(config_file, checkpoint_file, device="cpu")

        for i in tqdm(range(len(missing_files))):

            frame_id = missing_frames[i]
            img_path = missing_files[i]

            try:
                # test a single image and show the results
                result = inference_detector(model, img_path)
                #bboxes_people = result[0]
                bboxes_person = self.inference_processor(result)

                if self.nms:
                    bboxes_person, _ = self.nms(
                        bboxes_person[:, :4],
                        bboxes_person[:, 4],
                        0.25,
                        score_threshold=0.25)

                pred = [
                    dict(
                        boxes=torch.tensor(bboxes_person[:, :4]),
                        scores=torch.tensor(bboxes_person[:, 4]),
                        labels=torch.tensor([0] * len(bboxes_person)),
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
        try:
            preds_out = {key: preds[key] for key in frame_id_list}
        except:
            raise ValueError("Could not inder all frames due to misattribution of frame id")

        return preds_out