from .faster_rcnn_cityscapes import FasterRCNCityscapesDetector
from .mask_rcnn_coco import MaskRCNNCoco

class DetectorFactory():
    @staticmethod
    def get_detector(model_name, device="cuda", nms=False, task="pedestrian_detection"):
        if model_name.lower() == "faster-rcnn_cityscapes":
            return FasterRCNCityscapesDetector(model_name, device, nms, task)
        elif model_name == "mask-rcnn_coco":
            return MaskRCNNCoco(model_name, device, nms, task)
        else:
            raise NotImplementedError(f"Unknown model {model_name}")

