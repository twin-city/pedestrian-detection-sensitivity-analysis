import os.path as osp
from src.preprocessing.ecp_processing import ECPProcessing
from src.preprocessing.motsynth_processing import MotsynthProcessing
from src.dataset.dataset import Dataset
from src.preprocessing.coco_processing import COCOProcessing
from src.preprocessing.twincity_preprocessing import TwincityProcessing

class DatasetFactory():
    @staticmethod
    def get_dataset(dataset_name, max_samples_per_sequence, root, force_recompute=False, task="pedestrian_detection"):
        if "twincity" in dataset_name.lower():
            twincity_processor = TwincityProcessing(root, max_samples_per_sequence=max_samples_per_sequence, task=task)
            dataset = twincity_processor.get_dataset(force_recompute=force_recompute)
            return Dataset(dataset_name, max_samples_per_sequence, *dataset)
        elif dataset_name == "motsynth":
            raise NotImplementedError("Deprecated")
        elif dataset_name == "motsynth_small":
            motsynth_processor = MotsynthProcessing(root, max_samples_per_sequence=max_samples_per_sequence, task=task)
            dataset = motsynth_processor.get_dataset(force_recompute=force_recompute)
            return Dataset(dataset_name, max_samples_per_sequence, *dataset)
        elif dataset_name == "ECP":
            raise NotImplementedError("Deprecated")
        elif dataset_name == "ecp_small":
            ecp_processor = ECPProcessing(root, max_samples_per_sequence=max_samples_per_sequence, task=task)
            dataset = ecp_processor.get_dataset(force_recompute=force_recompute)
            return Dataset(dataset_name, max_samples_per_sequence, *dataset)
        # Handle case for coco datasets which have a coco.json file at the root
        elif osp.exists(osp.join(root, "coco.json")):
            coco_dataset = COCOProcessing(root, dataset_name, max_samples_per_sequence=max_samples_per_sequence, task=task)
            dataset = coco_dataset.get_dataset()
            return Dataset(dataset_name, max_samples_per_sequence, *dataset)
        else:
            raise NotImplementedError(f"Unknown dataset {dataset_name}")

