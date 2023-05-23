from src.preprocessing.ecp_processing import ECPProcessing
from src.preprocessing.motsynth_processing import MotsynthProcessing
from src.dataset.dataset import Dataset
from src.preprocessing.coco_processing import COCOProcessing
from src.preprocessing.twincity_preprocessing import TwincityProcessing

class DatasetFactory():
    @staticmethod
    def get_dataset(dataset_name, max_samples_per_sequence, root, coco_json_path=None, force_recompute=False):

        if dataset_name == "twincity":
            #root = "/home/raphael/work/datasets/twincity-Unreal/v5"
            #dataset = get_twincity_dataset(root, max_sample)
            twincity_processor = TwincityProcessing(root, max_samples_per_sequence=max_samples_per_sequence)
            dataset = twincity_processor.get_dataset(force_recompute=force_recompute)
            return Dataset(dataset_name, max_samples_per_sequence, *dataset)
        elif dataset_name == "motsynth":
            raise NotImplementedError("Deprecated")
            #root = "/home/raphael/work/datasets/MOTSynth/"
            #motsynth_processor = MotsynthProcessing(root, max_samples=max_sample)
            #dataset = motsynth_processor.get_dataset()
            return Dataset(dataset_name, max_sample, *dataset)
        elif dataset_name == "motsynth_small":
            #root = "/home/raphael/work/datasets/motsynth_small/"
            motsynth_processor = MotsynthProcessing(root, max_samples_per_sequence=max_samples_per_sequence)
            dataset = motsynth_processor.get_dataset(force_recompute=force_recompute)
            return Dataset(dataset_name, max_samples_per_sequence, *dataset)
        elif dataset_name == "ECP":
            raise NotImplementedError("Deprecated")
        elif dataset_name == "ecp_small":
            #root = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/"
            ecp_processor = ECPProcessing(root, max_samples_per_sequence=max_samples_per_sequence)
            dataset = ecp_processor.get_dataset(force_recompute=force_recompute)
            return Dataset(dataset_name, max_samples_per_sequence, *dataset)
        elif "coco" in dataset_name:
            coco_dataset = COCOProcessing(root, coco_json_path, dataset_name, max_samples_per_sequence=max_samples_per_sequence)
            dataset = coco_dataset.get_dataset()
            return Dataset(dataset_name, max_samples_per_sequence, *dataset)
        else:
            raise NotImplementedError(f"Unknown dataset {dataset_name}")

