from src.preprocessing.twincity_preprocessing2 import get_twincity_dataset
from src.preprocessing.ecp_processing import ECPProcessing
from src.preprocessing.motsynth_processing import MotsynthProcessing
from src.dataset.dataset import Dataset


class DatasetFactory():
    @staticmethod
    def get_dataset(dataset_name, max_sample):
        if dataset_name == "twincity":
            root = "/home/raphael/work/datasets/twincity-Unreal/v5"
            dataset = get_twincity_dataset(root, max_sample)
            return Dataset(dataset_name, max_sample, *dataset)
        elif dataset_name == "motsynth":
            root = "/home/raphael/work/datasets/MOTSynth/"
            motsynth_processor = MotsynthProcessing(root, max_samples=max_sample, video_ids=None)
            dataset = motsynth_processor.get_dataset()  # todo as class
            return Dataset(dataset_name, max_sample, *dataset)
        elif dataset_name == "EuroCityPerson":
            root = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/"
            ecp_processor = ECPProcessing(root, max_samples=max_sample)
            dataset = ecp_processor.get_dataset()
            return Dataset(dataset_name, max_sample, *dataset)
        else:
            #todo handle any coco-formatted dataset
            raise NotImplementedError(f"Unknown dataset {dataset_name}")

