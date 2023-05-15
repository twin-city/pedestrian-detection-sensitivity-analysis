import numpy as np
from configs_path import ROOT_DIR

class DatasetProcessing:
    def __init__(self, root, max_samples):
        self.root = root
        self.max_samples = max_samples
        np.random.seed(0)

        self.saves_dir = f"{ROOT_DIR}/data/preprocessing/{self.dataset_name}"

    def __str__(self):
        return self.dataset_name

    def get_dataset(self):
        """
        Get all the annotations and image file paths from the original dataset.
        :return:
        """
        raise NotImplementedError