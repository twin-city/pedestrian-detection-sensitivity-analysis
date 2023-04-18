import numpy as np



class DatasetProcessing:
    def __init__(self, root, max_samples):
        self.root = root
        self.max_samples = max_samples
        np.random.seed(0)

    def __str__(self):
        return self.dataset_name

    def get_dataset(self):
        """
        Get all the annotations and image file paths from the original dataset.
        :return:
        """
        raise NotImplementedError