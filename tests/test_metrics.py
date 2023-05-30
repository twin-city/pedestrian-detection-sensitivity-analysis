import unittest
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from src.detection.metrics import compute_fp_missratio

class testMetrics(unittest.TestCase):

    def test_metrics(self):
        """
        A blank image with user defined target bboxes and predicted bboxes
        :return:
        """


        preds = [
            dict(
                boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
                scores=torch.tensor([0.536]),
                labels=torch.tensor([0]),
            )
        ]
        target = [
            dict(
                boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
                labels=torch.tensor([0]),
            )
        ]

        #todo as well as check Pedestron & NightOwls --> Does it handle ignore-regions ?????? Use pycocotools instead
        # Torch mAP
        metric = MeanAveragePrecision()
        metric.update(preds, target)
        metric.compute()
        # pprint(metric.compute())

        fp_missratio = compute_fp_missratio(preds, target, threshold=0.5, excluded_gt=None)
        gt_fp_missratio = {'num_ground_truth': 1, 'num_false_positives': 0, 'false_positives': None, 'false_negatives': [], 'true_positives': [0], 'ignore_regions': [], 'missing_rate': 0.0}
        self.assertEqual(fp_missratio, gt_fp_missratio)




