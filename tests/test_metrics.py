import unittest
import pandas as pd
from src.utils import subset_dataframe

class TestMetrics(unittest.TestCase):

    def test_bbox_filtering(self):

        df = pd.read_csv("data/df_gtbbox_metadata_frame_twincity_test.csv")
        gtbbox_filtering3 = {"height": {">": 50}}
        df_subset = subset_dataframe(df, gtbbox_filtering3)
        excluded_gt = set(range(len(df))) - set(df_subset.index)

        self.assertEqual(excluded_gt, set([13,5]))


