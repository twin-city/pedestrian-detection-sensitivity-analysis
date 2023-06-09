import unittest
import pandas as pd
from src.utils import subset_dataframe

class TestDfFiltering(unittest.TestCase):

    def test_bbox_filtering(self):
        df = pd.read_csv("data/df_gtbbox_metadata_frame_twincity_test.csv")
        gtbbox_filtering3 = {"height": {">": 50}}
        df_subset = subset_dataframe(df, gtbbox_filtering3)
        excluded_gt = set(range(len(df))) - set(df_subset.index)
        self.assertEqual(excluded_gt, set([13,5]))

    def test_bbox_filtering2(self):
        df = pd.read_csv("data/df_gtbbox_metadata_frame_twincity_test.csv")
        gtbbox_filtering3 = {"height": {"between": (50, 100)}}
        df_subset = subset_dataframe(df, gtbbox_filtering3)
        excluded_gt = set(range(len(df))) - set(df_subset.index)
        self.assertEqual(excluded_gt, {0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18})


