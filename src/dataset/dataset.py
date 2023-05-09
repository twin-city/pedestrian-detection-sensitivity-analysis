import pandas as pd
import os.path as osp
import os
from src.utils import compute_correlations
from src.plot_utils import plot_correlations
from src.plot_utils import plot_dataset_statistics

class Dataset():
    def __init__(self, dataset_name, max_sample, root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata):
        self.dataset_name = dataset_name
        self.root = root
        self.targets = targets
        self.df_gtbbox_metadata = df_gtbbox_metadata
        self.df_frame_metadata = df_frame_metadata
        self.df_sequence_metadata = df_sequence_metadata

        # Create dir if needed
        self.results_dir = osp.join("../../", "results", dataset_name, f"{dataset_name}{max_sample}")
        os.makedirs(self.results_dir, exist_ok=True)

    def get_dataset_as_tuple(self):
        return self.root, self.targets, self.df_gtbbox_metadata, self.df_frame_metadata, self.df_sequence_metadata


    # I/O

    def create_markdown_description_table(self):
        n_images = self.df_frame_metadata.groupby("is_night").apply(len)
        n_seqs = self.df_frame_metadata.groupby("is_night").apply(lambda x: len(x["seq_name"].unique()))
        n_person = self.df_frame_metadata.groupby("is_night").apply(lambda x: x["num_person"].sum())
        weathers = self.df_frame_metadata["weather"].unique()

        df_descr = pd.DataFrame({
            "sequences (day/night)": f"{n_seqs[0]}/{n_seqs[1]}",
            "images (day/night)": f"{n_images[0]}/{n_images[1]}",
            "person (day/night)": f"{n_person[0]}/{n_person[1]}",
            "weather": ", ".join(list(weathers)),
        }, index=[self.dataset_name]).T

        with open(f'descr_{self.dataset_name}.md', 'w') as f:
            f.write(df_descr.to_markdown())

    # Plotting
    def plot_dataset_sequence_correlation(self, sequence_cofactors):
        corr_matrix, p_matrix = compute_correlations(self.df_frame_metadata.groupby("seq_name").apply(lambda x: x.mean()), sequence_cofactors)
        plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")

    def plot_dataset_statistics(self):
        plot_dataset_statistics(self.df_gtbbox_metadata, self.results_dir)
