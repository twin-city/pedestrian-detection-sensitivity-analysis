import pandas as pd
import os.path as osp
import os
from src.utils import compute_correlations
from src.plot_utils import plot_correlations
from src.plot_utils import plot_dataset_statistics

def print_mean_std(df, col, decimal=0):
    return f"{df[col].mean():.{decimal}f} +- ({df[col].std():.{decimal}f})"

def print_stat(df_stat):
    stat_day = 0 in df_stat.keys()
    stat_night = 1 in df_stat.keys()

    if stat_day and stat_night > 0:
        return f"{df_stat[0]}/{df_stat[1]}"
    elif stat_night == 0:
        return f"{df_stat[0]}/"
    elif stat_day == 0:
        return f"/{df_stat[1]}"
    else:
        raise ValueError("Cannot print in df_descr.md due to unknown error")

class Dataset():
    def __init__(self, dataset_name, max_sample, root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata):
        if max_sample is None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = dataset_name + "_"+str(max_sample)
        self.root = root
        self.targets = targets
        self.df_gtbbox_metadata = df_gtbbox_metadata
        self.df_frame_metadata = df_frame_metadata
        self.df_sequence_metadata = df_sequence_metadata
        self.max_sample = max_sample

        # Create dir if needed
        self.results_dir = osp.join("../../", "results", dataset_name, f"{dataset_name}{max_sample}")
        # os.makedirs(self.results_dir, exist_ok=True)
        # Show descriptive statistics
        # self.create_markdown_description_table()


    def get_dataset_dir(self):
        return f"{self.dataset_name}"


    def get_dataset_as_tuple(self):
        return self.root, self.targets, self.df_gtbbox_metadata, self.df_frame_metadata, self.df_sequence_metadata

    # I/O



    def create_markdown_description_table(self, folder_path="."):

        df_frame = self.df_frame_metadata.copy(deep=True)
        df_gtbbox = self.df_gtbbox_metadata.copy(deep=True)

        #todo fix
        if "is_night" not in df_frame.columns:
            df_frame["is_night"] = 0
        if "weather_original" not in df_frame.columns:
            df_frame["weather_original"] = "dry"
        if "occlusion_rate" not in df_gtbbox.columns:
            df_gtbbox["occlusion_rate"] = 0

        n_images = df_frame.groupby("is_night").apply(len)
        n_seqs = df_frame.groupby("is_night").apply(lambda x: len(x["sequence_id"].unique()))
        n_person = df_frame.groupby("is_night").apply(lambda x: x["num_pedestrian"].sum())
        weathers = df_frame["weather"].unique() #todo set as new weather cats ?
        weathers_cats = df_frame["weather_cats"].unique() #todo set as new weather cats ?

        #if self.max_sample is not None:
        dataset_version_name = f"{self.dataset_name}"
        #else:
        #    dataset_version_name = f"{self.dataset_name}"

        df_descr = pd.DataFrame({
            "sequences (day/night)": f"{print_stat(n_seqs)}",
            "images (day/night)": f"{print_stat(n_images)}",
            "person (day/night)": f"{print_stat(n_person)}",
            "weather": ", ".join(list(weathers)),
            "weather categories": ", ".join(list(weathers_cats)),
            # Additional info
            "height": print_mean_std(df_gtbbox.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True)), "height", decimal=0),
            "occlusion rate": print_mean_std(df_gtbbox.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True)), "occlusion_rate", decimal=2),
            "aspect ratio": print_mean_std(df_gtbbox.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True)), "aspect_ratio", decimal=2),

        }, index=[dataset_version_name]).T
        df_descr.index.name = "characteristics"

        #todo add pitch info also ??? Number of viewpoints ???


        # Save in a common dataframe to compare the datasets
        save_csv_path = osp.join(folder_path, 'df_descr.csv')
        save_md_path = osp.join(folder_path, 'df_descr.md')
        if not os.path.exists(save_csv_path):
            df_descr_all = df_descr
            df_descr_all.index.name = "characteristics"
        else:
            df_descr_all = pd.read_csv(save_csv_path).set_index("characteristics")
            if dataset_version_name not in df_descr_all.columns:
                df_descr_all = pd.concat([df_descr_all, df_descr], axis=1)
        df_descr_all.to_csv(save_csv_path)

        # todo add the dataset name in the title
        #with open(osp.join(self.results_dir, f'descr_{self.dataset_name}.md'), 'w') as f:
        #    f.write(df_descr.to_markdown())
        with open(save_md_path, 'w') as f:
            f.write(df_descr_all.to_markdown())

    # Plotting
    def plot_dataset_sequence_correlation(self, sequence_cofactors):
        corr_matrix, p_matrix = compute_correlations(self.df_frame_metadata.groupby("sequence_id").apply(lambda x: x.mean()), sequence_cofactors)
        plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")

    def plot_dataset_statistics(self):
        plot_dataset_statistics(self.df_gtbbox_metadata, self.results_dir)
