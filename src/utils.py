import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.detection.metrics import compute_fp_missratio2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
import matplotlib.patches as patches
import os.path as osp
from src.detection.metrics import compute_model_metrics_on_dataset



#todo get pval also same same

def get_linear_importance(df, metric, features):

    X = df[features]
    X = (X-X.mean())/X.std()
    y = df[metric]
    X_train, X_test = X[:len(X)//2], X[len(X)//2:]
    y_train, y_test = y[:len(y)//2], y[len(y)//2:]



    model = RidgeCV()
    model.fit(X_train, y_train)
    print(f'{metric} model score on training data: {model.score(X_train, y_train)}')
    print(f'{metric} model score on testing data: {model.score(X_test, y_test)}')

    coefs = pd.DataFrame(
       model.coef_,
       columns=['Coefficients'], index=X_train.columns
    )

    return coefs.values[:,0]



def get_permuation_importance(df, metric, features):

    X = df[features]
    X = (X-X.mean())/X.std()
    y = df[metric]
    X_train, X_test = X[:len(X)//2], X[len(X)//2:]
    y_train, y_test = y[:len(y)//2], y[len(y)//2:]

    forest = RandomForestRegressor(random_state=0)
    forest.fit(X_train, y_train)
    print(f'{metric} model score on training data: {forest.score(X_train, y_train)}')
    print(f'{metric} model score on testing data: {forest.score(X_test, y_test)}')

    # Compute importance
    result = permutation_importance(forest, X, y, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=features)

    return forest_importances.values


def filter_gt_bboxes(df, gt_bbox_filtering):
    df_subset = subset_dataframe(df, gt_bbox_filtering)
    excluded_gt = list(set(range(len(df))) - set(df_subset.index))
    return excluded_gt

def subset_dataframe(df, conditions):
    """

    :param df:
    :param conditions:
    :return:
    """
    # Create an empty mask
    mask = pd.Series([True] * len(df), index=df.index)

    # Iterate over each condition in the dictionary and update the mask accordingly
    for column, values in conditions.items():

        if column not in df.columns:
            print(f"Warning, column {column} was not found in dataframe to subset, disgarding.")
            new_mask = df.iloc[:,0] == df.iloc[:,0] #todo ugly hack
        else:
            if isinstance(values, dict):
                if '>' in values:
                    new_mask = df[column] >= values['>']
                if '<' in values:
                    new_mask = df[column] <= values['<']
                if 'value' in values:
                    new_mask = df[column] == values['value']
                if 'set_values' in values:
                    new_mask = df[column].isin(values['set_values'])
            elif isinstance(values, (list, set, np.ndarray)):
                new_mask = df[column].isin(values)
            elif isinstance(values, (int, float, str)):
                new_mask = df[column] == values
            else:
                raise NotImplementedError("The subset condition was not set according to subset_dataframe inputs.")

        if new_mask.sum() == 0:
            print(f"Warning, the condition {column} did not change the subset.")

        mask &= new_mask

    # Apply the mask to the DataFrame to get the subset
    subset_df = df[mask]

    if len(conditions) > 0 & len(subset_df) == len(df):
        print("Warning : filtering did not change the dataframe size")

    return subset_df






def compute_correlations(df, features):
    corr_matrix = df[features].corr(
        method=lambda x, y: pearsonr(x, y)[0])
    p_matrix = df[features].corr(
        method=lambda x, y: pearsonr(x, y)[1])
    return corr_matrix, p_matrix




#%% Plot utils


def target_2_torch(targets):
    return {key: [{
        "boxes": torch.tensor(val[0]["boxes"]),
        "labels": torch.tensor(val[0]["labels"]),
    }
    ] for key, val in targets.items()}



def target_2_json(targets):
    return {key: [{
        "boxes": val[0]["boxes"].numpy().tolist(),
        "labels": val[0]["labels"].numpy().tolist(),
    }
    ] for key, val in targets.items()}

import matplotlib.patches as patches




def compute_models_metrics_from_gtbbox_criteria(dataset_name, dataset, df_frame_metadata, gtbbox_filtering_cats, model_names):

    # Compute for multiple criteria
    df_metrics_criteria_list = []
    for key, val in gtbbox_filtering_cats.items():
        df_results_aspectratio = pd.concat(
            [compute_model_metrics_on_dataset(model_name, dataset_name, dataset, val, device="cuda")[0] for
             model_name in model_names])
        df_results_aspectratio["gtbbox_filtering_cat"] = key
        df_metrics_criteria_list.append(df_results_aspectratio)
    df_metrics_criteria = pd.concat(df_metrics_criteria_list, axis=0)
    df_analysis = pd.merge(df_metrics_criteria.reset_index(), df_frame_metadata, on="frame_id")

    return df_analysis