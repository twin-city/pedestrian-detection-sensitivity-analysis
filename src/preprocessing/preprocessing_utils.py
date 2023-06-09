import torch

def syntax_occl_ECP(x):
    # keep only occluded
    x = [x for x in x if "occluded" in x]
    if x and "occluded" in x[0]:
        return int(x[0].replace("occluded>", "")) / 100
    elif x:
        print(x)
    else:
        return 0

def syntax_truncated_ECP(x):
    # keep only truncated
    x = [x for x in x if "truncated" in x]
    if x and "truncated" in x[0]:
        return int(x[0].replace("truncated>", "")) / 100
    elif x:
        print(x)
    else:
        return 0


def syntax_criteria_ECP(x, criteria):
    # "sitting-lying"
    # keep only truncated
    x = [x for x in x if criteria in x]
    if x and criteria in x[0]:
        return 1
    elif x:
        print(x, criteria)
    else:
        return 0


def dftarget_2_torch(df):
    targets = df.groupby("frame_id").apply(
        lambda x: x[["x0", "y0", "x1", "y1"]].values).to_dict()
    targets_torch = {}
    for key, val in targets.items():
        # Target and labels
        target = [dict(boxes=torch.tensor(val))]
        target[0]["labels"] = torch.tensor([0] * len(target[0]["boxes"]))
        targets_torch[key] = target
    return targets_torch

def get_scalar_dict(dict, dict_frame_to_scalar):
    scalar_dict = {}
    for key, values in dict.items():
        if key in dict_frame_to_scalar.keys():
            for i, value in enumerate(values):
                scalar_dict[f"{dict_frame_to_scalar[key][i]}"] = value
        else:
            scalar_dict[key] = values
    return scalar_dict


def flatten_list(l):
    return [item for sublist in l for item in sublist]