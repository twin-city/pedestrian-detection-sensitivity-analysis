param_heatmap_metrics = {
    "MR": {
        "vmin": -0.15,
        "vmax": 0.2,
        "center": 0.,
        "cmap": "RdYlGn_r",
    },
    "FPPI": {
        "vmin": -0.15,
        "vmax": 0.2,
        "center": 0.,
        "cmap": "RdYlGn_r",
    },
}

metrics = ["MR", "FPPI"]

# What cases we study with performance difference

dict_filter_frames = {
    "Overall": [{}],
    "Day / Night": ({"is_night": 0}, {"is_night": 1}),
    # "Adverse Weather": ({"weather": ["dry"]}, {"weather": ["rainy"]}),
    "Camera Angle": ({"pitch": {"<": -10}}, {"pitch": {">": -10}}),
}


ODD_limit = [
    ({"is_night": 1}, "Night"),
    #({"adverse_weather": 1}, "Adverse Weather"),
    ({"pitch": {"<":-10}}, "High-angle shot"),
]

# What is our area of performance ?
ODD_criterias = None #  = {"MR": 0.5, "FPPI": 5,}

occl_thresh = [0.35, 0.8]
height_thresh = [20, 50, 120]

#%% GTbbox filtering parameters

gtbbox_filtering_all = {
    "Overall": {
        "ignore_region": 0,
        "occlusion_rate": {"<": 0.99},
        "height": {">": height_thresh[1]},
    },
}

gtbbox_filtering_height_cats = {
    "near": {
        "ignore_region": 0,
        "occlusion_rate": {"<": 0.01},
        "height": {">": height_thresh[2]},
    },

    "medium": {
        "ignore_region": 0,
        "occlusion_rate": {"<": 0.01},
        "height": {"between": (height_thresh[1], height_thresh[2])},
    },

    "far": {
        "ignore_region": 0,
        "occlusion_rate": {"<": 0.01},
        "height": {"between": (height_thresh[0], height_thresh[1])},
    },
}




gtbbox_filtering_occlusion_cats = {
    "No occlusion": {
        "ignore_region": 0,
        "occlusion_rate": {"<": 0.01},
        "height": {">": height_thresh[1]},
    },
    "Partial occlusion": {
        "ignore_region": 0,
        "occlusion_rate": {"between": (0.01, occl_thresh[0])},
        "height": {">": height_thresh[1]},
    },
    "Heavy occlusion": {
        "ignore_region": 0,
        "occlusion_rate": {"between": (occl_thresh[0], occl_thresh[1])},
        "height": {">": height_thresh[1]},
    },
}

gtbbox_filtering_aspectratio_cats = {
    "Typical aspect ratios": {
        "ignore_region": 0,
        "aspect_ratio_is_typical": 1,  #
        "occlusion_rate": {"<": 0.01},
    },
    "Atypical aspect ratios": {
        "ignore_region": 0,
        "aspect_ratio_is_typical": 0,  #
        "occlusion_rate": {"<": 0.01},
    },
}



gtbbox_filtering_cats = {}
gtbbox_filtering_cats.update(gtbbox_filtering_all)
gtbbox_filtering_cats.update(gtbbox_filtering_height_cats)
gtbbox_filtering_cats.update(gtbbox_filtering_aspectratio_cats)
gtbbox_filtering_cats.update(gtbbox_filtering_occlusion_cats)

