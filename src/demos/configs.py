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
    #"Adverse Weather": ({"weather": ["Partially cloudy"]}, {"weather": ["Foggy"]}),
    "Camera Angle": ({"pitch": {"<": -10}}, {"pitch": {">": -10}}),
}


ODD_limit = [
    ({"is_night": 1}, "Night"),
    ({"adverse_weather": 1}, "Adverse Weather"),
    ({"pitch": {"<":-10}}, "High-angle shot"),
]

# What is our area of performance ?
ODD_criterias = {
    "MR": 0.5,
    "FPPI": 5,
}

occl_thresh = [0.35, 0.8]
height_thresh = [20, 50, 120]


#%% GTbbox filtering parameters

gtbbox_filtering_all = {
    "Overall": {
        "occlusion_rate": {"<": 0.99},
        "height": {">": height_thresh[1]},
    },
}

gtbbox_filtering_height_cats = {
    "near": {
        "occlusion_rate": {"<": 0.01},
        "height": {">": height_thresh[2]},
    },

    "medium": {
        "occlusion_rate": {"<": 0.01},
        "height": {"between": (height_thresh[1], height_thresh[2])},
    },
}

"""
"far": {
    "occlusion_rate": (0.01, "max"),  # Unoccluded
    "height": (height_thresh[1], "max"),
    "height": (height_thresh[0], "min"),
},
"""

gtbbox_filtering_occlusion_cats = {
    "No occlusion": {
        "occlusion_rate": {"<": 0.01},
        "height": {"<": height_thresh[1]},
    },
    "Partial occlusion": {
        "occlusion_rate": (0.01, "min"),
        "occlusion_rate": (occl_thresh[0], "max"),
        "height": (height_thresh[1], "min"),
    },
    "Heavy occlusion": {
        "occlusion_rate": (0.01, "max"),  # Unoccluded
        "occlusion_rate": (occl_thresh[0], "min"),
        "occlusion_rate": (occl_thresh[1], "max"),
        "height": (height_thresh[1], "min"),
    },
}

gtbbox_filtering_aspectratio_cats = {
    "Typical aspect ratios": {
        "aspect_ratio_is_typical": 1,  #
        "occlusion_rate": (0.01, "max"),  # Unoccluded
    },
    "Atypical aspect ratios": {
        "aspect_ratio_is_typical": 0,  #
        "occlusion_rate": (0.01, "max"),  # Unoccluded
    },
}



gtbbox_filtering_cats = {}
gtbbox_filtering_cats.update(gtbbox_filtering_all)
gtbbox_filtering_cats.update(gtbbox_filtering_height_cats)
#gtbbox_filtering_cats.update(gtbbox_filtering_aspectratio_cats)
#gtbbox_filtering_cats.update(gtbbox_filtering_occlusion_cats)