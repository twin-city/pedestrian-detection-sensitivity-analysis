import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def compute_iou(box1, box2):
    """
    Function to compute Intersection over Union.

    Args:
    box1: [x1, y1, x2, y2] (Top left and bottom right coordinates of box 1)
    box2: [x1, y1, x2, y2] (Top left and bottom right coordinates of box 2)
    """
    # Calculate the (x1, y1, x2, y2) coordinates of the intersection of box1 and box2. Calculate its Area.
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the Area of box1 and box2
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the Union
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area

    return iou


# Example boxes
box1 = [1, 1, 4, 4]
box2 = [2, 3, 5, 5]

for box2 in ([2, 3, 5, 5],):

    iou = compute_iou(box1, box2)
    print(f"IoU: {iou}")

    # Visualization
    fig, ax = plt.subplots(1)
    ax.set_xlim([0, 8])
    ax.set_ylim([0, 8])

    # Draw the two boxes
    box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1], linewidth=2, edgecolor='g',
                                  facecolor='none')
    box2_rect = patches.Rectangle((box2[0], box2[1]), box2[2] - box2[0], box2[3] - box2[1], linewidth=2, edgecolor='b',
                                  facecolor='none')
    ax.add_patch(box1_rect)
    ax.add_patch(box2_rect)

    # Intersection area
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_rect = patches.Rectangle((inter_x1, inter_y1), inter_x2 - inter_x1, inter_y2 - inter_y1, linewidth=2,
                                   edgecolor='g', facecolor='g', alpha=0.5)
    ax.add_patch(inter_rect)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    # plt.text(4.2, 1, "Area = 9", c="r", size=20)
    # plt.text(5.2, 3, "Area = 6", c="b", size=20)





#%% Plot multiple boxes

import numpy as np

# Visualization

ax.set_xlim([0, 64])
ax.set_ylim([0, 64])

gt_bboxes = [
    [10, 10, 20, 40],
    [30, 10, 40, 40],
    [10, 45, 15, 60],
]

gt_bboxes_crowd = [
    [49, 42, 49 + 5, 42 + 15],
    [52, 48, 52 + 5, 48 + 15],
    [48, 45, 48 + 5, 45 + 15],
]

gt_bboxes_false_positive = [
    [48, 10, 53, 25],
    [56, 10, 61, 25],
    [11, 12, 22, 35],
]

pred_boxes = []
for i, box1 in enumerate(gt_bboxes + gt_bboxes_crowd + gt_bboxes_false_positive):
    pred_boxes.append(box1 + np.random.randint(-2, 2, size=(4)))


#%%

fig, ax = plt.subplots(1)
ax.set_xlim([0, 64])
ax.set_ylim([0, 64])



for box1 in gt_bboxes:
    box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1],
                                  linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(box1_rect)

for box1 in gt_bboxes_crowd:
    box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1],
                                  linewidth=2, edgecolor='y', facecolor='none')
    ax.add_patch(box1_rect)

# Draw prediction boxes
probas = [0.8, 0.99, 0.2, 0.1, 0.1, 0.1, 0.9, 0.1, 0.5]


for i, box1 in enumerate(pred_boxes):
    box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1],
                                  linewidth=2, edgecolor='b', facecolor='none', linestyle="--")
    ax.text(box1[2], box1[3]+1, probas[i], c="b", size=12)
    ax.add_patch(box1_rect)

# plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%%
import torch
from src.detection.metrics import compute_fp_missratio

pred_bbox = [
    {
    "boxes": torch.tensor(gt_bboxes + gt_bboxes_crowd + gt_bboxes_false_positive),
    "scores": torch.tensor(probas),
    }
]

target_bbox = [
    {
        "boxes": torch.tensor(gt_bboxes + gt_bboxes_crowd)
    }
]


threshold = 0.5
res = compute_fp_missratio(pred_bbox, target_bbox, threshold=threshold, excluded_gt=[3, 4, 5])


#%%

for threshold in [0.9, 0.8, 0.5]:
    res = compute_fp_missratio(pred_bbox, target_bbox, threshold=threshold, excluded_gt=[3, 4, 5])

    fig, ax = plt.subplots(1)
    ax.set_xlim([0, 64])
    ax.set_ylim([0, 64])

    for i, box1 in enumerate(pred_boxes):
        if probas[i] >= threshold:
            box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1],
                                          linewidth=2, edgecolor='b', facecolor='none', linestyle="-")
        else:
            box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1],
                                          linewidth=2, edgecolor='b',
                                          facecolor='none', linestyle="--", alpha=0.5)
        ax.text(box1[2], box1[3]+1, probas[i], c="b", size=12)
        ax.add_patch(box1_rect)

        for box1 in gt_bboxes_crowd:
            box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1],
                                          linewidth=2, edgecolor='y', facecolor='none')
            ax.add_patch(box1_rect)

    for i, box1 in enumerate(gt_bboxes):
        if i in res["true_positives"]:
            box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1],
                                          linewidth=2, edgecolor='lime', facecolor='none')
        else:
            box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1],
                                          linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(box1_rect)



    # plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"data/tuto_detection/fp_mr_example_threshold_{threshold}.png")
    plt.show()


#%%

#%%



fppis = []
mrs = []

thresholds = np.concatenate(([1], np.sort(np.unique(probas))[::-1]))

for threshold in thresholds:
    res = compute_fp_missratio(pred_bbox, target_bbox, threshold=threshold, excluded_gt=[3, 4, 5])
    fppi, mr = res["num_false_positives"], res["missing_rate"]
    fppis.append(fppi)
    mrs.append(mr)

#%%
fig, ax = plt.subplots(1,1)
ax.plot(fppis, mrs)
for threshold, mr, fppi in zip(thresholds, mrs, fppis):
    ax.scatter(fppi, mr, c="black")
    ax.text(fppi+0.1, mr+0.01, f"{threshold}", size=15)
plt.tight_layout()
plt.savefig(f"data/tuto_detection/matching_curve_fppi_mr.png")
plt.show()



#%% Do the plot multiple times







#%%




#%% Plot a stick person



def plot_stick_figure(ax, xmin, xmax, ymin, ymax):


    # Define the relative size and position of the head, body, arms and legs
    head_radius = 0.1 * (ymax - ymin)
    head_center = ((xmin + xmax) / 2, ymin + 0.6 * (ymax - ymin))

    body_top = head_center[1] - head_radius
    body_bottom = ymin + 0.2 * (ymax - ymin)

    arm_left = xmin + 0.3 * (xmax - xmin)
    arm_right = xmin + 0.7 * (xmax - xmin)
    arm_height = ymin + 0.4 * (ymax - ymin)

    leg_left = xmin + 0.3 * (xmax - xmin)
    leg_right = xmin + 0.7 * (xmax - xmin)
    leg_bottom = ymin

    # Draw the head
    head = plt.Circle(head_center, head_radius, fill=False)
    ax.add_patch(head)

    # Draw the body
    ax.plot([head_center[0], head_center[0]], [body_top, body_bottom], color='black')

    # Draw the arms
    ax.plot([arm_left, arm_right], [arm_height, arm_height], color='black')

    # Draw the legs
    ax.plot([head_center[0], leg_left], [body_bottom, leg_bottom], color='black')
    ax.plot([head_center[0], leg_right], [body_bottom, leg_bottom], color='black')

    # Set limits and aspect ratio
    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)
    #ax.set_aspect('equal')



# Call the function with bounding box values
# plot_stick_figure(0.2, 0.8, 0, 1)

