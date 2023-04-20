
# Must-have

Analysis at the box level
- BBox size / distance
- BBox difficulty (what is it exactly ? Reasonable ?)
- Occlusions

Tests
  - test fonctionels généraux
  - tests unitaires

Metrics
  - mAP

Features ? DebiAI for Visualization

# Nice-to-have

Statistical Tests to uncover biases / detect significance

New metrics (seeCode and reasonable !!! https://eurocity-dataset.tudelft.nl/eval/benchmarks/detection)

MotSynth bonuses per bbox
  - per person attribute

Compare dataset characteristics (anticipate --> so as to identify biases between datasets from their characteristics in advance !!!)

Other Datasets
  - Synscapes
  - Cityscapes Fog/Rain

Detailed vizualization
  - difficult cases
  - specific parameter cases

More metrics
    - cf EuroCityPerson


# Later-to-have

Faster
  - FPPI computation

Detailed analysis 
  - Check reasons for false positives ???? what class is predcited instead ? (synthétique ou alors faire passer un detecteur d'objet)

Uncover novel biases ? With all parameters without filtering ? Or even without knowing the parameters in advance ?

Trying other model
  - Faster-RCNN (NightOwls, Caltech Pedestrian, Cityscapes-Rain/Fog/Sun, CUHK)
  - Mask-RCNN
  - SoTA ??? SSD ??? Pedestron ???

# Reflexion / Biblio
Check ODD principles
Variety for independance would be nice !
Motsynth : choice occlusion is computed from keypoints
other ideas : https://dbcollection.readthedocs.io/en/latest/datasets/inria_ped.html
Checker si les solutions du privé donnent déjà des specs
Get a forked mmdet with added datasets (+ add the dataset description such as day and night ?)
- NMS ?

# Install
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
torchmetrics
tqdm
pandas
seaborn
conda install -c conda-forge opencv
This post did help https://discuss.pytorch.org/t/userwarning-cuda-initialization-cuda-unknown-error-this-may-be-due-to-an-incorrectly-set-up-environment-e-g-changing-env-variable-cuda-visible-devices-after-program-start-setting-the-available-devices-to-be-zero/129335/2

scikit-learn

cpu conda install pytorch torchvision torchaudio -c pytorch

See mmdet / mmcv for install
