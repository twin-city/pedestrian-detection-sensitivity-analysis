# Install & Datasets

## Install
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

## Datasets
- PennFudan
  - Isgroup : ignore region ? (if not it generates false positives because good detector find each people in the group instead of just 1)
- MoTSynth
  - ??
- ECP
- Twincity
  - ??
- Twincity
  - Hendled Glitch via ignore regions : too big bbox or multiple colors

## About filtering
typical / atypical above 50pix or not ?