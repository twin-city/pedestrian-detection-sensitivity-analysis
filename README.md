
# Pedestrian Detection Sensitivity Analysis

This project performs inference of baseline detection methods on datasets, both synthetic and real, and performs a sensitivity analysis on the results.

Given input detection models (RGB --> (x0,y0,x1,y1) it aims at :
- Analyzing their sensitivity toward parameters such as 
  - frame parameters (luminosity, weather, camera angle, camera distance)
  - boundig box parameters (height, occlusion)
- Provide an exhaustive benchmark on MoTSynth and EuroCityPerson datasets to make a proof of concept of the method, yielding empirical results indicating what works and what does not work in the synthetic dataset.

## How to use this project

Run `python src/demos/demo_benchmark_analysis.py`


## Datasets used 

dataset named as : ($name_$max samples)

| characteristics       | motsynth_600                                                              | twincity_50            | EuroCityPerson_30   |
|:----------------------|:--------------------------------------------------------------------------|:-----------------------|:--------------------|
| sequences (day/night) | 24/10                                                                     | 2/2                    | 31/7                |
| images (day/night)    | 404/170                                                                   | 70/100                 | 826/164             |
| person (day/night)    | 12096/5365                                                                | 1378/1835              | 5604/1182           |
| weather               | THUNDER, CLEAR, SMOG, EXTRASUNNY, FOGGY, CLOUDS, BLIZZARD, RAIN, OVERCAST | Partially cloudy, Snow | dry, rainy          |


- Download motsynth
- Download eurocityperson (research purpose only) 

For more information on EuroCityPerson and its license see https://eurocity-dataset.tudelft.nl/
For more information on MoTSynth and its license see https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=42

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

