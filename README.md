
# Pedestrian Detection Sensitivity Analysis

This project performs inference of baseline detection methods on datasets, both synthetic and real, and performs a sensitivity analysis on the results.

Given input detection models (RGB --> (x0,y0,x1,y1) it aims at :
- Analyzing their sensitivity toward parameters such as 
  - frame parameters (luminosity, weather, camera angle, camera distance)
  - boundig box parameters (height, occlusion)
- Provide an exhaustive benchmark on MoTSynth and EuroCityPerson datasets to make a proof of concept of the method, yielding empirical results indicating what works and what does not work in the synthetic dataset.

## How to use this project

Assuming each twincity folder is stored in `/home/raphael/work/datasets/twincity-Unreal/v5` :


Run `python main.py -d twincity -r /home/raphael/work/datasets/twincity-Unreal/v5 --max_samples 50 -output results_new -frame -gt --plot_image`


## Datasets used 

- Download motsynth
- Download eurocityperson (research purpose only) 

For more information on EuroCityPerson and its license see https://eurocity-dataset.tudelft.nl/  
For more information on MoTSynth and its license see https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=42  

