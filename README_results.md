
# Pedestrian Detection Sensitivity Analysis : An experiment to explore the usefullness of synthetic data for evaluating pedestrian detection

## Introduction

Low luminosity, adverse weather conditions, occlusion, camera angle, etc. are all factors that can affect the performance 
of pedestrian detection methods. To what extent ? Also, to what extent can an AI be discriminative ?

In this work, we intend to tackle these issues by taking a step back, 
and leverage synthetic data. Therefore we aim first at answering to the question : 

**Can we use synthetic data to evaluate pedestrian detection methods ?**  
We seek to answer this question at 2 levels : 
- Using SoTA Synthetic Dataset : is it doable in theory ?
- Using a custom synthetic dataset : is it doable in practice ?

**Why do we intend to answer the feasability and usefullness of building a custom synthetic dataset ?**  
Because if successfull :
- It would allow us to build datasets that are **tailored to our needs** (e.g. for the Ministry of Interior use-cases, such as image blurring), which is often 
hard to find in real datasets publicly available when working on niche applications.
- It would allow us to **control the characteristics of the data** (e.g. occlusion, weather, camera angle, etc.) 
- It would allow to **centralize on a common dataset** the evaluation of pedestrian detection methods . 
It is very hard to find a publicly available dataset that correspond both to the desired use-case, 
as well as having the needed level of exhaustivity. For example EuroCityPerson and NightOwls aim at exhaustivity on day/night as well
As weather conditions, but the camera angle is always the same due to their Autonomous Driving application.
- Compared to real data, it is **easier to generate** (e.g. no need to label the data, no need to find a location, etc.), and 
do not require any **privacy concern** (e.g. blurring faces, license plates, etc.).

In addition, there are other potential benefits : 
 - Possible **synergies for comptuter vision dataset building** (especially in 3D assets) 
 - A fully controlled dataset could be used to **exhibit potential biases** of AI solutions, such as discriminations that are 
explicitely pointed out in the [EU AI White Paper](https://commission.europa.eu/publications/white-paper-artificial-intelligence-european-approach-excellence-and-trust_en) as well 
as in the [JO 2024 Project of Law](https://www.assemblee-nationale.fr/dyn/16/textes/l16b0809_projet-loi#).

## References

Real datasets :

- **Pedestrian detection: A benchmark (2009)** : perform an exhaustive evaluation of pedestrian detection (PD) SoTA methods on a common dataset : the Caltech Dataset.

- **EuroCityPerson (2018)** : perform evaluation of PD SoTA methods on the Autonomous Driving (AD) EuroCityPerson dataset that focus on detection in various European urban environments, with both
day and night, as well as multiple weather conditions. They also provide a benchmark in the fashion of **Pedestrian detection: A benchmark (2009)** that 
exhibit significative differences in performance between day and night, city location, and between weather conditions.


Synthetic datasets :

- **MoTSynth (2020)** : perform evaluation of PD SoTA methods on the MoTSynth dataset that focus on detection and tracking in GTAV (Grand Theft Auto V) game engine.
It includes various weather conditions, day and night, cameras angle, and various occlusion levels. In the 2020 paper the focus is of leveraging MoTSynth for pre-training. 
In this work, we explore the use of MoTSynth for evaluating PD methods.


Additional  datasets : (not used in this work)

- **KITTI & Virtual KITTI (Virtual Worlds as Proxy for Multi-Object Tracking Analysis  2016)** : perform evaluation of PD SoTA methods on the Autonomous Driving (AD) KITTI dataset that focus on detection in urban 
environments, with both day and night, as well as multiple weather conditions. Virtual KITTI is designed to be a synthetic version of KITTI, all 
other things being equal. They "provide quantitative experimental evidence suggesting that 

> *(i) modern deep learning algorithms pre-trained on real data behave similarly in real and virtual worlds, 
and (ii) pre-training on virtual data improves performance. As the gap between real and virtual worlds is small,
virtual worlds enable measuring the impact of various weather and imaging conditions on recognition performance,
all other things being equal. We show these factors may affect drastically otherwise high-performing deep models for tracking."* 


- **NightOwls (2020)** : perform evaluation of PD SoTA methods on the Autonompus Driving (AD) NightOwls dataset that focus on detection at night.
Detection at night is a challenging task because of the low light conditions and saturation.





## Benchmark

We perforl a benchmark in the spirit of **Pedestrian detection: A benchmark (2009)**, but with a focus on the use of synthetic data.
We use MoTSynth, SoTA on synthetic data for pedestrian detection, as well as our custom Twincity dataset and compare it 
to real data via the EurocityPerson dataset.

### Dataset description

Small version of dataset named as : ($name_small_$max samples) indicate a subsampling of the full dataset for faster prototyping, but shall 
eventually be replaced by the full dataset.

| characteristics       | ecp_small_30   | motsynth_small_30                                                      | PennFudanPed_200   | Twincity-Unreal-v5_30   |
|:----------------------|:---------------|:-----------------------------------------------------------------------|:-------------------|:------------------------|
| sequences (day/night) | 31/7           | 24/10                                                                  | 1/                 | 2/2                     |
| images (day/night)    | 285/53         | 237/100                                                                | 170/               | 50/60                   |
| person (day/night)    | 1963/359       | 6966/3138                                                              | 423/               | 1084/1156               |
| weather               | dry, rainy     | thunder, clear, smog, extrasunny, foggy, clouds, snow, rainy, overcast | dry                | clouds, snow            |
| height                | 96 +- (57)     | 146 +- (91)                                                            | 273 +- (35)        | 146 +- (20)             |
| occlusion rate        | 0.14 +- (0.15) | 0.46 +- (0.21)                                                         | 0.00 +- (0.00)     | 0.00 +- (0.00)          |
| aspect ratio          | 0.38 +- (0.07) | 0.45 +- (0.14)                                                         | 0.40 +- (0.08)     | inf +- (nan)            |


### Experimental protocol

We take 2 baseline models from the MMDET model zoo : 
 - Faster R-CNN R50-FPN 1x trained on Cityscapes (https://github.com/open-mmlab/mmdetection/tree/master/configs/cityscapes)
 - Mask R-CNN R50-FPN 1x trained on COCO (https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn)

We use the same experimental protocol as **Pedestrian detection: A benchmark (2009)** and compute the MR (Miss Rate) vs FPPI (False Positive Per Image) for each method and each dataset. We also compute the MR difference between specific scenario and average. We use the same metrics as **Pedestrian detection: A benchmark (2009)** : 
 - MR (Miss Rate) : the percentage of pedestrians that are not detected
 - FPPI (False Positive Per Image) : the number of false positives per image

We compare different scenarios, and for each compute the MR (Miss Rate) vs FPPI (False Positive Per Image) curves according to 
 - Bounding Box parameters (as in **Pedestrian detection: A benchmark (2009)**) : aspect ratio, height, occlusion rate.
 - Image parameters : day vs night, camera angle, weather.


### Results

|                                                                                                                             | **Twincity (Ours, Synthetic)**                                                                                                                                                                             | MoTSynth (Synthetic)                                                                        | EuroCityPerson (Real)                                                                              |
|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Boundig Box parameter sensitivity : MR vs FPPI for bbox aspect ratio (line 1), height (line 2) and occlusion rate (line 3). | ![gtbbox_cofactor_fppi_mr.png](reference_results%2FTwincity-Unreal-v5_30%2Fgtbbox_cofactor_fppi_mr.png)                                                                                                               | ![gtbbox_cofactor_fppi_mr.png](reference_results%2Fmotsynth_small_30%2Fgtbbox_cofactor_fppi_mr.png) | ![gtbbox_cofactor_fppi_mr.png](reference_results%2Fecp_small_30%2Fgtbbox_cofactor_fppi_mr.png) |
| Frame parameter sensitivity :  MR vs FPPI for Day vs Night (line 2) and Camera Angle (line 3).                              | ![Image 1](reference_results/Twincity-Unreal-v5_30/frame_cofactor_fppi_mr.png)                                                                                                                             | ![frame_cofactor_fppi_mr.png](reference_results%2Fmotsynth_small_30%2Fframe_cofactor_fppi_mr.png) | ![frame_cofactor_fppi_mr.png](reference_results%2Fecp_small_30%2Fframe_cofactor_fppi_mr.png)   |

> To read the plots, the lower the better. 
> The x-axis is the FPPI (False Positive Per Image) and the y-axis is the MR (Miss Rate). 
> The curves are the MR vs FPPI for each scenario. The red curve is the average MR vs FPPI
> for all scenarios. The blue curve is the MR vs FPPI for the specific scenario. 
> The difference between the red and blue curves is the MR difference between the specific 
> scenario and the average.

First focusing on MoTSynth vs EurocityPerson : 
- Performance appear consistent between the 2 datasets for Bounding Box parameters (Aspect ratio, height, occlusion rate).
- Night is harder than Day for the faster-RCNN trained on Cityscapes in both datasets, which could
be explained by the fact that Cityscapes is a day dataset.
- Night abd Day are similar for the Mask-RCNN trained on COCO in both datasets. This could be explained
by the fact that COCO incorporates night images.
- On MoTSynth, pitch angle affects the performance of the Faster-RCNN trained on Cityscapes, but not on Mask-RCNN. 

And looking at our custom Twincity Dataset : 

> Note that the annotation in Twincity is for now different than those of other datasets. The available
> bounding boxes are extracted from an instance segmentation of the image, and therefore do not capture the
> occlusion information, which explains the lower performance overall.

- Night is harder than Day
- Camera Angle ??? (TODO)


### Conclusion

 - Overall comparison between MoTSynth and EuroCityPerson show that model behaviours
are consistent between Synthetic and Real datasets (provided datasets are similar enough 
modulo their synthetic or real nature). While the effect of weather is less clear, this is especially the case for 
   - Bounding Box parameters
   - Night and Day
   - Camera Angle ???
 - There are still significative performance difference between Twincity and the 2 other datasets. We explain
this difference via the yet different annotations, and hope to adapt it in the future to make it more consistent.
Modulo this difference in performance, the behaviours of methods in Twincity seem consistent to those 
in the other 2 datasets.


Take Home Message

> Synthetic datasets are a good alternative to real datasets for object detection, provided all other things than their synthetic/real nature are equal.

> The use of synthetic dataset allowed to go further than the best real dataset we found (EuroCityPerson)
> by incorporating parameters not found in real datasets (camera angle in MoTSynth), and possibly by exploring
> specific use-cases (e.g. fall detection, weapon detection, pedestrian flux estimation, etc.) via our custom dataset Twincity.
 





Synthetic can perform similarly with Real 
 - 
 - 
 - Still work to do on annotations for Twincity which differs from the 2 other datasets (see Twincity column). Effect are similar, but range of values differ.
 - On frame parameter sensitivity
   
   - Camera angle is not consistent across the synthetic datasets : 0° is harder than 45° for mask-RCNN_coco, but effect differ between Twincity and MoTSynth for FasterRCNN_Cityscapes.

To go further
- Due to lack of time, the comparison done here is qualitative and rely on 
comparing the MR vs FPPI curves. In order to perform a more quantitative comparison, we could analyse more
datasets and methods, and perform statistical tests to compare the behaviors in the different scenarios.
(e.g. same  ranking of methods in either real of synthetic version of the same scenario).

- Add more datasets (KITTI, Cityscapes, etc.).

- Implement more specific use-cases, such as fall detection, weapon detection, of pedestrian flux estimation. 