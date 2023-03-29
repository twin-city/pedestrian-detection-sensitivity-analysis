

Bug
- gpu inference does not work mmdet (problem CUDA error, due to conda install ?)


TODO motsynth
#todo assumes idex and frame_id are same same
#todo : occluded body joints
#todo city and time in metadata also, then set filters when we want to compare ? But apriori compare on all, no need per city ?
#todo balance may be needed though ...
#todo take subset of all the images nto be not too long ?

# Objective : show it works same-same in pedestrian detection task

Models to try : 

- Faster RCNN
  - Cityscapes
  - Caltech Pedestrian
  - NightOwls
- Mask RCNN 
  - MoTSynth (bonus)
- SoTA
  - SSD ?

Datasets 

- ECP
- MoTSynth
- (Caltech ?)
- (Synthia ?)

Parameters to try

- Weather
- Day / Night
- BBox size 
- BBox difficulty
- Occlusions







Code and reasonable !!! https://eurocity-dataset.tudelft.nl/eval/benchmarks/detection






TODO
- Get a forked mmdet with added datasets (+ add the dataset description such as day and night ?)
- Resultats contre intuitifs : il faut check le mAP !!!!! Commencer par la.
  - Sinon c'est que je me suis perdu dans trop de paramètres des datasets ... Reste la méteo ... Et aussi par bbox
- Too long for inference with a network : how to speed up ? GPU ? Or Pickle at once ?
- NMS ? Check analyze_per_img_dets
- Far scale : ça ça marche !!!! (mais est-ce interessant pour nous ? Oui quand même pour définir une zone où ça marche quoi)
- (après a voir si les solutions du privé ont déjà des zones définies ? Ce qui serait plus malin comme specs)


TODO Later
- relative package import such as mmdet ?
- Est-ce okay avec la proba ?
- did I do NMS ? Or is it to do yet ?

Caltech Dataset
Downloaded at ???

CUHK was downloaded

other ideas : 
https://dbcollection.readthedocs.io/en/latest/datasets/inria_ped.html
