
TODO
- master avec MoTSynth et Twincity
- factoriser code
  - classe dataset
- ignore regions of twincity that bug (bbox too big or multiple colors)



Download motsynth
Download eurocityperson

Run `python motsynth_demo.py`
Run `python ECP_demo.py`

For more information on MoTSynth and its license see https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=42 
For more information on EuroCityPerson and its license see https://eurocity-dataset.tudelft.nl/ 


# Results

Study on a baseline (Faster-RCNN trained on Cityscapes) :

| Dataset                  | MoTSynth (Synthetic)                                                                            | EuroCityPerson (Real)                                                                                                                              |
|--------------------------|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------| 
| Img example              | <img src="results/motsynth_img.png" alt="Motsynth example" style="max-width: 150;">             |  |
| pval or correlation test | <img src="results/motsynth_pval.png" alt="Motsynth pval" style="max-width: 150;">               | <img src="results/ecp_pval.png" alt="Eurocityperson pval" style="max-width: 150;">                                                                 |
| Day vs Night             | <img src="results/motsynth_dayvsnight.png" alt="Motsynth day vs night" style="max-width: 150;"> | <img src="results/ecp_dayvsnight.png" alt="Eurocityperson pval" style="max-width: 150;">                                                           |


pval indicates statstical significance for night/day for both MR and FPPI, and weather for MR.

We plot an example of performance for the baseline, in either day or night scenes in both datasets. Performance are worse at night.



