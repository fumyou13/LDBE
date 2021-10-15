# LDBE
Pytorch implementation for two papers:

"Domain Adaptive Semantic Segmentation without Source Data", ACM MM2021. Paper link: [http://arxiv.org/abs/2110.06484](http://arxiv.org/abs/2110.06484)

"Challenging Source-free Domain Adaptive Semantic Segmentation", submitted to TPAMI.

## Method
![](img/main1.png)

## Result
GTA5 -> Cityscapes:

|  Methods| Source-only | LD | LDBE |
|  ----  | ----  |----|----|
| mIoU | 35.7 | 45.5 | 49.2 |

SYNTHIA -> Cityscapes:

|  Methods   | Source-only | LD | LDBE |
|  ----  | ----  |----|----|
| mIoU (16-classes)  | 32.5 | 42.6 | 43.5 |
| mIoU (13-classes)  | 37.6 | 50.1 | 51.1 |


![](img/visual.png)

## Data

Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/).

Download [SYNTHIA](http://synthia-dataset.net/). Please use SYNTHIA-RAND-CITYSCAPES

Download [Cityscapes](https://www.cityscapes-dataset.com/).

Make sure the data path is consistent with the path in config file.


## Training

Stage 0: Training on the source domain data.

Run "run_so.py". The trained model is available at [google drive](https://drive.google.com/file/d/1nXQS_4nd9zgsSELzhiV_WgihWaNvP6_5/view?usp=sharing).

Stage 1: Label denoising (both positive learning and negative learning).

Set method:"ld" in config/ldbe_config.yml. Then, run "run.py". The trained model is available at [google drive](https://drive.google.com/file/d/10iYWhgrxJHNl2vR-u2N3-uVRXH7qGPfZ/view?usp=sharing).

Stage 2: Boundary enhancement

Set method:"be" in config/ldbe_config.yml. Then, run "run.py". The trained model is available at [google drive](https://drive.google.com/file/d/1eFgqnPRIiivtTPCk6AihKjaeduaIyu-a/view?usp=sharing).

## Citation
If you find our repository is helpful, please consider citing our paper
      @inproceedings{you2021domain,
            title = {Domain Adaptive Semantic Segmentation without Source Data}, 
            author = {Fuming You and Jingjing Li and Lei Zhu and Ke Lu and Zhi Chen and Zi Huang},
            booktitle = {ACM Multimedia}, 
            year = {2021}
      }
## Acknowledgement

[https://github.com/Solacex/CCM](https://github.com/Solacex/CCM)

[https://github.com/yzou2/CRST](https://github.com/yzou2/CRST)

## Contact
fumyou13@gmail.com
