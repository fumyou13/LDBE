# LDBE
Pytorch implementation for two papers (the paper will be released soon):

"Domain Adaptive Semantic Segmentation without Source Data", ACM MM2021.

"Challenging Source-free Domain Adaptive Semantic Segmentation", submitted to TPAMI.

## Abstact 
Domain adaptive semantic segmentation is recognized as a promising technique to alleviate the domain shift between the
labeled source domain and the unlabeled target domain in many real-world applications, such as automatic pilot. However, large
amounts of source domain data often introduce significant costs in storage and training, and sometimes the source data is inaccessible
due to privacy policies. To address these problems, we investigate Source-Free domain adaptive Semantic Segmentation (SFSS),
which assumes that the model is pre-trained on the source domain, and then adapted to the target domain without accessing source
data anymore. Based on extensive experiments and re-implementations of conventional self-training methods and recent source-free
domain adaptation methods, we first summarize three challenges of SFSS: learning from numerous noises, preventing the
“winner-takes-all” and boundary confusion. Then, we propose a two-stage framework including Label Denoising and Boundary
Enhancement (LDBE) to address it. In stage 1, we propose two effective and mutually reinforcing components: positive learning and
negative learning to perform label denoising from two perspectives, significantly removing numerous noises and prevents the
“winner-takes-all” dilemma. Then, we analyze the bottleneck of stage 1 training, and propose a novel data-augmentation method to
further enhance the boundary between each category in stage 2. Notably, our framework LDBE can be easily implemented and
incorporated with other methods to further enhance their performance. Extensive experiments on widely-used synthetic-to-real
benchmarks demonstrate our claims and the effectiveness of our framework, which achieves state-of-the-art performance and
outperforms the baseline with a large margin (+13.5% mIoU in GTA5 → Cityscapes and +13.5% mIoU in SYNTHIA → Cityscapes).
Even compared to the latest methods accessing the large amounts of source domain data, LDBE also yields competitive performance.

## Result
GTA5 -> Cityscapes:

|  Methods| Source-only | LD | LDBE |
| mIoU | 35.7 | 45.5 |49.2 |

SYNTHIA -> Cityscapes:

|  Methods   | Source-only | LD | LDBE |
| mIoU (16-classes)  | 32.5 | 42.6 | 43.5 |
| mIoU (13-classes)  | 37.6 | 50.1 | 51.1 |

## Data

Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/).

Download [SYNTHIA](http://synthia-dataset.net/). Please use SYNTHIA-RAND-CITYSCAPES

Download [Cityscapes](https://www.cityscapes-dataset.com/).

Make sure the data path is consistent with the path in config file.


## Training

Stage 0: Training on the source domain data.

Run "run_so.py". The trained model is available at ...

Stage 1: Label denoising (both positive learning and negative learning).

Set method:"ld" in config/ldbe_config.yml. Then, run "run.py". The trained model is available at ...

Stage 2: Boundary enhancement

Set method:"be" in config/ldbe_config.yml. Then, run "run.py". The trained model is available at ...

## Acknowledgement

[https://github.com/Solacex/CCM](https://github.com/Solacex/CCM)

[https://github.com/yzou2/CRST](https://github.com/yzou2/CRST)
