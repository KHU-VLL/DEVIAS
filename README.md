# [DEVIAS: Learning Disentangled Video Representations of Action and Scene](https://arxiv.org/abs/2312.00826)

[![arXiv](https://img.shields.io/badge/arXiv-2312.00826-red)](https://arxiv.org/abs/2312.00826)

This repository is the official implementation of the paper **"DEVIAS: Learning Disentangled Video Representations of Action and Scene"**, accepted as an **Oral** presentation at ECCV 2024🔥🔥.

## News
- We are diligently preparing the code and are excited to share it with you soon. Stay tuned for the release soon!
  
## Installation

Please prepare the environment following [INSTALL.md](docs/INSTALL.md).

## Dataset

The following datasets are used in this project. You can download them via the provided links.

1. **[Kinetics-400](https://opendatalab.com/OpenMMLab/Kinetics-400)** , **[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)** , **[HVU](https://github.com/holistic-video-understanding/HVU-Downloader)**
2. **[SCUBA (ICCV2023)](https://github.com/lihaoxin05/StillMix)**, **[HAT(Neurips2022)](https://github.com/princetonvisualai/HAT)**
3. **[Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something)**, **[ActivityNet](http://activity-net.org/download.html)**, **[Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html)**

Please download the datasets to replicate the results of this project. For the detail setting, please see [DATASET.md](docs/DATASET.md).

## Training
The instruction for training is in [TRAIN.md](docs/TRAIN.md).

## Evaluation
We evaluate DEVIAS based on **action** and **scene** recognition performances across both **seen** and **unseen** action-scene combination scenarios.
The instruction for evaluation is in [EVAL.md](docs/EVAL.md).

## Downstream Experiments
We find the disentangled action and scene representation of DEVIAS is beneficial for various downstream datasets.
The instruction for downstream experiments is in [DOWNSTREAM.md](docs/DOWNSTREAM.md).

## Acknowledgement
This codebase was built upon the work of [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [DivE](https://github.com/kdwonn/DivE). We appreciate their contributions to the original code.

## Citation

If you find our code and work useful, please consider citing:

```bibtex
@article{bae2024devias,
  title={DEVIAS: Learning Disentangled Video Representations of Action and Scene for Holistic Video Understanding},
  author={Kyungho Bae, Geo Ahn, Youngrae Kim and Jinwoo Choi},
  journal={European Conference on Computer Vision},
  year={2024}
}
