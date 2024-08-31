# Data Preparation

* For training  
We train DEVIAS and baseline models on [UCF-101](https://arxiv.org/abs/1212.0402) and [Kinetics-400](https://arxiv.org/abs/1705.07750), which are the most popular datasets for training action recognition models.
To use these datasets, please download the videos through [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) or [Kinetics-400](https://opendatalab.com/OpenMMLab/Kinetics-400).   
We also train the models on [HVU](https://arxiv.org/abs/1904.11451), which provides both action and scene ground-truth labels. You can download the videos from [the official repository](https://github.com/holistic-video-understanding/HVU-Downloader). Note that we use the subset of HVU, consisting of videos that have a single action label and a single scene label, with a total of 27,532 videos. The list of videos in the subset is `filelist/hvu/train.csv`.

* For evaluation  
We use various datasets for evaluating the models on both seen and unseen action-scene combination scenarios.  
    * For **seen** action-scene combinations  
    We use the original test/validation set of [UCF-101](https://arxiv.org/abs/1212.0402)/[Kinetics-400](https://arxiv.org/abs/1705.07750) for seen combination scenarios.
    * For **unseen** action-scene combinations  
    We use [SCUBA](https://arxiv.org/abs/2211.12883) and [HAT](https://openreview.net/forum?id=eOnQ2etkxto) to test the models in unseen combination scenarios.

        * [SCUBA](https://arxiv.org/abs/2211.12883)  
        You can download the dataset from [the official repository](https://github.com/lihaoxin05/StillMix). You need to download ```UCF101-SCUBA-*.tar.gz``` and ```K400-SCUBA-*.tar.gz```. 
        * [HAT](https://openreview.net/forum?id=eOnQ2etkxto)  
        You can download the dataset from [the official repository](https://github.com/princetonvisualai/HAT). You need to download the files in ```Kinetics-400 (E2FGVI as inpainting)``` and ```UCF101```. 
        
        We recommend to prepare the data directory as below :  
        ```txt
        root_dir
        ├── scuba
        │   ├── ucf101-vqgan
        │   └── ucf101-places
        │   └── ucf101-sinusoidal
        │   └── kinetics-vqgan
        │   └── kinetics-places
        │   └── kinetics-sinusoidal
        └── hat
            ├── ucf101
            │   ├── inpaint
            │   │   └── ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01
            │   └── seg
            │   └── rawframes
            └── kinetics
                ├── inpaint
                │   └── videos/f0Vd56Yjrtc_000165_000175
                └── seg
                └── original
        ```  
        In the case of HAT-Kinetics, run the command 'mkdir videos; mv \*/\* videos/' to use the same data directory structure for evaluating Scene-Only and HAT-Far/Random/Close.

* For downstream experiments  
Please download videos through the official sites of [Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html), [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something), [ActivityNet](http://activity-net.org/download.html).  
