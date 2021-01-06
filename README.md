# AIST++ API
 
This repo contains starter code for using AIST++ dataset. To download the
dataset or explore the details of this dataset, please go to our [website]().

## Installation
The code have been tested on `python>=3.7`. You can install the dependencies and
this repo by:
``` bash
pip install -r requirements.txt
python setup.py install
```

## How to use
We provide demo code for loading and visualizing AIST++ annotations. Before running the code, 
you should have downloaded the data including [AIST++ annotations](), [AIST Dance Videos]() and [SMPL model]() (for SMPL visualization only).

The directory structure of the data is expected to be:
```
<ANNOTATIONS_DIR>
├── motions/
├── keypoints2d/
├── keypoints3d/
├── splits/
├── cameras/
└── ignore_list.txt

<VIDEO_DIR>
└── *.mp4

<SMPL_DIR>
└── SMPL_NEUTRAL.pkl
```

as well as the
For loading and visualizing AIST++ 2D & 3D keypoints and SMPL joints. For example, to visualize a specific motion sequence, you can run:
```
python demos/run_vis.py \
  --anno_dir <AIST++_DATA_DIR> \
  --video_dir <AIST_VIDEO_DIR> \
  --smpl_dir <SMPL_DATA_DIR> \
  --save_dir ./visualization/ \
  --video_name gWA_sFM_c01_d27_mWA2_ch21 \
  --mode 2D
```
Note you can use `--mode` option to visualize either detected COCO-format 2D keypoints (`--mode=2D`), 3D re-projected COCO-format keypoints (`--mode=3D`) or SMPL-format joints (`--mode=SMPL`).


This repo also provides code we used for constructing this dataset from
multi-view [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/). The
construction pipeline starts from frame-by-frame 2D keypoints detection and
manual camera estimation. Then triangulation and bundle adjustment are applied to optimize the
camera parameters as well as the 3D keypoints. Finally we sequentially fit the SMPL model to 3D keypoints to get a motion sequence represented using joint angles and root trajactory. The following figure shows our pipeline overview.

<div align="center">
<img src="assets/aist_pipeline.jpg" width="1000px"/>
<p> AIST++ construction pipeline overview.</p>
</div>


