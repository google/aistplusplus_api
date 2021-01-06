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

#### Visualize 2D keypoints annotation.
The command below will plot 2D keypoints on to the raw video and save it to the
dictionary `./visualization/`.
``` bash
python demos/run_vis.py \
  --anno_dir <ANNOTATIONS_DIR> \
  --video_dir <VIDEO_DIR> \
  --save_dir ./visualization/ \
  --video_name gWA_sFM_c01_d27_mWA2_ch21 \
  --mode **2D**
```

#### Visualize 3D keypoints annotation.
The command below will project 3D keypoints onto the raw video using camera parameters, and save it to the
dictionary `./visualization/`.
``` bash
python demos/run_vis.py \
  --anno_dir <ANNOTATIONS_DIR> \
  --video_dir <VIDEO_DIR> \
  --save_dir ./visualization/ \
  --video_name gWA_sFM_c01_d27_mWA2_ch21 \
  --mode **3D**
```

### Visualize the SMPL joints annotation.
The command below will first calculate the 3D SMPL joint locations from our motion
annotations (joint rotations and root trajactories), then project them onto the
raw video and plot. The result will be saved into the dictionary
`./visualization/`.
``` bash
python demos/run_vis.py \
  --anno_dir <ANNOTATIONS_DIR> \
  --video_dir <VIDEO_DIR> \ 
  --smpl_dir <SMPL_DIR> \
  --save_dir ./visualization/ \ 
  --video_name gWA_sFM_c01_d27_mWA2_ch21 \ 
  --mode **SMPL**
```

#### Multi-view 3D keypoints and motion reconstruction.
This repo also provides code we used for constructing this dataset from
multi-view [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/). The
construction pipeline starts from frame-by-frame 2D keypoints detection and
manual camera estimation. Then triangulation and bundle adjustment are applied to optimize the
camera parameters as well as the 3D keypoints. Finally we sequentially fit the SMPL model to 3D keypoints to get a motion sequence represented using joint angles and root trajactory. The following figure shows our pipeline overview.

<div align="center">
<img src="assets/aist_pipeline.jpg" width="1000px"/>
<p> AIST++ construction pipeline overview.</p>
</div>


