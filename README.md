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
└── \*.mp4

<SMPL_DIR>
└── SMPL_NEUTRAL.pkl
```

#### Visualize 2D keypoints annotation
The command below will plot 2D keypoints onto the raw video and save it to the
dictionary `./visualization/`.
``` bash
python demos/run_vis.py \
  --anno_dir <ANNOTATIONS_DIR> \
  --video_dir <VIDEO_DIR> \
  --save_dir ./visualization/ \
  --video_name gWA_sFM_c01_d27_mWA2_ch21 \
  --mode 2D
```

#### Visualize 3D keypoints annotation
The command below will project 3D keypoints onto the raw video using camera parameters, and save it to the
dictionary `./visualization/`.
``` bash
python demos/run_vis.py \
  --anno_dir <ANNOTATIONS_DIR> \
  --video_dir <VIDEO_DIR> \
  --save_dir ./visualization/ \
  --video_name gWA_sFM_c01_d27_mWA2_ch21 \
  --mode 3D
```

### Visualize the SMPL joints annotation
The command below will first calculate the SMPL joint locations from our motion
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
  --mode SMPL
```

#### Multi-view 3D keypoints and motion reconstruction

This repo also provides code we used for constructing this dataset from
multi-view [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/). The
construction pipeline starts from frame-by-frame 2D keypoints detection and
manual camera estimation. Then triangulation and bundle adjustment are applied to optimize the
camera parameters as well as the 3D keypoints. Finally we sequentially fit the SMPL model to 3D keypoints to get a motion sequence represented using joint angles and root trajactory. The following figure shows our pipeline overview.

<div align="center">
<img src="assets/aist_pipeline.jpg" width="1000px"/>
<p> AIST++ construction pipeline overview.</p>
</div>

The annotations in AIST++ are in COCO-format (17) for 2D \& 3D keypoints, and
SMPL-format for human motion annotations. It is designed to serve for general
research purposes. However, in some cases you might need those data in different format
(e.g., [Openpose]() / [Alphapose]() keypoints format, or [SMPLX]() human motion
format). With the code we provide, it should be easy to construct your own
version of AIST++, with your own keypoint detector or human model defination.

Assume you have your own 2D keypoint detection results stored in `<KEYPOINTS_DIR>`, you can run the constructing pipeline start with preprocessing
the keypoints into the `.pkl` format that we support. The code we used at this
step is as follows but you might need to modify the script `run_preprocessing.py` in order to compatible with your own data.
``` bash
python processing/run_preprocessing.py \
  --keypoints_dir <KEYPOINTS_DIR> \
  --save_dir <ANNOTATIONS_DIR>/keypoints2d/
```

Then you can estimate the camera parameters using your 2D keypoints. This step
is optional as you can still use our camera parameters annotation which are
quite accurate. At this step, you will need the `<ANNOTATIONS_DIR>/cameras/mapping.txt` file which stores the mapping from videos to different environment settings.
``` bash
# If you would like to estimate your own camera parameters:
python processing/run_estimate_camera.py \
  --anno_dir <ANNOTATIONS_DIR> \
  --save_dir <ANNOTATIONS_DIR>/cameras/ \
# Or you can skip this step by just using our camera parameters annotation.
```

Next step is to perform 3D keypoints reconstruction from multi-view 2D keypoints
and camera parameters. You can just run:
``` bash
python processing/run_estimate_keypoints.py \
  --anno_dir <ANNOTATIONS_DIR> \
  --save_dir <ANNOTATIONS_DIR>/keypoints3d/ \
```

Finally we can estimate SMPL-format human motion data by sequencially fitting
the 3D keypoints to SMPL model. If you would like to use other human model such
as [SMPLX]() and so on, you will need to do some modifications in the script
`run_estimate_smpl.py`. You can run the following commands run the SMPL fitting.
``` bash
python processing/run_estimate_smpl.py \
  --anno_dir <ANNOTATIONS_DIR> \
  --smpl_dir <SMPL_DIR> \
  --save_dir <ANNOTATIONS_DIR>/motions \
```
Note that this step will take several days to process the entire dataset if your machine have only one GPU on it.
In practise, we run this step on a cluster so we here only provide the single-thread version code.


