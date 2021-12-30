#! /bin/bash

SEQUENCE_NAME=gBR_sBM_cAll_d04_mBR0_ch01

CUDA_VISIBLE_DEVICES=0,1,2,3 python processing/run_openpose.py --sequence_names=${SEQUENCE_NAME}
CUDA_VISIBLE_DEVICES=0,1,2,3 python processing/run_preprocessing.py --sequence_names=${SEQUENCE_NAME}
CUDA_VISIBLE_DEVICES=0,1,2,3 python processing/run_estimate_keypoints.py --sequence_names=${SEQUENCE_NAME}
CUDA_VISIBLE_DEVICES=0,1,2,3 python processing/run_estimate_smpl.py  --sequence_names=${SEQUENCE_NAME}
CUDA_VISIBLE_DEVICES=0,1,2,3 python processing/run_segmentation.py --sequence_names=${SEQUENCE_NAME}