# coding=utf-8
# Copyright 2020 The Google AI Perception Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Estimate foreground masks."""
import os

from absl import app
from absl import flags
from absl import logging
from aist_plusplus.loader import AISTDataset
from aist_plusplus.utils import ffmpeg_video_read
import numpy as np
import torch
import imageio
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_list(
    'sequence_names',
    None,
    'list of sequence names to be processed. None means to process all.')
flags.DEFINE_string(
    'anno_dir',
    '/home/ruilongli/data/AIST++_openpose/',
    'input local dictionary for AIST++ annotations.')
flags.DEFINE_string(
    'video_dir',
    '/home/ruilongli/data/AIST/videos/10M/',
    'input local dictionary for AIST Dance Videos.')
flags.DEFINE_string(
    'save_dir',
    '/home/ruilongli/data/AIST++_openpose/segmentation/',
    'output local dictionary that stores AIST++ segmentation masks.')
np.random.seed(0)


def estimate_background(input_video: str, alpha_video: str, output_image: str):
    video_reader = imageio.get_reader(input_video)
    alpha_reader = imageio.get_reader(alpha_video)
    background, weights = 0, 0
    for img, alpha in tqdm.tqdm(zip(video_reader, alpha_reader)):
        weight = (1 - np.float32(alpha) / 255.0)
        weights += weight
        background += np.float32(img) * weight
    background /= (weights + 1e-8)
    imageio.imwrite(output_image, np.uint8(background))


def main(_):
    # Here we use https://github.com/PeterL1n/RobustVideoMatting (GPL-3.0 License)
    # to get an initial alpha matting prediction.
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").cuda()
    converter = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

    # Here we use https://github.com/PeterL1n/BackgroundMattingV2 (MIT License)
    # to get an accurate alpha matting.
    if not os.path.exists("/tmp/model.pth"):
        os.system("gdown https://drive.google.com/uc?id=1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf -O /tmp/model.pth -q")
    if not os.path.exists("/tmp/BackgroundMattingV2"):
        os.system("cd /tmp/; git clone -q https://github.com/PeterL1n/BackgroundMattingV2")

    if FLAGS.sequence_names:
        seq_names = FLAGS.sequence_names
    else:
        aist_dataset = AISTDataset(FLAGS.anno_dir)
        seq_names = aist_dataset.mapping_seq2env.keys()

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    for seq_name in seq_names:    
        for view in AISTDataset.VIEWS:
            video_name = AISTDataset.get_video_name(seq_name, view)
            video_file = os.path.join(FLAGS.video_dir, video_name + ".mp4")
            if not os.path.exists(video_file):
                continue

            # step 1. initial alpha matting prediction (not accurate enough).
            logging.info('processing %s', video_file)
            alpha_file = os.path.join(FLAGS.save_dir, video_name + "_alpha1.mp4")
            if not os.path.exists(alpha_file):
                converter(
                    model,                           # The loaded model, can be on any device (cpu or cuda).
                    input_source=video_file,         # A video file or an image sequence directory.
                    downsample_ratio=None,           # [Optional] If None, make downsampled max size be 512px.
                    output_type='video',             # Choose "video" or "png_sequence"
                    output_alpha=alpha_file,         # [Optional] Output the raw alpha prediction.
                    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
                    seq_chunk=12,                    # Process n frames at once for better parallelism.
                    num_workers=1,                   # Only for image sequence input. Reader threads.
                    progress=True                    # Print conversion progress.
                )

            # step 2. estimate the background image from the inital alpha matting prediction.
            background_file = os.path.join(FLAGS.save_dir, video_name + "_bg.png")
            if not os.path.exists(background_file):
                estimate_background(video_file, alpha_file, background_file)

            # step 3. estimate the more accurate alpha matting.
            final_file = os.path.join(FLAGS.save_dir, video_name + "_alpha2")
            if not os.path.exists(final_file):
                os.system(
                    "cd /tmp/BackgroundMattingV2/; " +
                    "python inference_video.py " +
                        "--model-type mattingrefine " +
                        "--model-backbone resnet50 " +
                        "--model-backbone-scale 0.25 " +
                        "--model-refine-mode sampling " +
                        "--model-refine-sample-pixels 80000 " +
                        "--model-checkpoint '/tmp/model.pth' " +
                        "--video-src '%s' " % video_file +
                        "--video-bgr '%s' " % background_file +
                        "--output-dir '%s' " % final_file +
                        "--output-type pha"
                )
            if os.path.exists(final_file):
                os.system("mv %s/pha.mp4 %s.mp4; rm -rf %s" % (final_file, final_file, final_file))


if __name__ == '__main__':
  app.run(main)
