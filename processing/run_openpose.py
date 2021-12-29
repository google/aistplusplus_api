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
"""Detect frame-by-frame 2D keypoints using openpose."""
import os
import logging

from absl import app
from absl import flags
from aist_plusplus.loader import AISTDataset
from aist_plusplus.utils import ffmpeg_video_to_images

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'seq_name',
    'gKR_sFM_cAll_d29_mKR5_ch14',
    'input video name to be visualized.')
flags.DEFINE_string(
    'openpose_dir',
    '/home/ruilongli/workspace/openpose',
    'input openpose repo that contains the executable.')
flags.DEFINE_string(
    'video_dir',
    '/home/ruilongli/data/AIST/videos/10M/',
    'input local dictionary for AIST Dance Videos.')
flags.DEFINE_string(
    'image_save_dir',
    '/home/ruilongli/data/AIST/images/10M/',
    'output local dictionary that stores AIST images.')
flags.DEFINE_string(
    'openpose_save_dir',
    '/home/ruilongli/data/AIST++/openpose',
    'output local dictionary that stores AIST++ openpose results.')


def main(_):
    os.makedirs(FLAGS.image_save_dir, exist_ok=True)
    os.makedirs(FLAGS.openpose_save_dir, exist_ok=True)
    for view in AISTDataset.VIEWS:
        video_name = AISTDataset.get_video_name(FLAGS.seq_name, view)
        video_file = os.path.join(FLAGS.video_dir, video_name + ".mp4")
        if not os.path.exists(video_file):
            continue
        logging.info('processing %s', video_file)
        
        # extract images
        image_dir = os.path.join(FLAGS.image_save_dir, video_name)
        ffmpeg_video_to_images(video_file, image_dir, fps=60)
        
        # extract keypoints
        save_dir = os.path.join(FLAGS.openpose_save_dir, video_name)
        os.system(
            "cd %s; " % FLAGS.openpose_dir +
            "./build/examples/openpose/openpose.bin " +
                "--image_dir %s " % image_dir +
                "--write_json %s " % save_dir +
                "--display 0 --hand --face --render_pose 0"
        )

        # break

if __name__ == '__main__':
    app.run(main)
