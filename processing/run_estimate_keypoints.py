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
"""Estimate AIST++ 3D keypoints."""
import os
import pickle

from absl import app
from absl import flags
from absl import logging
from aist_plusplus.loader import AISTDataset
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir',
    '/usr/local/google/home/ruilongli/data/public/aist_plusplus_final/',
    'input local dictionary for AIST++ annotations.')
flags.DEFINE_string(
    'save_dir',
    '/usr/local/google/home/ruilongli/data/public/aist_plusplus_final/keypoints3d/',
    'output local dictionary that stores AIST++ 3D keypoints.')
np.random.seed(0)


def main(_):
  aist_dataset = AISTDataset(anno_dir=FLAGS.anno_dir)

  for seq_name, env_name in aist_dataset.mapping_seq2env.items():
    logging.info('processing %s', seq_name)

    # Load camera parameters
    cgroup = AISTDataset.load_camera_group(aist_dataset.camera_dir, env_name)

    # load 2D keypoints
    keypoints2d, det_scores, _ = AISTDataset.load_keypoint2d(
        aist_dataset.keypoint2d_dir, seq_name=seq_name)
    nviews, nframes, _, _ = keypoints2d.shape
    assert det_scores.shape[0] == nviews
    assert det_scores.shape[1] == nframes
    if seq_name == 'gBR_sBM_cAll_d04_mBR0_ch01':
      keypoints2d[4] = np.nan  # not synced view
    if seq_name == 'gJB_sBM_cAll_d07_mJB3_ch05':
      keypoints2d[6] = np.nan  # size 640x480

    # filter keypoints to select those best points
    kpt_thre = 0.15
    ignore_idxs = np.where(keypoints2d[:, :, :, 2] < kpt_thre)
    keypoints2d[ignore_idxs[0], ignore_idxs[1], ignore_idxs[2], :] = np.nan
    det_thre = 0.0
    ignore_idxs = np.where(det_scores < det_thre)
    keypoints2d[ignore_idxs[0], ignore_idxs[1], :, :] = np.nan
    keypoints2d = keypoints2d[:, :, :, 0:2]

    # 3D pose triangulation and temporal optimization.
    bones = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14),
        (14, 16), (0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4),
    ]  # COCO-format bone constrains
    keypoints3d = cgroup.triangulate(
        keypoints2d.reshape(nviews, -1, 2)
    ).reshape(nframes, -1, 3)
    keypoints3d_optim = cgroup.triangulate_optim(
        keypoints2d, constraints=bones, verbose=True
    ).reshape(nframes, -1, 3)

    # Save to pkl
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    keypoints_file = os.path.join(FLAGS.save_dir, f'{seq_name}.pkl')
    with open(keypoints_file, 'wb') as f:
      pickle.dump({
          'keypoints3d': keypoints3d,
          'keypoints3d_optim': keypoints3d_optim,
      }, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
