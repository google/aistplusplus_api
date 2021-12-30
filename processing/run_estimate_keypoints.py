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

flags.DEFINE_list(
    'sequence_names',
    None,
    'list of sequence names to be processed. None means to process all.')
flags.DEFINE_string(
    'anno_dir',
    '/home/ruilongli/data/AIST++_openpose/',
    'input local dictionary for AIST++ annotations.')
flags.DEFINE_string(
    'save_dir',
    '/home/ruilongli/data/AIST++_openpose/keypoints3d/',
    'output local dictionary that stores AIST++ 3D keypoints.')
flags.DEFINE_enum(
    'data_type',
    'openpose',
    ['internal', 'openpose'],
    'Which openpose detector is being used.'
)

np.random.seed(0)


def main(_):
  aist_dataset = AISTDataset(anno_dir=FLAGS.anno_dir)

  if FLAGS.sequence_names:
      seq_names = FLAGS.sequence_names
  else:
      seq_names = aist_dataset.mapping_seq2env.keys()

  for seq_name in seq_names:
    logging.info('processing %s', seq_name)
    env_name = aist_dataset.mapping_seq2env[seq_name]

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
    if FLAGS.data_type == "internal":
      # COCO-format bone constrains
      bones = [
          (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14),
          (14, 16), (0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4),
      ]
    elif FLAGS.data_type == "openpose":
      # https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html
      body_bones = np.array([
          (0, 15), (0, 16), (15, 17), (16, 18),
          (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), 
          (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (11, 23), (22, 23), (23, 24), (24, 22),
          (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (14, 20), (19, 20), (20, 21), (21, 19)
      ])
      bones = body_bones.tolist()
      # hand_bones = np.array([
      #     (0, 1), (1, 2), (2, 3), (3, 4),
      #     (0, 5), (5, 6), (6, 7), (7, 8),
      #     (0, 9), (9, 10), (10, 11), (11, 12),
      #     (0, 13), (13, 14), (14, 15), (15, 16),
      #     (0, 17), (17, 18), (18, 19), (19, 20)
      # ])
      # bones = np.concatenate([
      #     body_bones, hand_bones + 25, hand_bones + 25 + 21]).tolist()
    else:
      raise ValueError(FLAGS.data_type)
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
