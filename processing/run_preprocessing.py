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
"""Process frame-by-frame keypoints detection results to pkl."""
import glob
import json
import multiprocessing
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
    'keypoints_dir',
    '/home/ruilongli/data/AIST++_openpose/openpose/',
    'input local dictionary that stores 2D keypoints detection results in json.'
)
flags.DEFINE_string(
    'save_dir',
    '/home/ruilongli/data/AIST++_openpose/keypoints2d/',
    'output local dictionary that stores 2D keypoints detection results in pkl.'
)
flags.DEFINE_enum(
    'data_type',
    'openpose',
    ['internal', 'openpose'],
    'Which openpose detector is being used.'
)


def array_nan(shape, dtype=np.float32):
  array = np.empty(shape, dtype=dtype)
  array[:] = np.nan
  return array


def load_keypoints2d_file(file_path):
  """load 2D keypoints from keypoint detection results.

  Only one person is extracted from the results. If there are multiple
  persons in the prediction results, we select the one with the highest
  detection score.

  Args:
    file_path: the json file path.
    njoints: number of joints in the keypoint defination.

  Returns:
    A `np.array` with the shape of [njoints, 3].
  """
  if FLAGS.data_type == "internal":
    njoints = 17
  elif FLAGS.data_type == "openpose":
    njoints = 25
  else:
    raise ValueError(FLAGS.data_type)

  keypoint = array_nan((njoints, 3), dtype=np.float32)
  det_score = 0.0

  try:
    with open(file_path, 'r') as f:
      data = json.load(f)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(e)
    return keypoint, det_score

  if FLAGS.data_type == "internal":
    keypoints = np.array(data['keypoints']).reshape((-1, njoints, 3))
    det_scores = np.array(data['detection_scores'])
  elif FLAGS.data_type == "openpose":
    keypoints = []
    for person in data["people"]:
      # npoints: 25, 70, 21, 21
      # for key in ["pose", "face", "hand_left", "hand_right"]:
      for key in ["pose"]:
        keypoints.extend(person["%s_keypoints_2d" % key])
    keypoints = np.array(keypoints).reshape(len(data["people"]), -1, 3)
    assert keypoints.shape[1] == njoints, (
      "The shape is not right. %s v.s. %d" (str(keypoints.shape), njoints)
    )
    det_scores = np.mean(keypoints[:, 0:25, -1], axis=-1)
  else:
    raise ValueError(FLAGS.data_type)

  # The detection results may contain zero person or multiple people.
  if det_scores.shape[0] == 0:
    # There is no person in this image. We set NaN to this frame.
    return keypoint, det_score
  else:
    # There are multiple people (>=1) in this image. We select the one with
    # the highest detection score.
    idx = np.argmax(det_scores)
    keypoint = keypoints[idx]
    det_score = det_scores[idx]
    return keypoint, det_score


def load_keypoints2d(data_dir, seq_name):
  """Load 2D keypoints predictions for a set of multi-view videos."""
  # Parsing sequence name to multi-view video names
  video_names = [AISTDataset.get_video_name(seq_name, view)
                 for view in AISTDataset.VIEWS]

  # In case frames are missing, we first scan all views to get a union
  # of timestamps.
  paths_cache = {}
  timestamps = []
  for video_name in video_names:
    paths = sorted(glob.glob(os.path.join(data_dir, video_name, '*.json')))
    paths_cache[video_name] = paths
    if FLAGS.data_type == "internal":
      timestamps += [
        int(os.path.basename(p).split('.')[0].split('_')[-1]) for p in paths]
    elif FLAGS.data_type == "openpose":
      timestamps += [
        int(os.path.basename(p).split('.')[0].split('_')[0]) for p in paths]
    else:
      raise ValueError(FLAGS.data_type)
  timestamps = np.array(sorted(list(set(timestamps))))  # (N,)

  # Then we load all frames according to timestamps.
  keypoints2d = []
  det_scores = []
  for video_name in video_names:
    if FLAGS.data_type == "internal":
      paths = [
          os.path.join(data_dir, video_name, f'{video_name}_{ts}.json')
          for ts in timestamps
      ]
    elif FLAGS.data_type == "openpose":
      paths = [
          os.path.join(data_dir, video_name, f'{ts:08d}_keypoints.json')
          for ts in timestamps
      ]
    else:
      raise ValueError(FLAGS.data_type)
    keypoints2d_per_view = []
    det_scores_per_view = []
    for path in paths:
      keypoint, det_score = load_keypoints2d_file(path)
      keypoints2d_per_view.append(keypoint)
      det_scores_per_view.append(det_score)
    keypoints2d.append(keypoints2d_per_view)
    det_scores.append(det_scores_per_view)

  keypoints2d = np.array(
      keypoints2d, dtype=np.float32)  # (nviews, N, njoints, 3)
  det_scores = np.array(
      det_scores, dtype=np.float32)  # (nviews, N)
  return keypoints2d, det_scores, timestamps


def process_and_save(seq_name):
  keypoints2d, det_scores, timestamps = load_keypoints2d(
      FLAGS.keypoints_dir, seq_name=seq_name)
  os.makedirs(FLAGS.save_dir, exist_ok=True)
  save_path = os.path.join(FLAGS.save_dir, f'{seq_name}.pkl')
  with open(save_path, 'wb') as f:
    pickle.dump({
        'keypoints2d': keypoints2d,
        'det_scores': det_scores,
        'timestamps': timestamps,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(_):
  if FLAGS.sequence_names:
      seq_names = FLAGS.sequence_names
  else:
      aist_dataset = AISTDataset(FLAGS.anno_dir)
      seq_names = aist_dataset.mapping_seq2env.keys()

  pool = multiprocessing.Pool(16)
  pool.map(process_and_save, seq_names)


if __name__ == '__main__':
  app.run(main)

