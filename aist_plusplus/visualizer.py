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
"""Visualize the AIST++ Dataset."""

from . import utils
import cv2
import numpy as np

_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
           [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
           [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
           [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
           [255, 0, 170], [255, 0, 85]]


def plot_kpt(keypoint, canvas, color=None):
  for i, (x, y) in enumerate(keypoint[:, 0:2]):
    if np.isnan(x) or np.isnan(y) or x < 0 or y < 0:
      continue
    cv2.circle(canvas, (int(x), int(y)),
               7,
               color if color is not None else _COLORS[i % len(_COLORS)],
               thickness=-1)
  return canvas


def plot_on_video(keypoints2d, video_path, save_path, fps=60):
  assert len(keypoints2d.shape) == 3, (
      f'Input shape is not valid! Got {keypoints2d.shape}')
  video = utils.ffmpeg_video_read(video_path, fps=fps)
  for iframe, keypoint in enumerate(keypoints2d):
    if iframe >= video.shape[0]:
      break
    video[iframe] = plot_kpt(keypoint, video[iframe])
  utils.ffmpeg_video_write(video, save_path, fps=fps)


