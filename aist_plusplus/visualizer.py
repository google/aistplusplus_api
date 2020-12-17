"""Visualize the AIST++ Dataset."""

import cv2
import numpy as np
from . import utils

_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
           [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
           [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
           [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
           [255, 0, 170], [255, 0, 85]]


def vis_kpt2d(kpt, canvas):
  for kpt_idx, (x, y) in enumerate(kpt[:, 0:2]):
    if np.isnan(x) or np.isnan(y):
      continue
    cv2.circle(
        canvas, (int(x), int(y)),
        7,
        _COLORS[kpt_idx % len(_COLORS)],
        thickness=-1)
  return canvas


def plot_on_video(keypoints2d, video_path, save_path, fps=60):
  assert len(keypoints2d.shape) == 3, (
      f'Input shape is not valid! Got {keypoints2d.shape}')
  # Load video as a numpy array.
  video = utils.FfmpegReader(video_path, fps=fps).load()
  _, height, width, _ = video.shape
  # Plot keypoints onto it.
  for iframe, kpt in enumerate(keypoints2d):
    video[iframe] = vis_kpt2d(kpt, video[iframe])
  # Save visualization to video file.
  utils.FfmpegWriter(save_path, width, height).save(video)
