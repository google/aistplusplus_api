"""Visualize the AIST++ Dataset."""

import cv2
import numpy as np

from . import utils

_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
           [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
           [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
           [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
           [255, 0, 170], [255, 0, 85]]


def plot_kpt(keypoint, canvas):
  for i, (x, y) in enumerate(keypoint[:, 0:2]):
    if np.isnan(x) or np.isnan(y):
      continue
    cv2.circle(canvas, (int(x), int(y)),
               7,
               _COLORS[i % len(_COLORS)],
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


