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
"""Utils for AIST++ Dataset."""
import os

import ffmpeg
import numpy as np


def ffmpeg_video_read(video_path, fps=None):
  """Video reader based on FFMPEG.

  This function supports setting fps for video reading. It is critical
  as AIST++ Dataset are constructed under exact 60 fps, while some of
  the AIST dance videos are not percisely 60 fps.

  Args:
    video_path: A video file.
    fps: Use specific fps for video reading. (optional)
  Returns:
    A `np.array` with the shape of [seq_len, height, width, 3]
  """
  assert os.path.exists(video_path), f'{video_path} does not exist!'
  try:
    probe = ffmpeg.probe(video_path)
  except ffmpeg.Error as e:
    print('stdout:', e.stdout.decode('utf8'))
    print('stderr:', e.stderr.decode('utf8'))
    raise e
  video_info = next(stream for stream in probe['streams']
                    if stream['codec_type'] == 'video')
  width = int(video_info['width'])
  height = int(video_info['height'])
  stream = ffmpeg.input(video_path)
  if fps:
    stream = ffmpeg.filter(stream, 'fps', fps=fps, round='down')
  stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
  out, _ = ffmpeg.run(stream, capture_stdout=True)
  out = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
  return out.copy()


def ffmpeg_video_write(data, video_path, fps=25):
  """Video writer based on FFMPEG.

  Args:
    data: A `np.array` with the shape of [seq_len, height, width, 3]
    video_path: A video file.
    fps: Use specific fps for video writing. (optional)
  """
  assert len(data.shape) == 4, f'input shape is not valid! Got {data.shape}!'
  _, height, width, _ = data.shape
  os.makedirs(os.path.dirname(video_path), exist_ok=True)
  writer = (
      ffmpeg
      .input('pipe:', framerate=fps, format='rawvideo',
             pix_fmt='rgb24', s='{}x{}'.format(width, height))
      .output(video_path, pix_fmt='yuv420p')
      .overwrite_output()
      .run_async(pipe_stdin=True)
  )
  for frame in data:
    writer.stdin.write(frame.astype(np.uint8).tobytes())
  writer.stdin.close()


def ffmpeg_video_to_images(video_path, image_dir, fps=None, ext=".jpg") -> None:
  """Video to images converter based on FFMPEG.

  This function supports setting fps for video reading. It is critical
  as AIST++ Dataset are constructed under exact 60 fps, while some of
  the AIST dance videos are not percisely 60 fps.

  Args:
    video_path: A video file.
    image_dir: A output directory to store the images.
    fps: Use specific fps for video reading. (optional)
  """
  assert os.path.exists(video_path), f'{video_path} does not exist!'
  os.makedirs(image_dir, exist_ok=True)
  stream = ffmpeg.input(video_path)
  if fps:
    stream = ffmpeg.filter(stream, 'fps', fps=fps, round='down')
  stream = ffmpeg.output(
    stream, os.path.join(image_dir, '%08d' + ext), start_number=0)
  stream = ffmpeg.overwrite_output(stream)
  ffmpeg.run(stream, quiet=True)


def unify_joint_mappings(dataset='openpose25'):
  """Unify different joint definations.

  Output unified defination:
      ['Nose',
      'RShoulder', 'RElbow', 'RWrist',
      'LShoulder', 'LElbow', 'LWrist',
      'RHip', 'RKnee', 'RAnkle',
      'LHip', 'LKnee', 'LAnkle',
      'REye', 'LEye',
      'REar', 'LEar',
      'LBigToe', 'LHeel',
      'RBigToe', 'RHeel',]

  Args:
    dataset: `openpose25`, `coco`(17) and `smpl`.
  Returns:
    a list of indexs that maps the joints to a unified defination.
  """
  if dataset == 'openpose25':
    return np.array([
        0,
        2, 3, 4,
        5, 6, 7,
        9, 10, 11,
        12, 13, 14,
        15, 16,
        17, 18,
        19, 21,
        22, 24,
    ], dtype=np.int32)
  elif dataset == 'smpl':
    # note SMPL needs to be "left-right flipped" to be consistent
    # with others
    return np.array([
        24,
        16, 18, 20,
        17, 19, 21,
        1, 4, 7,
        2, 5, 8,
        26, 25,
        28, 27,
        32, 34,
        29, 31,
    ], dtype=np.int32)
  elif dataset == 'coco':
    return np.array([
        0,
        5, 7, 9,
        6, 8, 10,
        11, 13, 15,
        12, 14, 16,
        1, 2,
        3, 4,
    ], dtype=np.int32)
  else:
    raise ValueError(f'{dataset} is not supported')

