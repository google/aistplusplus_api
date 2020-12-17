"""Utils for AIST++ Dataset."""
import os

import ffmpeg
import numpy as np


class FfmpegReader:
  """Video Reader based on FFMPEG."""

  def __init__(self, video_path, fps=None):
    self.video_path = video_path
    self.fps = fps
    assert os.path.exists(video_path), 'f{video_path} does not exist!'

    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams']
                      if stream['codec_type'] == 'video')
    self.width = int(video_info['width'])
    self.height = int(video_info['height'])

  def load(self):
    stream = ffmpeg.input(self.video_path)
    if self.fps:
      stream = ffmpeg.filter(stream, 'fps', fps=self.fps, round='up')
    stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
    out, _ = ffmpeg.run(stream, capture_stdout=True)
    out = np.frombuffer(out, np.uint8).reshape([-1, self.height, self.width, 3])
    return out.copy()


class FfmpegWriter:
  """Video Writer based on FFMPEG."""

  def __init__(self, video_path, width, height, fps=60):
    self.writer = (
        ffmpeg
        .input('pipe:', framerate=fps, format='rawvideo',
               pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(video_path, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

  def save(self, video):
    for frame in video:
      self.writer.stdin.write(frame.astype(np.uint8).tobytes())
    self.writer.stdin.close()

