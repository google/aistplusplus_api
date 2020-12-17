"""Test code for running visualizer."""
import os

from absl import app
from absl import flags
from aist_plusplus.visualizer import plot_on_video
from aist_plusplus.loader import AISTDataset

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir',
    '/usr/local/google/home/ruilongli/data/public/aist_plusplus/',
    'input local dictionary for AIST++ annotations.')
flags.DEFINE_string(
    'video_dir',
    '/usr/local/google/home/ruilongli/data/AIST_plusplus/refined_10M_all_video/',
    'input local dictionary for AIST Dance Videos.')
flags.DEFINE_string(
    'video_name',
    'gWA_sFM_c01_d27_mWA2_ch21',
    'input sequence name to be visualized.')
flags.DEFINE_string(
    'save_dir',
    '/usr/local/google/home/ruilongli/data/public/aist_plusplus/tmp/',
    'output local dictionary that stores AIST++ visualization.')


def main(_):
  # Parsing data.
  aist_dataset = AISTDataset(FLAGS.anno_dir)
  video_path = os.path.join(FLAGS.video_dir, f'{FLAGS.video_name}.mp4')
  seq_name, view = AISTDataset.get_seq_name(FLAGS.video_name)
  view_idx = AISTDataset.VIEWS.index(view)
  keypoints2d, _, _ = AISTDataset.load_keypoint2d(
      aist_dataset.keypoint2d_dir, seq_name)
  keypoints2d = keypoints2d[view_idx, :, :, 0:2]

  # Visualize.
  os.makedirs(FLAGS.save_dir, exist_ok=True)
  save_path = os.path.join(FLAGS.save_dir, f'{FLAGS.video_name}.mp4')
  plot_on_video(keypoints2d, video_path, save_path, fps=30)


if __name__ == '__main__':
  app.run(main)

