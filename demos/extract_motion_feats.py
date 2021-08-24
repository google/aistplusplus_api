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
"""Demo code for motion feature extraction."""
from absl import app
from absl import flags
from aist_plusplus.loader import AISTDataset
from aist_plusplus.features.kinetic import extract_kinetic_features
from aist_plusplus.features.manual import extract_manual_features
from smplx import SMPL
import torch


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir',
    '/usr/local/google/home/ruilongli/data/public/aist_plusplus_final/',
    'input local dictionary for AIST++ annotations.')
flags.DEFINE_string(
    'smpl_dir',
    '/usr/local/google/home/ruilongli/data/SMPL/',
    'input local dictionary that stores SMPL data.')
flags.DEFINE_string(
    'video_name',
    'gWA_sFM_c01_d27_mWA2_ch21',
    'input video name to be visualized.')


def main(_):
  # Parsing data info.
  aist_dataset = AISTDataset(FLAGS.anno_dir)
  seq_name, view = AISTDataset.get_seq_name(FLAGS.video_name)

  # SMPL joints
  smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
      aist_dataset.motion_dir, seq_name)
  smpl = SMPL(model_path=FLAGS.smpl_dir, gender='MALE', batch_size=1)
  # Note here we calculate `transl` as `smpl_trans/smpl_scaling` for 
  # normalizing the motion in generic SMPL model scale.
  keypoints3d = smpl.forward(
      global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
      body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
      transl=torch.from_numpy(smpl_trans / smpl_scaling).float(),
      ).joints.detach().numpy()
  
  # extract features
  features_k = extract_kinetic_features(keypoints3d)
  print ("kinetic features:", features_k)
  features_m = extract_manual_features(keypoints3d)
  print ("manual features:", features_m)


if __name__ == '__main__':
  app.run(main)

