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
"""Estimate AIST++ SMPL-format Motion."""
import os
import pickle

from absl import app
from absl import flags
from absl import logging
from aist_plusplus.loader import AISTDataset
import numpy as np
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
    'save_dir',
    '/usr/local/google/home/ruilongli/data/public/aist_plusplus_final/motions/',
    'output local dictionary that stores AIST++ SMPL-format motion data.')
flags.DEFINE_list(
    'sequence_names',
    None,
    'list of sequence names to be processed. None means to process all.')
flags.DEFINE_string(
    'save_dir_gcs',
    None,
    'output GCS directory.')
np.random.seed(0)
torch.manual_seed(0)


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
    return np.array([
        24,
        17, 19, 21,
        16, 18, 20,
        2, 5, 8,
        1, 4, 7,
        25, 26,
        27, 28,
        29, 31,
        32, 34,
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


class SMPLRegressor:
  """SMPL fitting based on 3D keypoints."""

  def __init__(self, smpl_model_path, smpl_model_gener='MALE'):
    # Fitting hyper-parameters
    self.base_lr = 100.0
    self.niter = 10000
    self.metric = torch.nn.MSELoss()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.smpl_model_path = smpl_model_path
    self.smpl_model_gender = smpl_model_gener

    # Mapping to unify joint definations
    self.joints_mapping_smpl = unify_joint_mappings(dataset='smpl')

  def get_optimizer(self, smpl, step, base_lr):
    """Setup opimizer with a warm up learning rate."""
    if step < 100:
      optimizer = torch.optim.SGD([
          {'params': [smpl.transl], 'lr': base_lr},
          {'params': [smpl.scaling], 'lr': base_lr * 0.01},
          {'params': [smpl.global_orient], 'lr': 0.0},
          {'params': [smpl.body_pose], 'lr': 0.0},
          {'params': [smpl.betas], 'lr': 0.0},
      ])

    elif step < 400:
      optimizer = torch.optim.SGD([
          {'params': [smpl.transl], 'lr': base_lr},
          {'params': [smpl.scaling], 'lr': base_lr * 0.01},
          {'params': [smpl.global_orient], 'lr': base_lr * 0.001},
          {'params': [smpl.body_pose], 'lr': 0.0},
          {'params': [smpl.betas], 'lr': 0.0},
      ])

    else:
      optimizer = torch.optim.SGD([
          {'params': [smpl.transl], 'lr': base_lr},
          {'params': [smpl.scaling], 'lr': base_lr * 0.01},
          {'params': [smpl.global_orient], 'lr': base_lr * 0.001},
          {'params': [smpl.body_pose], 'lr': base_lr * 0.001},
          {'params': [smpl.betas], 'lr': 0.0},
      ])
    return optimizer

  def fit(self, keypoints3d, dtype='coco', verbose=True):
    """Run fitting to optimize the SMPL parameters."""
    assert dtype == 'coco', 'only support coco format for now.'
    assert len(keypoints3d.shape) == 3, 'input shape should be [N, njoints, 3]'
    mapping_target = unify_joint_mappings(dataset=dtype)
    keypoints3d = keypoints3d[:, mapping_target, :]
    keypoints3d = torch.from_numpy(keypoints3d).float().to(self.device)
    batch_size, njoints = keypoints3d.shape[0:2]

    # Init learnable smpl model
    smpl = SMPL(
        model_path=self.smpl_model_path,
        gender=self.smpl_model_gender,
        batch_size=batch_size).to(self.device)

    # Start fitting
    for step in range(self.niter):
      optimizer = self.get_optimizer(smpl, step, self.base_lr)

      output = smpl.forward()
      joints = output.joints[:, self.joints_mapping_smpl[:njoints], :]
      loss = self.metric(joints, keypoints3d)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if verbose and step % 10 == 0:
        logging.info(f'step {step:03d}; loss {loss.item():.3f};')

    # Return results
    return smpl, loss.item()


def main(_):
  aist_dataset = AISTDataset(FLAGS.anno_dir)
  smpl_regressor = SMPLRegressor(FLAGS.smpl_dir, 'MALE')

  if FLAGS.sequence_names:
    seq_names = FLAGS.sequence_names
  else:
    seq_names = aist_dataset.mapping_seq2env.keys()

  for seq_name in seq_names:
    logging.info('processing %s', seq_name)

    # load 3D keypoints
    keypoints3d = AISTDataset.load_keypoint3d(
        aist_dataset.keypoint3d_dir, seq_name, use_optim=True)

    # SMPL fitting
    smpl, loss = smpl_regressor.fit(keypoints3d, dtype='coco', verbose=True)

    # One last time forward
    with torch.no_grad():
      _ = smpl.forward()
    body_pose = smpl.body_pose.detach().cpu().numpy()
    global_orient = smpl.global_orient.detach().cpu().numpy()
    smpl_poses = np.concatenate([global_orient, body_pose], axis=1)
    smpl_scaling = smpl.scaling.detach().cpu().numpy()
    smpl_trans = smpl.transl.detach().cpu().numpy()

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    motion_file = os.path.join(FLAGS.save_dir, f'{seq_name}.pkl')
    with open(motion_file, 'wb') as f:
      pickle.dump({
          'smpl_poses': smpl_poses,
          'smpl_scaling': smpl_scaling,
          'smpl_trans': smpl_trans,
          'smpl_loss': loss,
      }, f, protocol=pickle.HIGHEST_PROTOCOL)

  # upload results to GCS
  if FLAGS.save_dir_gcs:
    import gcs_utils
    gcs_utils.upload_files_to_gcs(
        local_folder=FLAGS.save_dir,
        gcs_path=FLAGS.save_dir_gcs)

if __name__ == '__main__':
  app.run(main)
