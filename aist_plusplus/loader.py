"""AIST++ Dataset Loader."""
import json
import os
import pickle

import aniposelib
import numpy as np

class AISTDataset:
  """A dataset class for loading, processing and plotting AIST++."""

  VIEWS = ['c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09']

  def __init__(self, anno_dir, split_file):
    assert os.path.exists(anno_dir), f'Data does not exist at {anno_dir}!'

    # Init paths
    self.camera_dir = os.path.join(anno_dir, 'cameras/')
    self.motion_dir = os.path.join(anno_dir, 'motions/')
    self.keypoint_dir = os.path.join(anno_dir, 'keypoints/')
    self.filter_file = os.path.join(anno_dir, 'ignore_list.txt')
    self.seq_names = np.loadtxt(split_file, dtype=str)

  @classmethod
  def get_video_name(cls, seq_name, view):
    """Get AIST video name from sequence name."""
    return seq_name.replace('cAll', view)

  @classmethod
  def get_seq_name(cls, video_name):
    """Get AIST video name from sequence name."""
    tags = video_name.split('_')
    if len(tags) == 3:
      tags[1] = 'cAll'
    else:
      tags[2] = 'cAll'
    return '_'.join(tags)

  @classmethod
  def load_camera_group(cls, camera_dir, env_name):
    """Load a set of cameras in the environment."""
    file_path = os.path.join(camera_dir, f'{env_name}.json')
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'r') as f:
      params = json.load(f)
    cameras = []
    for param_dict in params:
      camera = aniposelib.cameras.Camera(name=param_dict['name'],
                                         size=param_dict['size'],
                                         matrix=param_dict['matrix'],
                                         rvec=param_dict['rotation'],
                                         tvec=param_dict['translation'],
                                         dist=param_dict['distortions'])
      cameras.append(camera)
    camera_group = aniposelib.cameras.CameraGroup(cameras)
    return camera_group

  @classmethod
  def load_motion(cls, motion_dir, seq_name):
    """Load a motion sequence represented using SMPL format."""
    file_path = os.path.join(motion_dir, f'{seq_name}.json')
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'r') as f:
      data = pickle.load(f)
    smpl_poses = data['smpl_poses']  # (N, 24, 3)
    smpl_scaling = data['smpl_scaling']  # (1,)
    smpl_trans = data['smpl_trans']  # (N, 3)
    return smpl_poses, smpl_scaling, smpl_trans

  @classmethod
  def load_keypoint(cls, keypoint_dir, seq_name, mode='3d'):
    """Load a keypoint sequence represented using COCO format."""
    file_path = os.path.join(keypoint_dir, f'{seq_name}.json')
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'r') as f:
      data = pickle.load(f)
    if mode == '3d':
      return data['keypoints3d']  # (N, 17, 3)
    elif mode == '3d_optim':
      return data['keypoints3d_optim']  # (N, 17, 3)
    elif mode == '2d':
      return data['keypoints2d']  # (9, N, 17, 3)
    else:
      raise ValueError(f'{mode} is not valid!')
