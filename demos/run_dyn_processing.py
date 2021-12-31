import os
import logging
import json
import glob

from absl import app
from absl import flags
from aist_plusplus.loader import AISTDataset
from aist_plusplus.utils import ffmpeg_video_to_images
from smplx import SMPL
import torch
import imageio
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_list(
    'sequence_names',
    "gBR_sBM_cAll_d04_mBR0_ch01",
    'list of sequence names to be processed. None means to process all.')
flags.DEFINE_string(
    'anno_dir',
    '/home/ruilongli/data/AIST++/',
    'input local dictionary for AIST++ annotations.')
flags.DEFINE_string(
    'smpl_dir',
    '/home/ruilongli/data/smpl_model/smpl/',
    'input local dictionary that stores SMPL data.')
flags.DEFINE_string(
    'video_dir',
    '/home/ruilongli/data/AIST/videos/10M/',
    'input local dictionary for AIST Dance Videos.')
flags.DEFINE_string(
    'video_alpha_dir',
    '/home/ruilongli/data/AIST++/segmentation/',
    'output local dictionary that stores AIST++ segmentation masks.')
flags.DEFINE_string(
    'output_dir',
    '/home/ruilongli/data/AIST++_dyn',
    'output local dictionary that stores AIST images.')


def main(_):
    aist_dataset = AISTDataset(anno_dir=FLAGS.anno_dir)

    for seq_name in FLAGS.sequence_names:
        output_dir = os.path.join(FLAGS.output_dir, seq_name)

        # split images & masks
        for view in AISTDataset.VIEWS:
            video_name = AISTDataset.get_video_name(seq_name, view)
            logging.info("processing %s" % video_name)
            
            video_file = os.path.join(FLAGS.video_dir, video_name + ".mp4")
            image_dir = os.path.join(output_dir, "images", view)
            os.makedirs(image_dir, exist_ok=True)
            ffmpeg_video_to_images(video_file, image_dir, fps=60, ext=".jpg")

            video_file = os.path.join(FLAGS.video_alpha_dir, video_name + "_alpha1.mp4")
            image_dir = os.path.join(output_dir, "alpha1", view)
            os.makedirs(image_dir, exist_ok=True)
            ffmpeg_video_to_images(video_file, image_dir, fps=60, ext=".png")

            video_file = os.path.join(FLAGS.video_alpha_dir, video_name + "_alpha2.mp4")
            image_dir = os.path.join(output_dir, "alpha2", view)
            os.makedirs(image_dir, exist_ok=True)
            ffmpeg_video_to_images(video_file, image_dir, fps=60, ext=".png")

        # camera data
        env_name = aist_dataset.mapping_seq2env[seq_name]
        cgroup = AISTDataset.load_camera_group(aist_dataset.camera_dir, env_name)
        camera_data = cgroup.get_dicts()
        with open(os.path.join(output_dir, "camera.json"), "w") as fp:
            json.dump(camera_data, fp)

        # pose data
        pose_data = {}

        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
            aist_dataset.motion_dir, seq_name)
        smpl_poses = torch.from_numpy(smpl_poses).float()
        smpl_scaling = torch.from_numpy(smpl_scaling).float()
        smpl_trans = torch.from_numpy(smpl_trans).float()

        smpl = SMPL(model_path=FLAGS.smpl_dir, gender='MALE', batch_size=1)
        with torch.no_grad():
            rest_output, rest_transforms = smpl.forward(
                scaling=smpl_scaling.reshape(1, 1),
            )
        pose_data["rest_joints"] = rest_output.joints.squeeze(0)[:24]
        pose_data["rest_verts"] = rest_output.vertices.squeeze(0)
        pose_data["rest_tfs"] = rest_transforms.squeeze(0)

        with torch.no_grad():
            pose_output, pose_transforms = smpl.forward(
                global_orient=smpl_poses[:, 0:1],
                body_pose=smpl_poses[:, 1:],
                transl=smpl_trans,
                scaling=smpl_scaling.reshape(1, 1),
            )
        pose_data["joints"] = pose_output.joints[:, :24]
        pose_data["verts"] = pose_output.vertices
        pose_data["tfs"] = pose_transforms
        pose_data["params"] = torch.cat(
            [smpl_poses, smpl_trans, smpl_scaling.expand(smpl_poses.shape[0], 1)], 
            dim=-1
        )
        for key, value in pose_data.items():
            print (key, value.shape)

        torch.save(pose_data, os.path.join(output_dir, "pose_data.pt"))

        # post process alpha1 & alpha2 to trimap mask
        for view in AISTDataset.VIEWS:
            video_name = AISTDataset.get_video_name(seq_name, view)
            logging.info("processing %s" % video_name)
            image_dir = os.path.join(output_dir, "images", view)
            image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            alpha1_dir = os.path.join(output_dir, "alpha1", view)
            alpha1_files = sorted(glob.glob(os.path.join(alpha1_dir, "*.png")))
            alpha2_dir = os.path.join(output_dir, "alpha2", view)
            alpha2_files = sorted(glob.glob(os.path.join(alpha2_dir, "*.png")))
            mask_dir = os.path.join(output_dir, "mask", view)
            os.makedirs(mask_dir, exist_ok=True)

            for image_file, alpha1_file, alpha2_file in zip(
                image_files, alpha1_files, alpha2_files
            ):
                image = imageio.imread(image_file)
                alpha1 = imageio.imread(alpha1_file) / 255.0
                alpha2 = imageio.imread(alpha2_file) / 255.0
                fg_mask = (alpha1 > 0.5) & (alpha2 > 0.5)
                bg_mask = (alpha1 < 0.5) & (alpha2 < 0.5)
                mask = np.zeros_like(image)
                mask[fg_mask] = 255
                mask[bg_mask] = 0
                mask[~ (fg_mask | bg_mask)] = 128
                imageio.imwrite(
                    os.path.join(mask_dir, os.path.basename(alpha1_file)), mask)

if __name__ == '__main__':
    app.run(main)
