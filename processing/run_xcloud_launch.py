# Lint as: python3
"""Launcher for running AIST Processing on PyTorch with GPUs."""
import os

from absl import app
from absl import flags

import xcloud as xm

FLAGS = flags.FLAGS
flags.DEFINE_string('image_uri', None,
                    'A URI to a prebuilt Docker image, including tag.')
flags.DEFINE_string('acc_type', 'v100', 'Accelerator type`).')
flags.DEFINE_string(
    'anno_dir',
    'aist_plusplus_api/data/aist_plusplus_final/',
    'input local dictionary for AIST++ annotations.')
flags.DEFINE_string(
    'smpl_dir',
    'aist_plusplus_api/data/SMPL/',
    'input local dictionary that stores SMPL data.')
flags.DEFINE_string(
    'save_dir',
    'aist_plusplus_api/data/aist_plusplus_final/motions/',
    'output local dictionary that stores AIST++ SMPL-format motion data.')
flags.DEFINE_string(
    'save_dir_gcs',
    'gs://aist_plusplus_public/aist_plusplus_final/motions/',
    'output GCS directory.')


def main(_):
  runtime = xm.CloudRuntime(
      location='america',
      cpu=4,
      memory=15,
      accelerator=xm.GPU('nvidia-tesla-' + FLAGS.acc_type.lower(), 1),
  )

  args = {'anno_dir': FLAGS.anno_dir,
          'smpl_dir': FLAGS.smpl_dir,
          'save_dir': FLAGS.save_dir,
          'save_dir_gcs': FLAGS.save_dir_gcs}
  if FLAGS.image_uri:
    # Option 1 This will use a user-defined docker image.
    executable = xm.CloudDocker(
        name='aist-xcloud',
        runtime=runtime,
        image_uri=FLAGS.image_uri,
        args=args,
    )
  else:
    # Option 2 This will build a docker image for the user.
    executable = xm.CloudPython(
        name='aist-xcloud',
        runtime=runtime,
        project_path=(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        module_name='aist_plusplus_api.processing.run_estimate_smpl',
        args=args,
        build_steps=(
            xm.steps.copy_project_files('aist_plusplus_api') +
            xm.steps.install_python3() +
            ['apt-get update',
             'apt-get install -y libsm6 libxext6 libxrender-dev',
             'apt-get install -y git cmake'] +
            xm.steps.install_requirements('aist_plusplus_api') +
            ['pip3 install --upgrade pip',
             'pip3 install scikit-build',
             'pip3 install cmake'
             'python3 aist_plusplus_api/setup.py install']
            ),
    )
  with open(os.path.join(FLAGS.anno_dir, 'splits/all.txt'), 'r') as f:
    seq_names = f.readlines()
    seq_names = sorted([f.strip() for f in seq_names])

  parameters = [{'sequence_names': seq_names[i*70:(i+1)*70]} for i in range(21)]
  exploration = xm.ParameterSweep(
      executable, parameters, max_parallel_work_units=1000)
  xm.launch(xm.ExperimentDescription('aist-xcloud'), exploration)


if __name__ == '__main__':
  app.run(main)
