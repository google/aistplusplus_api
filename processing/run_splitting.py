"""Split AIST++ Train/Val/Test Set."""
import random
import re

from absl import app
from absl import flags
from aist_plusplus.loader import AISTDataset

MUSIC_ID_TESTVAL = '(mBR0|mPO1|mLO2|mMH3|mLH4|mHO5|mWA0|mKR2|mJS3|mJB5)'
MOTION_GEN_ID_TESTVAL = '.*_sBM_.*_(mBR|mPO|mLO|mMH|mLH|mHO|mWA|mKR|mJS|mJB).*_(ch01|ch02)'
MOTION_GEN_ID_TEST_PAIRED = '.*_sBM_.*_(mBR0|mPO1|mLO2|mMH3|mLH4|mHO5|mWA0|mKR2|mJS3|mJB5)_ch02'
MOTION_GEN_ID_VAL_PAIRED = '.*_sBM_.*_(mBR0|mPO1|mLO2|mMH3|mLH4|mHO5|mWA0|mKR2|mJS3|mJB5)_ch01'


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir',
    '/usr/local/google/home/ruilongli/data/public/aist_plusplus/',
    'input local dictionary for AIST++ annotations.')
random.seed(0)


dancer_trainval = [f'd{id:02d}' for id in range(1, 31) if id % 3 != 0]
dancer_test = [f'd{id:02d}' for id in range(1, 31) if id % 3 == 0]


def main(_):
  aist_dataset = AISTDataset(FLAGS.anno_dir)
  seq_names = aist_dataset.mapping_seq2env.keys()

  seq_name_trainval = {}
  seq_name_test = []
  for seq_name in seq_names:
    gener, chorio, _, dancer, _, _ = seq_name.split('_')
    key = f'{gener}_{chorio}'
    if dancer in dancer_trainval:
      if key not in seq_name_trainval:
        seq_name_trainval[key] = []
      seq_name_trainval[key].append(seq_name)
    elif dancer in dancer_test:
      seq_name_test.append(seq_name)
    else:
      raise ValueError

  tmp_names = []
  seq_name_val = []
  for key, seq_names in seq_name_trainval.items():
    tmp_names += seq_names
    if 'sBM' in  key:
      seq_name_val += random.sample(seq_names, k=6)
    elif 'sFM' in key:
      seq_name_val += random.sample(seq_names, k=1)
    else:
      raise ValueError
  seq_name_train = list(set(tmp_names) - set(seq_name_val))
  print (len(seq_name_train), len(seq_name_val), len(seq_name_test))

  with open('/usr/local/google/home/ruilongli/data/public/aist_plusplus/splits/pose_train.txt', 'w') as f:
    f.write('\n'.join(seq_name_train))
  with open('/usr/local/google/home/ruilongli/data/public/aist_plusplus/splits/pose_val.txt', 'w') as f:
    f.write('\n'.join(seq_name_val))
  with open('/usr/local/google/home/ruilongli/data/public/aist_plusplus/splits/pose_test.txt', 'w') as f:
    f.write('\n'.join(seq_name_test))

  seq_names = aist_dataset.mapping_seq2env.keys()
  cross_modal_train = []
  cross_modal_val = []
  cross_modal_test = []
  for seq_name in seq_names:
    if re.match(MOTION_GEN_ID_TEST_PAIRED, seq_name):
      cross_modal_test.append(seq_name)
    elif re.match(MOTION_GEN_ID_VAL_PAIRED, seq_name):
      cross_modal_val.append(seq_name)
    else:
      if not re.match(MOTION_GEN_ID_TESTVAL, seq_name) and \
         not re.match(MUSIC_ID_TESTVAL, seq_name.split('_')[-2]):
        cross_modal_train.append(seq_name)
  print (len(cross_modal_train), len(cross_modal_val), len(cross_modal_test))

  with open('/usr/local/google/home/ruilongli/data/public/aist_plusplus/splits/crossmodal_train.txt', 'w') as f:
    f.write('\n'.join(cross_modal_train))
  with open('/usr/local/google/home/ruilongli/data/public/aist_plusplus/splits/crossmodal_val.txt', 'w') as f:
    f.write('\n'.join(cross_modal_val))
  with open('/usr/local/google/home/ruilongli/data/public/aist_plusplus/splits/crossmodal_test.txt', 'w') as f:
    f.write('\n'.join(cross_modal_test))

if __name__ == '__main__':
  app.run(main)
