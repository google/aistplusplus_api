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
# 
# Edited by Gavin Gray to use optionally use aria2c
"""Download AIST++ videos from AIST Dance Video Database website."""
import argparse
import multiprocessing
import os
import sys
import urllib.request
import shutil
from functools import partial

SOURCE_URL = 'https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/'
LIST_URL = 'https://storage.googleapis.com/aist_plusplus_public/20121228/video_list.txt'

def _download(video_url, download_folder):
  save_path = os.path.join(download_folder, os.path.basename(video_url))
  urllib.request.urlretrieve(video_url, save_path)
 
if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Scripts for downloading AIST++ videos.')
  parser.add_argument(
      '--download_folder',
      type=str,
      required=True,
      help='where to store AIST++ videos.')
  parser.add_argument(
      '--num_processes',
      type=int,
      default=1,
      help='number of threads for multiprocessing.')
  parser.add_argument('--aria2c',
      action='store_true',
      help='use aria2c to download the videos')
  aria2c_exists = shutil.which("aria2c") is not None

  args = parser.parse_args()
  if args.aria2c:
      assert aria2c_exists, "aria2c does not appear to be installed"
  os.makedirs(args.download_folder, exist_ok=True)

  seq_names = urllib.request.urlopen(LIST_URL)
  seq_names = [seq_name.strip().decode('utf-8') for seq_name in seq_names]
  video_urls = [
      os.path.join(SOURCE_URL, seq_name + '.mp4') for seq_name in seq_names]

  if args.aria2c:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
      urlsfile = os.path.join(tmpdirname, "aria-urls.txt")
      with open(urlsfile, "w") as f:
        f.write("\n".join(video_urls))
      subprocess.run(["aria2c",
                      "-c",
                      "--dir="+args.download_folder,
                      "--input-file="+urlsfile,
                      "--max-concurrent-downloads=%i"%args.num_processes])
  else:
    download_func = partial(_download, download_folder=args.download_folder)
    pool = multiprocessing.Pool(processes=args.num_processes)
    for i, _ in enumerate(pool.imap_unordered(download_func, video_urls)):
      sys.stderr.write('\rdownloading %d / %d' % (i + 1, len(video_urls)))
    sys.stderr.write('\ndone.\n')
