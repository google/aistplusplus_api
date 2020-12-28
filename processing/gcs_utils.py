"""Tools for Google Cloud Storage."""
import os
import re
from absl import logging
from google.cloud import storage


def parsing_gcs_path(gcs_path):
  no_prefix = gcs_path[len('gs://'):]
  splits = no_prefix.split('/')
  gcs_bucket, gcs_folder = splits[0], '/'.join(splits[1:])
  return gcs_bucket, gcs_folder


def upload_files_to_gcs(local_folder, gcs_path):
  """Upload files into Google Could Storage.

  e.g. upload_file_to_gcs(
    '/root/output/',
    'gs://research-brain-choreo-gen-xgcp/ruilongli/aist_xcloud/output/')
  """
  gcs_bucket, gcs_folder = parsing_gcs_path(gcs_path)

  storage_client = storage.Client()
  bucket = storage_client.get_bucket(gcs_bucket)
  for dirpath, _, filenames in os.walk(local_folder):
    for name in filenames:
      filename = os.path.join(dirpath, name)
      blob = storage.Blob(os.path.join(gcs_folder, name), bucket)
      with open(filename, 'rb') as f:
        blob.upload_from_file(f)
      logging.info('[upload] blob path: %s', blob.path)
      logging.info('[upload] bucket path: gs://%s/%s', gcs_bucket, gcs_folder)


def download_files_from_gcs(local_folder,
                            gcs_path,
                            pattern=None,
                            delimiter=None):
  """Download files from Google Could Storage.

  e.g. download_files_from_gcs(
    '/root/input/',
    'gs://research-brain-choreo-gen-xgcp/ruilongli/aist_xcloud/input/',
    pattern='pattern_.*.json')
  """
  os.makedirs(local_folder, exist_ok=True)
  gcs_bucket, gcs_folder = parsing_gcs_path(gcs_path)

  storage_client = storage.Client()
  bucket = storage_client.get_bucket(gcs_bucket)
  blobs = bucket.list_blobs(prefix=gcs_folder, delimiter=delimiter)
  if pattern:
    regex = re.compile(pattern)
    blobs = [blob for blob in blobs if regex.match(os.path.basename(blob.name))]
  for blob in blobs:
    logging.info('[download] blob path: %s', blob.path)
    logging.info('[download] bucket path: gs://%s/%s', gcs_bucket, gcs_path)
    dst_path = os.path.join(local_folder, os.path.basename(blob.name))
    blob.download_to_filename(dst_path)
