# Utilities for dealing with google cloud storage

import os
import gcsfs
from common.utils import CONFIG, get_output_dir

GCLOUD_TOKEN = os.path.abspath(CONFIG.storage.token_dir)

# todo: replace all usage of BUCKET_FS with GCS_FS
GS_FS = gcsfs.GCSFileSystem(token=GCLOUD_TOKEN)

def clear_gs_path(gs_path):
    """ Clears the gs path of the gs:// prefix """
    if gs_path.startswith("gs://"):
        return gs_path[5:]
    return gs_path

def download_gs_file(gs_path):
    """ Checks to see if we cached this file, if not, downloads it. Returns local path """
    gs_path = clear_gs_path(gs_path)
    if "DB_PATH" in os.environ:
        parent_dir = os.environ["DB_PATH"]
    else:
        parent_dir = get_output_dir()
    local_path = f"{parent_dir}/{gs_path}"
    if not os.path.exists(local_path):
        print(f"Downloading {gs_path} to {local_path}")
        local_folder = os.path.dirname(local_path)
        os.makedirs(local_folder, exist_ok=True)
        GS_FS.get(gs_path, local_path, recursive=True)
    return local_path