import os
import importlib
import requests
from tqdm import tqdm

from appdirs import user_cache_dir
import os

# ----------------------------------------------------------------------------------------------------------------------
# Download all additional files necessary for the application of the given method
# ----------------------------------------------------------------------------------------------------------------------
def init_model_files(method, files):
    cache_dir = get_cache_dir()
    method_cache_dir = os.path.join(cache_dir, method)
    os.makedirs(method_cache_dir, exist_ok=True)

    file_paths = {}

    for file in files:
        file_path = os.path.join(method_cache_dir, file)
        base_url = "https://potechius.com/Downloads/ColorTransferLibModels"
        url = os.path.join(base_url, method, file)
        download_model_file(url, file_path)

        file_paths[file] = file_path
    
    return file_paths

# ----------------------------------------------------------------------------------------------------------------------
# Downloads a given file at the given location
# ----------------------------------------------------------------------------------------------------------------------
def download_model_file(url, save_path):
    if os.path.exists(save_path):
        return

    with requests.get(url, stream=True) as r:
        print("Download: " + url)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 8192
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=block_size):
                t.update(len(chunk))
                f.write(chunk)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")

# ----------------------------------------------------------------------------------------------------------------------
# Returns the directory of the the .cache folder and create a ColorTransferLib folder
# ----------------------------------------------------------------------------------------------------------------------
def get_cache_dir():
    cache_dir = user_cache_dir('ColorTransferLib')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def check_compatibility(src, ref, compatibility_list):
    # return element
    output = {
        "status_code": 0,
        "response": "",
        "object": None
    }
    
    if src.get_type() not in compatibility_list["src"]:
        output["status_code"] = -1
        output["response"] = "No support for the following source type: " + src.get_type()

    if ref.get_type() not in compatibility_list["ref"]:
        output["status_code"] = -1
        output["response"] = "No support for the following reference type: " + ref.get_type()

    return output

# ----------------------------------------------------------------------------------------------------------------------
# read all available algorithms from the Algorithms folder and import them
# ----------------------------------------------------------------------------------------------------------------------
def get_methods():
    available_methods = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/../Algorithms")
    toremove = ["__init__.pyc", "__init__.py", "__pycache__", ".DS_Store", "FCM", "GMM","RGH","DPT","PSN","BCC","NST","TPS","EB3","PDF","HIS","FUZ","CAM","GPC","RHG","CCS", "MKL"]
    #toremove = ["__init__.pyc", "__init__.py", "__pycache__", ".DS_Store", "FCM", "GMM", "TPS"]
    available_methods = [m for m in available_methods if m not in toremove]
    return available_methods

# ----------------------------------------------------------------------------------------------------------------------
# read all available metrics from the Evaluation folders
# ----------------------------------------------------------------------------------------------------------------------
def get_metrics():
    available_metrics = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/../Evaluation")
    toremove = ["__init__.pyc", "__init__.py", "__pycache__", ".DS_Store", "VSI"]
    available_metrics = [m for m in available_metrics if m not in toremove]
    return available_metrics