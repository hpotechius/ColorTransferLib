import os
import importlib

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
    #toremove = ["__init__.pyc", "__init__.py", "__pycache__", "FCM", "GMM","RGH","DPT","PSN","BCC","NST","TPS","EB3","PDF","GLO","HIS","FUZ","CAM"]
    toremove = ["__init__.pyc", "__init__.py", "__pycache__", ".DS_Store", "FCM", "GMM"]
    available_methods = [m for m in available_methods if m not in toremove]
    return available_methods

# ----------------------------------------------------------------------------------------------------------------------
# read all available metrics from the Evaluation folders
# ----------------------------------------------------------------------------------------------------------------------
def get_metrics():
    available_metrics = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/../Evaluation")
    toremove = ["__init__.pyc", "__init__.py", "__pycache__", ".DS_Store"]
    available_metrics = [m for m in available_metrics if m not in toremove]
    return available_metrics