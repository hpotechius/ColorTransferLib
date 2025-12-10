"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""



 
import warnings
import importlib
import sys
import json
import os
import copy
import subprocess
import pickle

# Suppresses the following warning: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 
# 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to 
# True in Numba 0.59.0.
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# Suppresses the following warning: 
# (1) UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
# warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
# (2) UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in  the future, please use 
# 'weights' instead.
warnings.filterwarnings("ignore", message=".*deprecated.*")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR


from ColorTransferLib.Utils.BaseOptions import BaseOptions
# from ColorTransferLib.Utils.Helper import get_methods, get_metrics

#available_methods = get_methods()
#available_metrics = get_metrics()

#print(available_methods)

# for m in available_methods:
#     exec(m + " = getattr(importlib.import_module('ColorTransferLib.Algorithms."+m+"."+m+"'), '"+m+"')")

# for m in available_metrics:
#     exec(m + " = getattr(importlib.import_module('ColorTransferLib.Evaluation."+m+"."+m+"'), '"+m+"')")

# Useful for preventing the status prints from the algorithms which were integrated from public repositories
VAR_BLOCKPRINT = False
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Proxy class for all color transfer algorithms within this project. This class allows the call of the algorithms with
# different kind of input data without preprocessing. The following input data can be processed:
#
# -----------------------------------------
# | Source                 | Reference    |
# -----------------------------------------
# | Image                  | Image        |
# | Image                  | Pointcloud   |
# | Image                  | Mesh         |
# | Pointcloud             | Image        |
# | Pointcloud             | Pointcloud   |
# | Pointcloud             | Mesh         |
# | Video                  | Image        |
# | Video                  | Pointcloud   |
# | Video                  | Mesh         |
# | Volumetric Video       | Image        |
# | Volumetric Video       | Pointcloud   |
# | Volumetric Video       | Mesh         |
# -----------------------------------------
#
# The following approaches are currently supported:
# global - 2001 - Color Transfer between Images
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class ColorTransfer:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, src, ref, approach):
        self.__src = src
        self.__ref = ref

        # check the types of the objects

        self.__out = None
        self.__approach = approach

        with open(os.path.dirname(os.path.abspath(__file__)) + "/Options/" + approach + ".json", 'r') as f:
            options = json.load(f)
            self.__options = BaseOptions(options)


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def apply(self):
        if VAR_BLOCKPRINT: blockPrint()
        m = self.__approach

        try:
            # dynamically import the module
            module = importlib.import_module(f'ColorTransferLib.Algorithms.{m}.{m}')
            
            # Call the class of the module
            cls = getattr(module, m)
            
            # Call the apply method of the class
            self.__out = cls.apply(self.__src, self.__ref, self.__options)
        except ImportError as e:
            print(f"Error while importing the module: {e}")
        except AttributeError as e:
            print(f"Error while retrieving the class or method: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


        if VAR_BLOCKPRINT: enablePrint()
        return self.__out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def set_option(self, key, value):
        if not key in self.__options.get_keys():
            raise RuntimeError("\033[91m" + "Error: Key <" + key + "> does not exist!" + "\033[0m")
        else:
            self.__options.set_option(key, value)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def set_options(self, options):
        self.__options.set_options(options)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_available_methods():
        pass
        # av_methods = []
        # for met in available_methods:
        #     dirname = os.path.dirname(__file__)
        #     filename = os.path.join(dirname, "Options/" + met + ".json")

        #     with open(filename, 'r') as f:
        #         options = json.load(f)

        #     av_m = {
        #         "name": met,
        #         "options": options,
        #         "abstract": getattr(sys.modules[__name__], met).get_info()["abstract"],
        #         "title": getattr(sys.modules[__name__], met).get_info()["title"],
        #         "year": getattr(sys.modules[__name__], met).get_info()["year"],
        #         "types": getattr(sys.modules[__name__], met).get_info()["types"]
        #     }
        #     av_methods.append(av_m)
        # return av_methods



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Proxy class for all color transfer evlauation metric within this project. This class allows the call of the algorithms with
# different kind of input data without preprocessing.
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class ColorTransferEvaluation():
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, src, ref, out):
        self.__src = src
        self.__ref = ref
        self.__out = out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def apply(self, approach):
        m = approach
        try:
            # dynamically import the module
            module = importlib.import_module(f'ColorTransferLib.Evaluation.{m}.{m}')
            
            # Call the class of the module
            cls = getattr(module, m)
            
            ss = copy.deepcopy(self.__src)
            rr = copy.deepcopy(self.__ref)
            oo = copy.deepcopy(self.__out)

            # Call the apply method of the class
            self.__outeval = cls.apply(ss, rr, oo)
        except ImportError as e:
            print(f"Error while importing the module: {e}")
            self.__outeval = "none"
        except AttributeError as e:
            print(f"Error while retrieving the class or method: {e}")
            self.__outeval = "none"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.__outeval = "none"
        return self.__outeval
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    # @staticmethod
    # def get_available_metrics():
    #     return available_metrics