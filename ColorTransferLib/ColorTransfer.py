"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import importlib
from copy import deepcopy
import sys
import json
import os

from ColorTransferLib.Utils.BaseOptions import BaseOptions

# read all available algorithms from the Algorithms folder and import them
available_methods = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/Algorithms")
available_methods.remove("__init__.pyc")
available_methods.remove("__init__.py")
available_methods.remove("__pycache__")
for m in available_methods:
    exec(m + " = getattr(importlib.import_module('ColorTransferLib.Algorithms."+m+"."+m+"'), '"+m+"')")

# read all available metrics from the Evaluation folders
available_metrics = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/Evaluation")
available_metrics.remove("__init__.py")
#available_metrics.remove("__init__.pyc")
available_metrics.remove("__pycache__")

# read all available algorithms from the <algorithms.json> config file and import them
# with open(os.path.dirname(os.path.abspath(__file__)) + "/Config/algorithms.json", 'r') as f:
#     available_methods = json.load(f)
#     for m in available_methods:
#         exec(m + " = getattr(importlib.import_module('ColorTransferLib.Algorithms."+m+"."+m+"'), '"+m+"')")

# read all available metrics from the <metrics.json> config file
# with open(os.path.dirname(os.path.abspath(__file__)) + "/Config/metrics.json", 'r') as f:
#     available_metrics = json.load(f)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Proxy class for all color transfer algorithms within this project. This class allows the call of the algorithms with
# different kind of input data without preprocessing. The following input data can be processed:
#
# ------------------------------
# | Source      | Reference    |
# ------------------------------
# | Image       | Image        |
# | Image       | Pointcloud   |
# | Pointcloud  | Image        |
# | Pointcloud  | Pointcloud   |
# ------------------------------
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
        self.__src_type = src.get_type()
        self.__ref = ref
        self.__ref_type = ref.get_type()

        self.__out = deepcopy(src)
        self.__approach = approach

        with open(os.path.dirname(os.path.abspath(__file__)) + "/Options/" + approach + ".json", 'r') as f:
            options = json.load(f)
            self.__options = BaseOptions(options)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def apply(self):
        self.__out = globals()[self.__approach].apply(self.__src, self.__ref, self.__options)
        return self.__out
        exit()
        if self.__approach == "GlobalColorTransfer":
            src_color = self.__src.get_colors()
            ref_color = self.__ref.get_colors()
            out_colors = GlobalColorTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_colors(out_colors)
        elif self.__approach == "MongeKLColorTransfer":
            src_color = self.__src.get_colors()
            ref_color = self.__ref.get_colors()
            out_colors = MongeKLColorTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_colors(out_colors)
        elif self.__approach == "PdfColorTransfer":
            src_color = self.__src.get_colors()
            ref_color = self.__ref.get_colors()
            out_colors = PdfColorTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_colors(out_colors)
        elif self.__approach == "NeuralStyleTransfer":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = NeuralStyleTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_raw(out_colors)
        elif self.__approach == "HistoGAN":
            src_color = self.__src.get_raw()
            ref_color = self.__ref.get_raw()
            out_colors = HistoGAN.apply(src_color, ref_color, self.__options)
            self.__out.set_raw(out_colors)
        elif self.__approach == "FrequencyColorTransfer":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = FrequencyColorTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_raw(out_colors)
        elif self.__approach == "HistogramAnalogy":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = HistogramAnalogy.apply(src_color, ref_color, self.__options)
            self.__out.set_raw(out_colors)
        elif self.__approach == "CamsTransfer":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = CamsTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_raw(out_colors)
        elif self.__approach == "DeepPhotoStyleTransfer":
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = DeepPhotoStyleTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_raw(out_colors)
        elif self.__approach == "TpsColorTransfer":
            # NOTE RGB space needs multiplication with 255
            src_color = self.__src.get_raw() 
            ref_color = self.__ref.get_raw() 
            out_colors = TpsColorTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_raw(out_colors)
        elif self.__approach == "FuzzyColorTransfer":
            scale = 300
            self.__src.resize(width=scale * self.__src.get_width() // self.__src.get_height(), height=scale)
            self.__ref.resize(width=scale * self.__ref.get_width() // self.__ref.get_height(), height=scale)
            self.__out.resize(width=scale * self.__src.get_width() // self.__src.get_height(), height=scale)
            src_color = self.__src.get_colors()
            ref_color = self.__ref.get_colors()
            out_colors = FuzzyColorTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_colors(out_colors)
        elif self.__approach == "GmmEmColorTransfer":
            scale = 200
            self.__src.resize(width=scale * self.__src.get_width() // self.__src.get_height(), height=scale)
            self.__ref.resize(width=scale * self.__ref.get_width() // self.__ref.get_height(), height=scale)
            self.__out.resize(width=scale * self.__src.get_width() // self.__src.get_height(), height=scale)
            src_color = self.__src.get_raw() * 255.0
            ref_color = self.__ref.get_raw() * 255.0
            out_colors = GmmEmColorTransfer.apply(src_color, ref_color, self.__options)
            self.__out.set_raw(out_colors)
        elif self.__approach == "Eb3dColorTransfer":
            if self.__src.get_type() == "Image":
                print("No support for source with type <Image>.")
                exit(1)
            else:
                out_colors = Eb3dColorTransfer.apply(self.__src, self.__ref, self.__options)
                self.__out.set_colors(out_colors)
        elif self.__approach == "PSNetStyleTransfer":
            if self.__src.get_type() == "Image":
                print("No support for source with type <Image>.")
                exit(1)
            else:
                self.__out = PSNetStyleTransfer.apply(self.__src, self.__ref, self.__options)

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
        av_methods = []
        for met in available_methods:
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, "Options/" + met + ".json")

            with open(filename, 'r') as f:
                options = json.load(f)

            av_m = {
                "name": met,
                "options": options,
                "abstract": getattr(sys.modules[__name__], met).get_info()["abstract"],
                "title": getattr(sys.modules[__name__], met).get_info()["title"],
                "year": getattr(sys.modules[__name__], met).get_info()["year"]
            }
            av_methods.append(av_m)
        return av_methods

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_available_metrics():
        return available_metrics
