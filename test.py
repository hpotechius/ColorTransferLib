"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from ColorTransferLib.DataTypes.Mesh import Mesh
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo

from ColorTransferLib.DataTypes.Image import Image
from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.ColorTransfer import ColorTransfer, ColorTransferEvaluation

from ColorTransferLib.DataTypes.LightField import LightField
from ColorTransferLib.DataTypes.GaussianSplatting import GaussianSplatting

import os


# List of abbreviations for all style transfer methods
st_methods = ["Gatys15", "Luan17", "Cao20", "Afifi21", "Deng22"]
# List of abbreviations for colorization methods
cz_methods = ["Su20", "Ji22", "Kang23"]
# List of abbreviations for all evaluation methods
ev_methods = ["PSNR", "HI", "Corr", "BD", "MSE", "RMSE", "CF", "MSSSIM", "SSIM", "GSSIM", "IVSSIM", "IVEGSSIM", "FSIM", "BRISQUE", "NIQE", "VSI", "CTQM", "LPIPS", "NIMA", "CSS"]

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# METHOD TESTS
# (1) test_all_CT_methods: Test all color transfer methods for a single supported data type, typically image-to-image transfer.
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def test_all_CT_all_datatypes(out_path):
    # List of abbreviations for all color transfer methods
    ct_methods = ["Reinhard01", "Chang03", "Pitie05", "Xiao06", "Reinhard07", "Xiao09", "Qian10", "Grogan19", "Lee20", "Afifi21_2", "Goude21"] 

    print("Loading all files...")
    # Source input
    src_gs = GaussianSplatting(file_path='testdata/gaussiansplatting/plush.splat')
    src_lf = LightField(file_path='testdata/lightfields/legolow_reduced.mp4', size=(3, 3))
    src_vv = VolumetricVideo(folder_path='testdata/volumetricvideos/$volumetric$human', file_name='human')
    src_im = Image(file_path='testdata/images/Mona_Lisa.png')
    src_vd = Video(file_path='testdata/videos/test_vid_00.mp4')
    src_pc = Mesh(file_path='testdata/pointclouds/Statue_Athena.ply', datatype="PointCloud")
    src_me = Mesh(file_path='testdata/meshes/$mesh$Amethyst/Amethyst.obj', datatype="Mesh")

    # Reference input
    ref_gs = GaussianSplatting(file_path='testdata/gaussiansplatting/plush.splat')
    ref_lf = LightField(file_path='testdata/lightfields/legolow_reduced.mp4', size=(3, 3))
    ref_vv = VolumetricVideo(folder_path='testdata/volumetricvideos/$volumetric$human', file_name='human')
    ref_im = Image(file_path='testdata/images/The_Scream.png')
    ref_vd = Video(file_path='testdata/videos/test_vid_00.mp4')
    ref_pc = Mesh(file_path='testdata/pointclouds/Orange.ply', datatype="PointCloud")
    ref_me = Mesh(file_path='testdata/meshes/$mesh$Apple/Apple.obj', datatype="Mesh")

    src_dict = {
        "Image": src_im,
        "Video": src_vd,
        "GaussianSplatting": src_gs,
        "LightField": src_lf,
        "VolumetricVideo": src_vv,
        "PointCloud": src_pc,
        "Mesh": src_me
    }

    ref_dict = {
        "Image": ref_im,
        "Video": ref_vd,
        "GaussianSplatting": ref_gs,
        "LightField": ref_lf,
        "VolumetricVideo": ref_vv,
        "PointCloud": ref_pc,
        "Mesh": ref_me
    }

    type_src_array = ["Image", "Video", "GaussianSplatting", "LightField", "VolumetricVideo", "PointCloud", "Mesh"]
    type_ref_array = ["Image", "GaussianSplatting", "PointCloud", "Mesh"]

    for method in ct_methods:
        print("--------------------------------------")
        print("- Method: " + method)
        print("--------------------------------------")
        # Create output directory
        if not os.path.exists(out_path + "/" + method):
            os.makedirs(out_path + "/" + method)


        for src_type in type_src_array:
            for ref_type in type_ref_array:
                print("Processing: " + method + " - " + src_type + " -> " + ref_type + " ...", end=' ')
                src = src_dict[src_type]
                ref = ref_dict[ref_type]

                ct = ColorTransfer(src, ref, method)
                out = ct.apply()

                #print(out)
                #exit()

                if out["status_code"] == 0:
                    out["object"].write(f"{out_path}/{method}/"+ method + "_" + src_type + "_" + ref_type)
                    print("\033[92mDone\033[0m")
                else:
                    print(f'\033[93m{out["response"]}\033[0m')


# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def test_all_ST_all_datatypes(out_path):
    # List of abbreviations for all color transfer methods
    ct_methods = ["Gatys15", "Luan17", "Cao20", "Afifi21", "Deng22"]

    print("Loading all files...")
    # Source input
    src_gs = GaussianSplatting(file_path='testdata/gaussiansplatting/plush.splat')
    src_lf = LightField(file_path='testdata/lightfields/legolow_reduced.mp4', size=(3, 3))
    src_vv = VolumetricVideo(folder_path='testdata/volumetricvideos/$volumetric$human', file_name='human')
    src_im = Image(file_path='testdata/images/Mona_Lisa.png')
    src_vd = Video(file_path='testdata/videos/test_vid_00.mp4')
    src_pc = Mesh(file_path='testdata/pointclouds/Statue_Athena.ply', datatype="PointCloud")
    src_me = Mesh(file_path='testdata/meshes/$mesh$Amethyst/Amethyst.obj', datatype="Mesh")

    # Reference input
    ref_gs = GaussianSplatting(file_path='testdata/gaussiansplatting/plush.splat')
    ref_lf = LightField(file_path='testdata/lightfields/legolow_reduced.mp4', size=(3, 3))
    ref_vv = VolumetricVideo(folder_path='testdata/volumetricvideos/$volumetric$human', file_name='human')
    ref_im = Image(file_path='testdata/images/The_Scream.png')
    ref_vd = Video(file_path='testdata/videos/test_vid_00.mp4')
    ref_pc = Mesh(file_path='testdata/pointclouds/Orange.ply', datatype="PointCloud")
    ref_me = Mesh(file_path='testdata/meshes/$mesh$Apple/Apple.obj', datatype="Mesh")

    src_dict = {
        "Image": src_im,
        "Video": src_vd,
        "GaussianSplatting": src_gs,
        "LightField": src_lf,
        "VolumetricVideo": src_vv,
        "PointCloud": src_pc,
        "Mesh": src_me
    }

    ref_dict = {
        "Image": ref_im,
        "Video": ref_vd,
        "GaussianSplatting": ref_gs,
        "LightField": ref_lf,
        "VolumetricVideo": ref_vv,
        "PointCloud": ref_pc,
        "Mesh": ref_me
    }

    type_src_array = ["Image", "Video", "GaussianSplatting", "LightField", "VolumetricVideo", "PointCloud", "Mesh"]
    type_ref_array = ["Image", "GaussianSplatting", "PointCloud", "Mesh"]


    for method in ct_methods:
        print("--------------------------------------")
        print("- Method: " + method)
        print("--------------------------------------")
        # Create output directory
        if not os.path.exists(out_path + "/" + method):
            os.makedirs(out_path + "/" + method)


        for src_type in type_src_array:
            for ref_type in type_ref_array:
                print("Processing: " + method + " - " + src_type + " -> " + ref_type + " ...", end=' ')
                src = src_dict[src_type]
                ref = ref_dict[ref_type]

                ct = ColorTransfer(src, ref, method)
                out = ct.apply()

                #print(out)
                #exit()

                if out["status_code"] == 0:
                    out["object"].write(f"{out_path}/{method}/"+ method + "_" + src_type + "_" + ref_type)
                    print("\033[92mDone\033[0m")
                else:
                    print(f'\033[93m{out["response"]}\033[0m')

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def test_all_CZ_all_datatypes(out_path):
    # List of abbreviations for all color transfer methods
    ct_methods = ["Su20", "Ji22", "Kang23"] 

    print("Loading all files...")
    # Source input
    src_im = Image(file_path='testdata/images/Mona_Lisa.png')

    # Reference input
    ref_im = Image(file_path='testdata/images/The_Scream.png')

    src_dict = {
        "Image": src_im,
    }

    ref_dict = {
        "Image": ref_im,
    }

    type_src_array = ["Image"]
    type_ref_array = ["Image"]


    for method in ct_methods:
        print("--------------------------------------")
        print("- Method: " + method)
        print("--------------------------------------")
        # Create output directory
        if not os.path.exists(out_path + "/" + method):
            os.makedirs(out_path + "/" + method)


        for src_type in type_src_array:
            for ref_type in type_ref_array:
                print("Processing: " + method + " - " + src_type + " -> " + ref_type + " ...", end=' ')
                src = src_dict[src_type]
                ref = ref_dict[ref_type]

                ct = ColorTransfer(src, ref, method)
                out = ct.apply()

                #print(out)
                #exit()

                if out["status_code"] == 0:
                    out["object"].write(f"{out_path}/{method}/"+ method + "_" + src_type + "_" + ref_type)
                    print("\033[92mDone\033[0m")
                else:
                    print(f'\033[93m{out["response"]}\033[0m')

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def test_all_EVAL():
    # List of abbreviations for all evaluation methods
    ev_methods = ["PSNR", "HI", "Corr", "BD", "MSE", "RMSE", "CF", "MSSSIM", "SSIM", "GSSIM", "IVSSIM", "IVEGSSIM", "FSIM", "BRISQUE", "NIQE", "VSI", "CTQM", "LPIPS", "NIMA", "CSS"]

    for method in ev_methods:
        print("Processing: " + method + " ...", end=' ')
        src = Image(file_path='testdata/eval_test/Mona_Lisa.png')
        ref = Image(file_path='testdata/eval_test/The_Scream.png')
        out = Image(file_path='testdata/eval_test/GLO_Image_Image.png')

        cte = ColorTransferEvaluation(src, ref, out)
        eva = cte.apply(method)
        print(eva)
