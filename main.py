"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
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

import cv2
import numpy as np
import os
import glob


from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def test_colorTransfer_all_combinations(method):
    src_gs = GaussianSplatting(file_path='/home/potechius/Code/ColorTransferLib/testdata/gaussiansplatting/plush.splat')
    src_lf = LightField(file_path='/home/potechius/Code/ColorTransferLib/testdata/lightfields/amethyst.mp4', size=(17, 17))
    src_vv = VolumetricVideo(folder_path='/home/potechius/Code/ColorTransferLib/testdata/volumetricvideos/$volumetric$human', file_name='human')
    # src_im = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-00.png')
    src_im = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/Mona_Lisa.png')
    src_vd = Video(file_path='/home/potechius/Code/ColorTransferLib/testdata/videos/earth.mp4')
    src_pc = Mesh(file_path='/home/potechius/Code/ColorTransferLib/testdata/pointclouds/Azurit.ply', datatype="PointCloud")
    src_me = Mesh(file_path='/home/potechius/Code/ColorTransferLib/testdata/meshes/$mesh$Amethyst/Amethyst.obj', datatype="Mesh")

    ref_gs = GaussianSplatting(file_path='/home/potechius/Code/ColorTransferLib/testdata/gaussiansplatting/plush.splat')
    ref_lf = LightField(file_path='/home/potechius/Code/ColorTransferLib/testdata/lightfields/legolow.mp4', size=(17, 17))
    ref_vv = VolumetricVideo(folder_path='/home/potechius/Code/ColorTransferLib/testdata/volumetricvideos/$volumetric$human', file_name='human')
    # ref_im = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-06.png')
    ref_im = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/The_Scream.png')
    ref_vd = Video(file_path='/home/potechius/Code/ColorTransferLib/testdata/videos/bunny.mp4')
    ref_pc = Mesh(file_path='/home/potechius/Code/ColorTransferLib/testdata/pointclouds/Violin.ply', datatype="PointCloud")
    ref_me = Mesh(file_path='/home/potechius/Code/ColorTransferLib/testdata/meshes/$mesh$Apple/Apple.obj', datatype="Mesh")

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

    # type_src_array = ["Image", "Video", "GaussianSplatting", "LightField", "VolumetricVideo", "PointCloud", "Mesh"]
    # type_ref_array = ["Image", "GaussianSplatting", "PointCloud", "Mesh"]
    type_src_array = ["Image"]
    type_ref_array = ["Image"]

    for src_type in type_src_array:
        for ref_type in type_ref_array:
            print("Processing: " + method + " - " + src_type + " -> " + ref_type)
            src = src_dict[src_type]
            ref = ref_dict[ref_type]

            ct = ColorTransfer(src, ref, method)
            out = ct.apply()

            if out["status_code"] == 0:
                out["object"].write("/home/potechius/Code/ColorTransferLib/testdata/results/out_"+ method + "_" + src_type + "_" + ref_type)
                # out["object"].write("/home/potechius/Code/ColorTransferLib/testdata/results/test")
            else:
                print("Error: " + out["response"])
            print("Processed: " + method + " - " + src_type + " -> " + ref_type)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def test_pointcloud_colorTransfer(method):
    src = Mesh(file_path='/home/potechius/Code/ColorTransferLib/testdata/pointclouds/Statue_Athena.ply', datatype="PointCloud")
    # ref = Mesh(file_path='/home/potechius/Code/ColorTransferLib/testdata/pointclouds/Statue_Lion.ply', datatype="PointCloud")
    ref = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/The_Kiss.png')

    ct = ColorTransfer(src, ref, method)
    out = ct.apply()

    if out["status_code"] == 0:
        out["object"].write("/home/potechius/Code/ColorTransferLib/testdata/results/out_" + method)
    else:
        print("Error: " + out["response"])

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def test_evaluation_all():
    pass

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def test_image_colorTransfer(method):
    print("Processing: " + method)
    src = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-06.png')
    ref = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-00.png')

    ct = ColorTransfer(src, ref, method)
    out = ct.apply()

    if out["status_code"] == 0:
        out["object"].write("/home/potechius/Code/ColorTransferLib/testdata/results/out_" + method)
    else:
        print("Error: " + out["response"])
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def test_image_evaluation(method):
    src = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-00.png')
    ref = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-06.png')
    out = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-00.png')

    cte = ColorTransferEvaluation(src, ref, out)
    eva = cte.apply(method)
    print(eva)
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def apply_color_transfer_localy(src, ref, method):
    pass

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':  
 
    # test_colorTransfer_all_combinations("ST2")
    # exit()

    # 
    # test_pointcloud_colorTransfer("PSN")




    # done = ["GPC", "FUZ", "GLO", "CCS", "DPT", "BCC", "HIS", "MKL", "TPS", "NST", "PDF", "RHG", "CAM"]
    # done = ["CFM"]
    # for elem in done:
    #     test_image_colorTransfer(elem)
    # exit()




    from ColorTransferLib.Algorithms.GLO.GLO import GLO
    from ColorTransferLib.Utils.BaseOptions import BaseOptions
    import json
    with open("/home/potechius/Code/ColorTransferLib/ColorTransferLib/Options/GLO.json", 'r') as f:
        options = json.load(f)
        opt = BaseOptions(options)
    # src = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/Mona_Lisa.png')
    src = Video(file_path='/home/potechius/Code/ColorTransferLib/testdata/videos/output.mp4')
    ref = Image(file_path='/home/potechius/Code/ColorTransferLib/testdata/images/256_interior-00.png')
    out = GLO.apply(src, ref, opt)    
    if out["status_code"] == 0:
        out["object"].write("/home/potechius/Code/ColorTransferLib/testdata/results/out_GLO")
    else:
        print("Error: " + out["response"])





    # from ColorTransferLib.Algorithms.PSN.PSN import PSN
    # from ColorTransferLib.Utils.BaseOptions import BaseOptions
    # import json
    # with open("/home/potechius/Code/ColorTransferLib/ColorTransferLib/Options/PSN.json", 'r') as f:
    #     options = json.load(f)
    #     opt = BaseOptions(options)
    # src = Mesh(file_path='/home/potechius/Code/ColorTransferLib/testdata/pointclouds/Orange.ply', datatype="PointCloud")
    # ref = Mesh(file_path='/home/potechius/Code/ColorTransferLib/testdata/pointclouds/Orange.ply', datatype="PointCloud")
    # oo = PSN.apply(src, ref, opt)
    # print(oo)


    # done = ["PSN"]
    # for elem in done:
    #     test_pointcloud_colorTransfer(elem)


    # test_image_evaluation("NIMA")
    exit()

    #src = Video(file_path='/media/potechius/External/data/Videos/sample-5s.mp4')


    #src = VolumetricVideo(folder_path='/media/potechius/External/data/VolumetricVideos/$volumetric$Human', file_name='Human')

    #src.write("/media/potechius/External/data/VolumetricVideos/out")
    #exit()

    #src.write("/media/potechius/External/data/Videos/out.mp4")

    #frames = src.get_raw()
    # for i, frame in enumerate(frames):
    #     print(frame.dtype)
    #     cv2.imshow(f'Frame {i}', frame)
    #     # Wait for 'interval' seconds
    #     if cv2.waitKey(int(1000)) & 0xFF == ord('q'):
    #         break

    # cv2.destroyAllWindows()


    #exit()


    # Evaluation Example
    # TODO
