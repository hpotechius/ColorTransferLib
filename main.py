"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from ColorTransferLib.MeshProcessing.Mesh import Mesh
from ColorTransferLib.ImageProcessing.Image import Image
from ColorTransferLib.ColorTransfer import ColorTransfer, ColorTransferEvaluation



# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':  
    # 2D Color/Style Transfer Example
    src = Image(file_path='/media/potechius/External/data/Images/Wanderer_above_the_Sea_of_Fog.png')
    ref = Image(file_path='/media/potechius/External/data/Images/The_Scream.png')  
    #out = Image(file_path='/media/potechius/External/data/Images/out.png')  
    
    #cte = ColorTransferEvaluation(src, ref, out)
    #eva = cte.apply("VSI")
    #print(eva)

    #exit()

    #src = Mesh(file_path='/home/potechius/Downloads/3D/src.ply', datatype="PointCloud")
    #ref = Mesh(file_path='/home/potechius/Downloads/3D/ref.ply', datatype="PointCloud")  

    #src = Mesh(file_path='/home/potechius/Downloads/3D_mesh/Apple.obj', datatype="Mesh")
    #ref = Mesh(file_path='/home/potechius/Downloads/3D_mesh/Pillow.obj', datatype="Mesh")  


    algo = "GLO"
    ct = ColorTransfer(src, ref, algo)
    out = ct.apply()

    if out["status_code"] == 0:
        out["object"].write("/media/potechius/External/data/Images/out")
    else:
        print("Error: " + out["response"])
    print("Done")

    # Evaluation Example
    # TODO


