"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import os
import cv2
import matplotlib.pyplot as plt
import gdown
import zipfile36 as zipfile


from ColorTransferLib.MeshProcessing.Mesh import Mesh
from ColorTransferLib.ImageProcessing.Image import Image
from ColorTransferLib.ColorTransfer import ColorTransfer, ColorTransferEvaluation



# ------------------------------------------------------------------------------------------------------------------
# Downloads necessary objects, e.g., network models, to the correct location 
# ------------------------------------------------------------------------------------------------------------------
def downloadMaterial():
    if not os.path.exists("Models") and not os.path.exists("data"):
        print("Download DATA.zip ...")
        url = "https://drive.google.com/file/d/1ShJpPFJ9vCu5Vb7FJk7aSnn7ywFiT0GJ/view?usp=share_link"
        output_path = 'DATA.zip'
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        # Extract DATA.zip
        print("Extract DATA.zip ...")
        with zipfile.ZipFile("DATA.zip","r") as zip_ref:
            zip_ref.extractall()
        # Delete DATA.zip
        print("Delete DATA.zip ...")
        os.remove("DATA.zip")

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':  
    # 2D Color/Style Transfer Example
    src = Image(file_path='/home/potechius/Downloads/2D/src.png')
    ref = Image(file_path='/home/potechius/Downloads/2D/ref.png')  
    out = Image(file_path='/home/potechius/Downloads/2D/result_GLO.png')  
    
    cte = ColorTransferEvaluation(src, ref, out)
    eva = cte.apply("VSI")
    print(eva)

    exit()

    #src = Mesh(file_path='/home/potechius/Downloads/3D/src.ply', datatype="PointCloud")
    #ref = Mesh(file_path='/home/potechius/Downloads/3D/ref.ply', datatype="PointCloud")  

    #src = Mesh(file_path='/home/potechius/Downloads/3D_mesh/Apple.obj', datatype="Mesh")
    #ref = Mesh(file_path='/home/potechius/Downloads/3D_mesh/Pillow.obj', datatype="Mesh")  


    algo = "GPC"
    ct = ColorTransfer(src, ref, algo)
    out = ct.apply()

    if out["status_code"] == 0:
        out["object"].write("/home/potechius/Downloads/3D_mesh/result_"+algo)
    else:
        print("Error: " + out["response"])
    print("Done")

    # Evaluation Example
    # TODO


