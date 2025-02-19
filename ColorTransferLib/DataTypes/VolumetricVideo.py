"""
Copyright 2024 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""
import numpy as np
import open3d as o3d
import os
from ColorTransferLib.DataTypes.Mesh import Mesh

class VolumetricVideo:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # datatype -> [PointCloud, Mesh]
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, folder_path=None, file_name=None, meshes=None):
        self.__type = "VolumetricVideo"
        self.__numMeshes = 0
        self.__meshes = []
        self.__file_name = file_name

        if folder_path==None and meshes!=None:
            self.__meshes = meshes
            self.__numMeshes = len(meshes)
            return

        # Verbinde files_path und file_name zu einem Pfad
        full_path = os.path.join(folder_path, file_name)

        for i in range(1800):
            i_str = str(i).zfill(5)
            file_path = f"{full_path}_{i_str}.obj"

            if os.path.exists(file_path):
                self.__numMeshes += 1
                print(f"Loading {file_path}")
                mesh = Mesh(file_path=file_path, datatype="Mesh")
                self.__meshes.append(mesh)
            else:
                break

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Writes the mesh to the specified path
    # ------------------------------------------------------------------------------------------------------------------
    def write(self, path):
        print(path)
        new_file_name = path.split("/")[-1]
        out_folder, _ = os.path.split(path)

        out_folder += "/$volumetric$" + path.split("/")[-1]

        # Erstelle den Ordner, falls er nicht existiert
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            print(f"Ordner {out_folder} wurde erstellt.")
        else:
            print(f"Ordner {out_folder} existiert bereits.")



        for i in range(self.__numMeshes):
            i_str = str(i).zfill(5)
            file_path = f"{out_folder}/{new_file_name}_{i_str}"

            # opne3d saves the textures of obj files with a "_0", "_1" etc ending, because multiple textures are
            # possible -> this ending has to be removed from the png file and within the mtl file.
            o3d.io.write_triangle_mesh(file_path + ".obj", self.__meshes[i].get_mesh())
            img_path = file_path + "_0.png"
            new_img_path = file_path + ".jpg"
            file_name = file_path.split("/")[-1]
            os.rename(img_path, new_img_path)

            mtl_path = file_path + ".mtl"
            readFile = open(mtl_path, "r")
            data = readFile.read()
            data = data.replace(file_name + "_0.png", file_name + ".jpg")
            writeFile = open(mtl_path, "w")
            writeFile.write(data)

    def set_file_name(self, file_name):
        self.__file_name = file_name
                
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
        

    def get_type(self):
        return self.__type
    

    def get_file_name(self):
        return self.__file_name
    
    # ------------------------------------------------------------------------------------------------------------------
    # Returns the colors of all vertices as numpy array with shape (len(vertices), 1, 3). Necessary for the
    # ColorTransferLib
    # ------------------------------------------------------------------------------------------------------------------
    def get_colors(self):
        return [mesh.get_colors() for mesh in self.__meshes]

    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def get_raw(self):
        return [mesh.get_raw() for mesh in self.__meshes]

    # ------------------------------------------------------------------------------------------------------------------
    # returns the image 
    # ------------------------------------------------------------------------------------------------------------------
    def get_meshes(self):
        return self.__meshes