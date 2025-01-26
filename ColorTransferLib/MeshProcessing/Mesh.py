"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""
import numpy as np
import open3d as o3d
import os
import cv2

class Mesh:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # datatype -> [PointCloud, Mesh]
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, file_path=None, datatype=None):
        self.__type = datatype

        # vertex properties
        self.__vertices_enabled = False
        self.__vnormals_enabled = False
        self.__vcolors_enabled = False
        self.__vertex_positions = []
        self.__vertex_normals = []
        self.__vertex_colors = []
        self.__num_vertices = 0

        self.__pcd = None

        # face properties
        self.__faces_enabled = False
        self.__fnormals_enabled = False
        self.__faces = []
        self.__num_faces = 0

        # texture properties
        self.__texture_enabled = False
        self.__texture = None
        self.__texture_mask = None
        self.__texture_size = (0,0,0)
        self.__texture_uvs = []
    
        if datatype == "PointCloud":
            self.__init_pointcloud(file_path)
        elif datatype == "Mesh":
            self.__init_mesh(file_path)


    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def __init_mesh(self, file_path):
        self.__pcd = o3d.io.read_triangle_mesh(file_path)

        self.__vertices_enabled = self.__pcd.has_vertices()
        self.__vcolors_enabled = self.__pcd.has_vertex_colors()
        self.__vnormals_enabled = self.__pcd.has_vertex_normals()

        self.__vertex_positions = np.asarray(self.__pcd.vertices) if self.__vertices_enabled else []
        self.__vertex_colors = np.asarray(self.__pcd.vertex_colors).astype("float32") if self.__vcolors_enabled else []
        self.__vertex_normals = np.asarray(self.__pcd.vertex_normals) if self.__vnormals_enabled else []

        self.__num_vertices = self.__vertex_positions.shape[0] if self.__vertices_enabled else 0


        self.__faces_enabled = self.__pcd.has_triangles()
        self.__fnormals_enabled = self.__pcd.has_triangle_normals()

        self.__face_positions = np.asarray(self.__pcd.triangles) if self.__faces_enabled else []
        self.__face_normals = np.asarray(self.__pcd.triangle_normals) if self.__fnormals_enabled else []

        self.__num_faces = self.__face_positions.shape[0] if self.__faces_enabled else 0

        texture_path = file_path.split(".")[0] + ".png"
        print(texture_path)
        if os.path.isfile(texture_path):
            self.__pcd.textures =  [o3d.io.read_image(texture_path).flip_vertical()]
        else:
            texture_path = file_path.split(".")[0] + ".jpg"
            if os.path.isfile(texture_path):
                self.__pcd.textures =  [o3d.io.read_image(texture_path).flip_vertical()]

        self.__texture_enabled = self.__pcd.has_textures()
        self.__texture = np.asarray(self.__pcd.textures[0]).astype("float32") / 255 if self.__texture_enabled else None

        # remove alpha channel
        if self.__texture.shape[2] == 4:
            self.__texture = self.__texture[:,:,:3]

        self.__texture_size = np.asarray(self.__texture.shape) if self.__texture_enabled else None
        self.__texture_uvs = self.__pcd.triangle_uvs

        # set all material ids to 0 because they are per default: 1
        self.__pcd.triangle_material_ids = o3d.utility.IntVector(np.asarray(self.__pcd.triangle_material_ids) * 0)

        # Create a mask from the texture using UV mapping
        # if self.__texture_enabled:
        #     self.__mask = self.__create_mask_from_uv_mapping(self.__texture, self.__texture_uvs, self.__face_positions, self.__texture_size)

        #     flipped_mask = cv2.flip(self.__mask, 0)
        #     cv2.imwrite("/home/potechius/Code/ColorTransferLib/testdata/results/ou.png", flipped_mask)
        #     cv2.imwrite("/home/potechius/Code/ColorTransferLib/testdata/results/ou2.png", self.__texture * 255)
        # else:
        #     self.__mask = None

    # def __create_mask_from_uv_mapping(self, texture, uvs, faces, texture_size, scale_factor=1.1):
    #     mask = np.zeros((texture_size[0], texture_size[1]), dtype=np.uint8)
    #     uvs = (uvs * np.array([texture_size[1], texture_size[0]])).astype(int)
    #     for i in range(0, len(faces)):
    #         pts = uvs[faces[i]].reshape((-1, 2))
    #         centroid = np.mean(pts, axis=0)
    #         scaled_pts = (pts - centroid) * scale_factor + centroid
    #         scaled_pts = scaled_pts.astype(int).reshape((-1, 1, 2))
    #         cv2.fillPoly(mask, [scaled_pts], 255)
    #     return mask

    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    def __init_pointcloud(self, file_path):
        self.__pcd = o3d.io.read_point_cloud(file_path)

        self.__vertices_enabled = self.__pcd.has_points()
        self.__vcolors_enabled = self.__pcd.has_colors()
        self.__vnormals_enabled = self.__pcd.has_normals()

        self.__vertex_positions = np.asarray(self.__pcd.points) if self.__vertices_enabled else []
        self.__vertex_colors = np.asarray(self.__pcd.colors).astype("float32") if self.__vcolors_enabled else []
        self.__vertex_normals = np.asarray(self.__pcd.normals) if self.__vnormals_enabled else []

        self.__num_vertices = self.__vertex_positions.shape[0] if self.__vertices_enabled else 0

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Writes the mesh to the specified path
    # ------------------------------------------------------------------------------------------------------------------
    def write(self, path):
        if self.__type == "PointCloud":
            o3d.io.write_point_cloud(path + ".ply", self.__pcd)
        elif self.__type == "Mesh":
            # opne3d saves the textures of obj files with a "_0", "_1" etc ending, because multiple textures are
            # possible -> this ending has to be removed from the png file and within the mtl file.
            o3d.io.write_triangle_mesh(path + ".obj", self.__pcd)
            img_path = path + "_0.png"
            new_img_path = path + ".png"
            file_name = path.split("/")[-1]
            os.rename(img_path, new_img_path)

            mtl_path = path + ".mtl"
            readFile = open(mtl_path, "r")
            data = readFile.read()
            data = data.replace(file_name + "_0.png", file_name + ".png")
            writeFile = open(mtl_path, "w")
            writeFile.write(data)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # GETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def get_mesh(self):
        return self.__pcd

    # ------------------------------------------------------------------------------------------------------------------
    # Returns if vertex colors are available
    # ------------------------------------------------------------------------------------------------------------------
    def has_vertex_colors(self):
        return self.__vcolors_enabled
    
    # ------------------------------------------------------------------------------------------------------------------
    # Returns if vertex normals are available
    # ------------------------------------------------------------------------------------------------------------------
    def has_vertex_normals(self):
        return self.__vnormals_enabled

    # ------------------------------------------------------------------------------------------------------------------
    # Returns if vertex normals are available
    # ------------------------------------------------------------------------------------------------------------------
    def has_vertex_colors(self):
        return self.__vcolors_enabled

    # ------------------------------------------------------------------------------------------------------------------
    # Returns the positions of all vertices as numpy array with shape (len(vertices), 1, 3).
    # ------------------------------------------------------------------------------------------------------------------
    def get_vertex_positions(self):
        return self.__vertex_positions
        # list_pos = [vertex.get_position() for vertex in self.__vertices]
        # numpy_pos = np.asarray(list_pos, dtype=np.float32).reshape(len(list_pos), 1, 3)
        # return numpy_pos

    # ------------------------------------------------------------------------------------------------------------------
    # Returns the colors of all vertices as numpy array with shape (len(vertices), 1, 3). Alpha channel will be
    # ignored.
    # ------------------------------------------------------------------------------------------------------------------
    def get_vertex_colors(self):
        return self.__vertex_colors  
        # list_color = [vertex.get_color()[:-1] for vertex in self.__vertices]
        # numpy_color = np.asarray(list_color, dtype=np.float32).reshape(len(list_color), 1, 3)
        # return numpy_color


    # ------------------------------------------------------------------------------------------------------------------
    # Returns the colors of all vertices as numpy array with shape (len(vertices), 1, 3). Necessary for the
    # ColorTransferLib
    # ------------------------------------------------------------------------------------------------------------------
    def get_colors(self):
        if self.__type == "PointCloud":
            return np.expand_dims(self.__vertex_colors, 1)
        elif self.__type == "Mesh":
            tex_height = self.__texture_size[0]
            tex_width = self.__texture_size[1]
            tex_channel = self.__texture_size[2]
            # print(tex_height, tex_width, tex_channel)

            reshaped_texture = self.__texture.reshape(tex_width * tex_height, 1, tex_channel)
            return reshaped_texture

    # ------------------------------------------------------------------------------------------------------------------
    # TEMPORARY because some color transfer algorithms need this
    # ------------------------------------------------------------------------------------------------------------------
    def get_raw(self):
        if self.__type == "PointCloud":
            return np.expand_dims(self.__vertex_colors, 1)
        elif self.__type == "Mesh":
            tex_channel = self.__texture_size[2]
            return np.resize(self.__texture,(512,512,tex_channel))
            # Note: Resizing leads to a wrongs ordered pixel values
            # return np.resize(self.__texture,(256,256,3))
    # ------------------------------------------------------------------------------------------------------------------
    # TEMPORARY because some color transfer algorithms need this
    # ------------------------------------------------------------------------------------------------------------------
    def set_raw(self, colors, normalized=False):
        if not normalized:
            colors /= 255
        if self.__type == "PointCloud":
            self.__pcd.colors = o3d.utility.Vector3dVector(np.squeeze(colors))
        elif self.__type == "Mesh":
            # Prevents: RuntimeError: Image can only be initialized from c-style buffer.
            colors = np.asarray(colors, order="C")
            self.__texture = colors
            self.__pcd.textures = [o3d.geometry.Image((colors*255).astype("uint8"))]
        
    # ------------------------------------------------------------------------------------------------------------------
    # Returns the normals of all vertices as numpy array with shape (len(vertices), 1, 3).
    # ------------------------------------------------------------------------------------------------------------------
    def get_vertex_normals(self):
        return self.__vertex_normals
        # list_normal = [vertex.get_normal() for vertex in self.__vertices]
        # numpy_normal = np.asarray(list_normal, dtype=np.float32).reshape(len(list_normal), 1, 3)
        # return numpy_normal
    
    # ------------------------------------------------------------------------------------------------------------------
    # Returns the faces
    # ------------------------------------------------------------------------------------------------------------------
    def get_faces(self):
        return self.__faces
    
    # ------------------------------------------------------------------------------------------------------------------
    # Returns if texture is available
    # ------------------------------------------------------------------------------------------------------------------
    def has_texture(self):
        return self.__texture_enabled
    
    # ------------------------------------------------------------------------------------------------------------------
    # Returns texture image
    # ------------------------------------------------------------------------------------------------------------------
    def get_texture(self):
        return self.__texture

    # ------------------------------------------------------------------------------------------------------------------
    # Returns if face normals are available
    # ------------------------------------------------------------------------------------------------------------------
    def has_face_normals(self):
        return self.__fnormals_enabled

    # ------------------------------------------------------------------------------------------------------------------
    # Returns the number of vertices
    # ------------------------------------------------------------------------------------------------------------------
    def get_num_vertices(self):
        return self.__num_vertices

    # ------------------------------------------------------------------------------------------------------------------
    # Returns the number of faces
    # ------------------------------------------------------------------------------------------------------------------
    def get_num_faces(self):
        return self.__num_faces

    # ------------------------------------------------------------------------------------------------------------------
    # Returns the type of object
    # ------------------------------------------------------------------------------------------------------------------
    def get_type(self):
        return self.__type

    # ------------------------------------------------------------------------------------------------------------------
    # returns the color histogram, mean and variance
    # ------------------------------------------------------------------------------------------------------------------
    def get_color_statistic(self):
        if self.__type == "Mesh":
            color = np.reshape(self.__texture, (self.__texture.shape[0]*self.__texture.shape[1], self.__texture.shape[2]))
        elif self.__type == "PointCloud":
            color = self.__vertex_colors

        rgb_c = (color * 255.0).astype(np.int)
        histo_red = np.asarray(np.histogram(rgb_c[:,0], bins=np.arange(257))[0]).reshape(256,1)
        histo_green = np.asarray(np.histogram(rgb_c[:,1], bins=np.arange(257))[0]).reshape(256,1)
        histo_blue = np.asarray(np.histogram(rgb_c[:,2], bins=np.arange(257))[0]).reshape(256,1)
        histo = np.concatenate((histo_red, histo_green, histo_blue), axis=1)
        mean = np.mean(rgb_c, axis=0).astype(np.int)
        std = np.std(rgb_c, axis=0).astype(np.int)
        return histo, mean, std
    
    # ------------------------------------------------------------------------------------------------------------------
    # returns 
    # ------------------------------------------------------------------------------------------------------------------
    def get_color_distribution(self):
        if self.__type == "Mesh":
            tex_height = self.__texture_size[0]
            tex_width = self.__texture_size[1]
            tex_channel = self.__texture_size[2]
            color = self.__texture.reshape(tex_height * tex_width, tex_channel)
        elif self.__type == "PointCloud":
            color = self.__vertex_colors

        color = color[np.random.randint(color.shape[0], size=5000), :]

        rgb_c = (color * 255).astype(np.int)
        rgb_c = np.unique(rgb_c, axis=0)
        return rgb_c

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # SETTER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # <positions> have to be provided in shape (len(vertices), 3)
    # ------------------------------------------------------------------------------------------------------------------
    def set_vertex_positions(self, points):
        self.__pcd.points = o3d.utility.Vector3dVector(points)


    # ------------------------------------------------------------------------------------------------------------------
    # <colors> have to be provided in shape (len(vertices), 3)
    # ------------------------------------------------------------------------------------------------------------------
    def set_vertex_colors(self, colors):
        self.__pcd.colors = o3d.utility.Vector3dVector(colors)

    # ------------------------------------------------------------------------------------------------------------------
    # <colors> have to be provided in shape (len(vertices), 1, 3) -> Necessary for the ColorTransferLib 
    # ------------------------------------------------------------------------------------------------------------------
    def set_colors(self, colors):
        if self.__type == "PointCloud":   
            self.__pcd.colors = o3d.utility.Vector3dVector(colors.squeeze())
        elif self.__type == "Mesh":
            tex_height = self.__texture_size[0]
            tex_width = self.__texture_size[1]
            tex_channel = self.__texture_size[2]
            colors = np.asarray(colors, order="C")
            self.__pcd.textures = [o3d.geometry.Image((colors.reshape(tex_height, tex_width, tex_channel) * 255).astype("uint8"))]
            self.__texture = np.asarray(self.__pcd.textures[0]).astype("float32") / 255 if self.__texture_enabled else None


    # ------------------------------------------------------------------------------------------------------------------
    # <normals> have to be provided in shape (len(vertices), 3)
    # ------------------------------------------------------------------------------------------------------------------
    def set_vertex_normals(self, normals):
        self.__pcd.normals = o3d.utility.Vector3dVector(normals)

    # ------------------------------------------------------------------------------------------------------------------
    # ... 
    # ------------------------------------------------------------------------------------------------------------------
    def set_faces(self, faces):
        pass
        # self.__faces = faces
        # self.__faces_enabled = True
        # self.__num_faces = len(faces)


    # ------------------------------------------------------------------------------------------------------------------
    # RGB color space is quantized in a 10x10x10 area 
    # ------------------------------------------------------------------------------------------------------------------
    def get_3D_color_histogram(self):
        if self.__type == "PointCloud":   
            cols = self.__vertex_colors
        elif self.__type == "Mesh":
            tex_height = self.__texture_size[0]
            tex_width = self.__texture_size[1]
            tex_channel = self.__texture_size[2]
            cols = self.__texture.reshape(tex_width * tex_height, tex_channel)
        upd = np.clip(np.floor(cols * 10).astype(np.int8), 0, 9)
        # numpy unique is slow -> maybe optimizaiton possible
        uni = np.unique(upd, axis=0, return_counts=True)
        con = np.concatenate((uni[0], uni[1].reshape((-1, 1))), axis=1)
        return con

    # ------------------------------------------------------------------------------------------------------------------
    # create voxelgrid from given point cloud 
    # ------------------------------------------------------------------------------------------------------------------
    def get_voxel_grid(self, voxel_level):
        scale_f = voxel_level
        print(voxel_level)
        # Initialize a point cloud object
        pcd = self.__pcd
        # fit to unit cube
        sc_f_min = pcd.get_min_bound()
        sc_f_max = pcd.get_max_bound()
        sc_f = np.max(sc_f_max - sc_f_min)
        #pcd.scale(1 / sc_f, center=pcd.get_center())
        # Create a voxel grid from the point cloud with a voxel_size of 0.01
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=scale_f * sc_f)

        #voxel_grid.scale(scale=57.0)

        voxelret = {
            "centers": [],#np.empty((0,3), np.float32),
            "colors": []#np.empty((0,3), np.float32)
        }

        voxels = voxel_grid.get_voxels()

        for vox in voxels:
            #np.append(voxelret["centers"], np.array([voxel_grid.get_voxel_center_coordinate(vox.grid_index)]), axis=0)
            #np.append(voxelret["colors"], np.array([vox.color]), axis=0)
            voxelret["centers"].append(voxel_grid.get_voxel_center_coordinate(vox.grid_index))
            voxelret["colors"].append(vox.color)

        voxelret["centers"] = np.asarray(voxelret["centers"])
        voxelret["colors"] = np.asarray(voxelret["colors"])
        voxelret["scale"] = scale_f * sc_f

        return voxelret
