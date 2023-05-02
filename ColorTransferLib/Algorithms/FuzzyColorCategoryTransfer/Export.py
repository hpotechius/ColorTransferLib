import numpy as np
import open3d as o3d



class Export():
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------ 
    # def __write_convex_hull_mesh(colors, shape, path, color, color_space="LAB"):
    #     if color_space == "RGB":
    #         ex = np.asarray(colors)[:, np.newaxis]
    #         cex = cv2.cvtColor(ex, cv2.COLOR_Lab2RGB)
    #         mesh, validity = FCCT.__calc_convex_hull(cex.squeeze())
    #     else:
    #         mesh, validity = FCCT.__calc_convex_hull(colors)

    #     if validity:
    #         colors = np.full(shape, color)
    #         mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    #         o3d.io.write_triangle_mesh(filename=path, 
    #                                 mesh=mesh, 
    #                                 write_ascii=True,
    #                                 write_vertex_normals=False,
    #                                 write_vertex_colors=True,
    #                                 write_triangle_uvs=False)
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------ 
    def write_convex_hull_mesh(mesh, path, color):
        num_vert = np.asarray(mesh.vertices).shape[0]

        colors = np.tile(color, (num_vert,1))
        #colors = np.full(num_vert, color)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_triangle_mesh(filename=path, 
                                   mesh=mesh, 
                                   write_ascii=True,
                                   write_vertex_normals=False,
                                   write_vertex_colors=True,
                                   write_triangle_uvs=False)
        
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def write_colors_as_PC(colors, rgb_colors, path):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(colors)
        pc.colors = o3d.utility.Vector3dVector(rgb_colors)
        o3d.io.write_point_cloud(path, pc)